"""A minimal HTTP/SSE agent loop with fixed or changing cache rules.

The default CLI case keeps one tool result for the first two completed batches,
then keeps two. Use --case fixed for the original keep-one rule.
"""

import argparse
import json
from pathlib import Path

from akv.agent import CachePolicy, CacheState
from akv.agent.backends import send_chat_request
from akv.agent.types import BackendTraceContext, ToolCall

RECORDS = {
    "parcel-104": {"note": "Parcel handed to the regional carrier.", "next_record": "route-B7"},
    "route-B7": {"note": "Carrier transferred the parcel to the city depot.", "next_record": "depot-C2"},
    "depot-C2": {"note": "The parcel is out for final delivery.", "next_record": "receipt-F9"},
    "receipt-F9": {"delivery_location": "North Library, locker 17", "next_record": None},
}
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "read_record",
            "description": "Read one delivery record. Follow next_record until it is null.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
        },
    },
)


def execute_tool(call: ToolCall) -> str:
    """Tool dispatch belongs to this demo, not to AKV or the model server."""
    try:
        if call.name != "read_record":
            raise ValueError(f"Unknown tool: {call.name}")
        args = json.loads(call.arguments)
        if not isinstance(args, dict) or set(args) != {"record_id"}:
            raise ValueError("Expected exactly one argument: record_id")
        record_id = args["record_id"]
        if not isinstance(record_id, str) or record_id not in RECORDS:
            raise ValueError("Unknown record_id")
        return json.dumps(RECORDS[record_id])
    except (ValueError, TypeError) as error:
        return json.dumps({"error": str(error)})


def run_agent(
    base_url: str,
    policy: CachePolicy,
    output: Path,
    *,
    model: str = "gpt-oss-120b",
    max_steps: int = 8,
    max_tokens: int = 2048,
    case: str = "switch",
) -> str:
    if max_steps <= 0 or max_tokens <= 0:
        raise ValueError("max_steps and max_tokens must be positive")
    if case not in {"fixed", "switch"}:
        raise ValueError("case must be fixed or switch")
    output.mkdir(parents=True, exist_ok=False)
    history = [
        {"role": "system", "content": "Use read_record to trace the parcel. Read one record at a time."},
        {
            "role": "user",
            "content": "Start at parcel-104. Follow next_record until null, then report the final delivery location.",
        },
    ]
    trace = {"case": case, "status": "incomplete", "messages": history, "steps": []}
    state = CacheState()
    completed_calls = 0
    try:
        for step in range(1, max_steps + 1):
            # 1. The harness converts its history/tools to the final chat payload.
            payload = {
                "model": model,
                "messages": history,
                "tools": list(TOOLS),
                "tool_choice": "auto",
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": max_tokens,
                "temperature": 1.0,
                "reasoning_effort": "medium",
                "separate_reasoning": True,
            }
            # 2. Apply this request's rules using the cumulative state.
            options = {"drop_target_responses": 2} if case == "switch" and completed_calls >= 3 else None
            target = options["drop_target_responses"] if options else policy.drop_target_responses
            prepared, next_state = policy.process(payload, state=state, options=options, api_style="sglang")
            state_snapshot = next_state.to_dict()
            new_operations = state_snapshot["operations"][len(state.operations) :]
            event = {
                "step": step,
                "completed_tool_batches": completed_calls,
                "policy_options": options,
                "drop_target_responses": target,
                "canonical_message_ids": list(range(len(history))),
                "drop_message": prepared.get("drop_message"),
                "reposition": prepared.get("reposition"),
                "new_operations": new_operations,
                "cache_state": state_snapshot,
            }
            trace["steps"].append(event)
            print(
                f"\nStep {step} (completed batches={completed_calls}, keep={target}): "
                f"drop={prepared.get('drop_message')}, reposition={prepared.get('reposition')}",
                flush=True,
            )
            print(f"  New cache operations: {json.dumps(new_operations)}", flush=True)
            # 3. Send unchanged; no message conversion is allowed after AKV.
            result = send_chat_request(
                base_url,
                prepared,
                trace_context=BackendTraceContext(output / "calls" / f"{step:02d}", {"step": step}),
            )
            state = next_state
            event.update(
                finish_reason=result.finish_reason, usage=result.usage, backend_metadata=result.metadata
            )
            history.append(result.message.as_openai(include_reasoning_content=True))
            if result.finish_reason == "length":
                raise RuntimeError(
                    "Demo stopped at the generation limit; use a new output and more max_tokens"
                )
            calls = result.message.tool_calls
            if not calls:
                if result.finish_reason != "stop" or not (result.message.content or "").strip():
                    raise RuntimeError("The model returned neither a complete answer nor tool calls")
                trace["status"] = "completed"
                print(f"Answer: {result.message.content}", flush=True)
                return result.message.content
            # 4. Execute every call and append every result before the next generation.
            for call in calls:
                content = execute_tool(call)
                history.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
                )
                print(f"  message {len(history) - 1}: {call.name}({call.arguments}) -> {content}", flush=True)
            completed_calls += 1
        raise RuntimeError("Demo reached max_steps without a final answer")
    except Exception as error:
        trace["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Includes original history even when old tool results are hidden from the model.
        (output / "trajectory.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--output", type=Path, default=Path("local/demo-01"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--case",
        choices=("fixed", "switch"),
        default="switch",
        help="switch: keep 1 for tool batches 1-2, then keep 2; fixed: always keep 1",
    )
    args = parser.parse_args()
    policy = CachePolicy(drop_target_responses=1, reposition_after_call=2, reposition_interval_calls=1)
    run_agent(
        args.base_url,
        policy,
        args.output,
        model=args.model,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        case=args.case,
    )


if __name__ == "__main__":
    main()
