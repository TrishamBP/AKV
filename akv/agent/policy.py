"""Cache policies with caller-owned cumulative operations."""

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from akv.agent.messages import atomic_message_units, history_tool_calls

from .reposition import RepositionConfig, message_tokens, reposition_events, token_layout
from .types import (
    CacheOperation,
    CacheState,
    ChatMessageTokenLayout,
    ChatTokenCounter,
    ContextDecision,
    JsonDict,
    PolicyRequest,
)


@dataclass(frozen=True)
class CachePolicy:
    drop_target_responses: int | None = None
    mode: str = "kv_drop"
    rolling_drop_target: str = "tool"
    reposition_after_call: int | None = None
    reposition_interval_calls: int | None = None
    reposition_trigger_tokens: int | None = None
    token_counter: ChatTokenCounter | None = None
    drop_after_call: int | None = None
    drop_interval_calls: int | None = None
    drop_trigger_tokens: int | None = None
    drop_target_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.drop_target_responses is not None and self.drop_target_tokens is not None:
            raise ValueError("Set exactly one of drop_target_responses or drop_target_tokens")
        if self.triggered and self.drop_target_responses is None and self.drop_target_tokens is None:
            raise ValueError("A drop trigger requires a drop target")
        if self.mode not in ("text_drop", "kv_drop"):
            raise ValueError("mode must be text_drop or kv_drop")
        if self.rolling_drop_target not in ("tool", "assistant", "both"):
            raise ValueError("rolling_drop_target must be tool, assistant, or both")
        for name in (
            "drop_target_responses",
            "drop_after_call",
            "drop_interval_calls",
            "drop_trigger_tokens",
            "drop_target_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if self.drop_after_call is not None and self.drop_trigger_tokens is not None:
            raise ValueError("Drop trigger modes are mutually exclusive")
        if self.drop_interval_calls is not None and self.drop_after_call is None:
            raise ValueError("drop_interval_calls requires drop_after_call")
        if self.triggered and (self.mode != "kv_drop" or self.rolling_drop_target != "tool"):
            raise ValueError("Triggered deletion requires tool-only kv_drop")
        if self.drop_target_tokens is not None:
            if not self.triggered:
                raise ValueError("drop_target_tokens requires a drop trigger")
            if self.drop_trigger_tokens is not None and self.drop_target_tokens >= self.drop_trigger_tokens:
                raise ValueError("drop_target_tokens must be below drop_trigger_tokens")
        if (
            self.drop_trigger_tokens is not None or self.drop_target_tokens is not None
        ) and self.token_counter is None:
            raise ValueError("Token drop requires a token counter")
        RepositionConfig(
            self.reposition_after_call,
            self.reposition_interval_calls,
            self.reposition_trigger_tokens,
            self.token_counter,
        )
        if self.mode != "kv_drop" and (
            self.reposition_after_call is not None or self.reposition_trigger_tokens is not None
        ):
            raise ValueError("Reposition requires kv_drop")

    @property
    def triggered(self) -> bool:
        return self.drop_after_call is not None or self.drop_trigger_tokens is not None

    def process(
        self,
        payload: Mapping[str, Any],
        *,
        state: CacheState | None = None,
        options: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        api_style: str | None = None,
        completed_call_message_ids: Sequence[int] | None = None,
        prompt_end_message_ids: Mapping[int, int] | None = None,
    ) -> tuple[JsonDict, CacheState]:
        """Prepare one request without changing payload, state, or this policy.

        Options override this object's defaults for this call only. Explicit
        actions replace automatic decisions; an empty mapping requests no new
        operation. Only the current prompt boundary can receive new operations.
        Reprocessing an adopted prefix returns the existing schedule unchanged.
        """
        if actions is not None and options is not None:
            raise ValueError("Pass either rule options or explicit actions, not both")
        current = replace(self, **options) if options is not None else self
        state = CacheState() if state is None else state
        request = self._read_request(
            payload,
            api_style=api_style,
            completed_call_message_ids=completed_call_message_ids,
            prompt_end_message_ids=prompt_end_message_ids,
        )
        size = len(request.messages)
        context = digest(
            {
                **{
                    key: payload.get(key)
                    for key in ("model", "tools", "chat_template_kwargs", "reasoning_effort")
                },
                "api_style": api_style,
                "mode": current.mode,
            }
        )
        if state.message_count and (
            size < state.message_count
            or prefix_digest(request, state.message_count) != state.history_digest
            or context != state.context_digest
        ):
            raise ValueError(
                "Cache state requires the same history prefix, model, rendering settings and mode"
            )
        hidden = {i for operation in state.operations for i in operation.drop_message_ids}
        selected, reposition = (), False
        if actions is not None:
            selected, reposition = parse_actions(actions, size, hidden)
            if reposition and current.mode != "kv_drop":
                raise ValueError("Reposition requires kv_drop")
        if size > state.message_count:
            if actions is None:
                selected, reposition = current.current_actions(request, state, hidden)
            if selected or reposition:
                operation = CacheOperation(size - 1, tuple(selected), reposition)
                operations = (*state.operations, operation)
            else:
                operations = state.operations
            state = CacheState(operations, size, prefix_digest(request, size), context)
        decision = context_decision(request, state, current.mode)
        return self._prepare_payload(payload, decision), state

    @staticmethod
    def _check_request(payload: Mapping[str, Any]) -> None:
        if any(payload.get(name) is not None for name in ("drop_message", "drop_rule", "reposition")):
            raise ValueError("Pass a canonical request without existing context instructions")
        if "extra_body" in payload:
            raise ValueError("Pass the final HTTP payload, with SDK extra_body already merged")

    def _read_request(
        self,
        payload: Mapping[str, Any],
        *,
        api_style: str | None = None,
        completed_call_message_ids: Sequence[int] | None = None,
        prompt_end_message_ids: Mapping[int, int] | None = None,
    ) -> PolicyRequest:
        """Extract policy facts from the exact message order sent to the server.

        Native tool-result batches supply their own boundaries. Other observation
        formats and controller suffixes require explicit harness metadata, including
        all historical boundaries. Nothing is remembered between calls. Rendering
        parameters come from the final JSON payload; exact token layouts remain the
        model-specific token counter's responsibility.
        """
        self._check_request(payload)
        messages = deepcopy(tuple(payload["messages"]))
        tools = deepcopy(tuple(payload.get("tools") or ()))
        template = payload.get("chat_template_kwargs") or {}
        thinking = template.get("enable_thinking")
        effort = payload.get("reasoning_effort")
        if not messages:
            raise ValueError("Canonical history must be nonempty")
        native = {unit[-1] for unit in atomic_message_units(messages, 0) if len(unit) > 1}
        boundaries = (
            tuple(sorted(native)) if completed_call_message_ids is None else tuple(completed_call_message_ids)
        )
        if any(
            type(b) is not int
            or not 0 <= b < len(messages)
            or messages[b]["role"] not in ("tool", "user")
            or messages[b]["role"] == "tool"
            and b not in native
            for b in boundaries
        ) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError("Logical boundaries must be complete, ordered tool batches or user observations")
        if not native <= set(boundaries):
            raise ValueError("Logical boundaries must include every completed native tool batch")
        suffixes = dict(prompt_end_message_ids or {})
        if any(
            type(boundary) is not int
            or type(end) is not int
            or boundary not in boundaries
            or end != boundary + 1
            or end >= len(messages)
            or messages[end]["role"] != "user"
            for boundary, end in suffixes.items()
        ):
            raise ValueError("Prompt suffix must be the user message immediately after a boundary")
        return PolicyRequest(
            call_index=len(boundaries),
            messages=messages,
            tools=tools,
            enable_thinking=thinking,
            reasoning_effort=effort,
            chat_template_kwargs=deepcopy(template),
            api_style=api_style,
            completed_call_message_ids=boundaries,
            prompt_end_message_ids=suffixes,
        )

    def _prepare_payload(self, payload: Mapping[str, Any], decision: ContextDecision) -> JsonDict:
        """Return a private final HTTP payload, retaining all unrelated fields.

        IDs refer to this payload's message order. Send the result without further
        message insertion, merging or reordering. The caller owns canonical history.
        """
        self._check_request(payload)
        prepared = deepcopy(dict(payload))
        if decision.messages is not None:
            prepared["messages"] = deepcopy(list(decision.messages))

        def message_id(value: int) -> int:
            if type(value) is not int or not 0 <= value < len(prepared["messages"]):
                raise ValueError("Context IDs must identify current payload messages")
            return value

        if decision.drop_message is not None:
            schedule = {}
            for trigger, ids in decision.drop_message.items():
                trigger = message_id(trigger)
                schedule[str(trigger)] = [message_id(value) for value in ids]
                if any(value > trigger for value in ids):
                    raise ValueError("Drop events cannot reference future messages")
            prepared["drop_message"] = schedule
        if decision.reposition is not None:
            values = [message_id(value) for value in decision.reposition]
            if not values or values != sorted(set(values)):
                raise ValueError("reposition must contain increasing unique message IDs")
            prepared["reposition"] = values
        return prepared

    def current_actions(
        self, request: PolicyRequest, state: CacheState, hidden: set[int]
    ) -> tuple[list[int], bool]:
        """Evaluate rules at the current boundary, using earlier chosen actions."""
        ids = [i for i, message in enumerate(request.messages) if message["role"] == "tool"]
        visible = [i for i in ids if i not in hidden]
        new_batch = any(i >= state.message_count for i in request.completed_call_message_ids)
        drop_due = (
            not self.triggered
            or new_batch
            and scheduled(request.call_index, self.drop_after_call, self.drop_interval_calls)
        )
        layouts = {}

        def layout() -> ChatMessageTokenLayout:
            return token_layout(request, self.token_counter, len(request.messages) - 1, layouts)

        def tokens(indices: Iterable[int]) -> int:
            return sum(message_tokens(layout(), i) for i in indices)

        if self.drop_trigger_tokens is not None:
            drop_due = layout().prompt_tokens - tokens(hidden) > self.drop_trigger_tokens
        selected = []
        if drop_due and self.drop_target_responses is not None:
            selected = visible[: -self.drop_target_responses]
        elif drop_due and self.drop_target_tokens is not None:
            remaining = layout().prompt_tokens - tokens(hidden)
            for message_id in visible:
                if remaining <= self.drop_target_tokens:
                    break
                selected.append(message_id)
                remaining -= message_tokens(layout(), message_id)
        if self.rolling_drop_target != "tool" and self.drop_target_responses is not None and drop_due:
            expired = hidden.intersection(ids) | set(selected)
            schedule = {len(request.messages) - 1: tuple(expired)}
            selected = list(
                rolling_message_schedule(request, schedule, self.rolling_drop_target).get(
                    len(request.messages) - 1, ()
                )
            )
        selected = sorted(set(selected) - hidden)
        reposition = new_batch and scheduled(
            request.call_index, self.reposition_after_call, self.reposition_interval_calls
        )
        if self.reposition_trigger_tokens is not None:
            last = max((op.message_id for op in state.operations if op.reposition), default=-1)
            compacted = {i for op in state.operations if op.message_id <= last for i in op.drop_message_ids}
            compacted_tokens = tokens(compacted)
            reposition = (
                layout().prompt_tokens - compacted_tokens > self.reposition_trigger_tokens
                and tokens(hidden | set(selected)) > compacted_tokens
            )
        return selected, reposition

    def decide(self, request: PolicyRequest) -> ContextDecision:
        """Reconstruct a fixed-rule schedule for a standalone history.

        This lower-level helper has no state. Use process for an evolving agent
        conversation, per-call rule changes, or explicit actions.
        """
        units = atomic_message_units(request.messages, 0)
        ids = tuple(i for i, message in enumerate(request.messages) if message["role"] == "tool")
        layouts = {}
        drop_metadata = {}
        if self.triggered:
            schedule, drop_metadata = triggered_drop_schedule(
                request,
                drop_target_responses=self.drop_target_responses,
                after_call=self.drop_after_call,
                interval_calls=self.drop_interval_calls,
                trigger_tokens=self.drop_trigger_tokens,
                token_counter=self.token_counter,
                target_tokens=self.drop_target_tokens,
                layouts=layouts,
            )
        elif self.drop_target_responses is not None:
            schedule = {trigger: (old,) for old, trigger in zip(ids, ids[self.drop_target_responses :])}
        else:
            schedule = {}
        if self.rolling_drop_target != "tool":
            schedule = rolling_message_schedule(request, schedule, self.rolling_drop_target)
        dropped = {i for values in schedule.values() for i in values}
        reposition, tokens = reposition_events(
            request,
            schedule,
            after_call=self.reposition_after_call,
            interval_calls=self.reposition_interval_calls,
            trigger_tokens=self.reposition_trigger_tokens,
            token_counter=self.token_counter,
            prompt_end_message_ids_by_boundary=request.prompt_end_message_ids,
            layouts=layouts,
        )
        metadata = {
            "policy": "akv",
            "implementation_version": "rolling-tool-drop-v2",
            "mode": self.mode,
            "drop_target_responses": self.drop_target_responses,
            "drop_target_tokens": self.drop_target_tokens,
            "rolling_drop_target": self.rolling_drop_target,
            "tool_response_count": len(ids),
            "retained_tool_response_count": sum(i not in dropped for i in ids),
            "dropped_tool_response_count": sum(i in dropped for i in ids),
            "dropped_assistant_message_count": sum(
                request.messages[i]["role"] == "assistant" for i in dropped
            ),
            "dropped_message_ids": sorted(dropped),
            "drop_message": schedule,
            "drop_event_count": len(schedule),
            "rolling_window_tool_response_ids": (
                list(ids[-self.drop_target_responses :]) if self.drop_target_responses is not None else None
            ),
            "reposition_after_call": self.reposition_after_call,
            "reposition_interval_calls": self.reposition_interval_calls,
            "reposition_requested": bool(reposition),
            "reposition_message_ids": list(reposition or ()),
            "reposition_event_count": len(reposition or ()),
            **tokens,
            **drop_metadata,
        }
        messages = None
        if self.mode == "text_drop" and dropped:
            visible = []
            for unit in units:
                calls = {
                    call["id"]: call["function"]["name"]
                    for call in history_tool_calls(request.messages[unit[0]])
                }
                for i in unit:
                    if i in dropped:
                        continue
                    message = deepcopy(request.messages[i])
                    # A retained result still needs its tool author when its assistant is gone.
                    if message["role"] == "tool" and unit[0] in dropped and not message.get("name"):
                        message["name"] = calls[message["tool_call_id"]]
                    visible.append(message)
            messages = tuple(visible)
        return ContextDecision(
            messages=messages,
            drop_message=(schedule or None) if self.mode == "kv_drop" else None,
            reposition=reposition,
            metadata=metadata,
        )


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prefix_digest(request: PolicyRequest, size: int) -> str:
    return digest(
        {
            "messages": request.messages[:size],
            "boundaries": [i for i in request.completed_call_message_ids if i < size],
            "suffixes": {b: end for b, end in request.prompt_end_message_ids.items() if b < size},
        }
    )


def scheduled(call: int, after: int | None, interval: int | None) -> bool:
    """Use absolute completed-batch counts; do not catch up missed triggers."""
    return (
        after is not None
        and call >= after
        and (call == after or interval is not None and (call - after) % interval == 0)
    )


def parse_actions(actions: Mapping[str, Any], size: int, hidden: set[int]) -> tuple[list[int], bool]:
    if not isinstance(actions, Mapping) or set(actions) - {"drop_message_ids", "reposition"}:
        raise ValueError("Actions must contain only drop_message_ids and reposition")
    ids = actions.get("drop_message_ids", ())
    reposition = actions.get("reposition", False)
    if not isinstance(ids, (tuple, list)) or any(type(i) is not int or not 0 <= i < size for i in ids):
        raise ValueError("Actions require valid current message IDs")
    if type(reposition) is not bool:
        raise ValueError("Action reposition must be boolean")
    return sorted(set(ids) - hidden), reposition


def context_decision(request: PolicyRequest, state: CacheState, mode: str) -> ContextDecision:
    schedule = {op.message_id: op.drop_message_ids for op in state.operations if op.drop_message_ids}
    reposition = tuple(op.message_id for op in state.operations if op.reposition) or None
    messages = None
    if mode == "text_drop" and schedule:
        hidden = {i for ids in schedule.values() for i in ids}
        visible = []
        for unit in atomic_message_units(request.messages, 0):
            calls = {
                call["id"]: call["function"]["name"] for call in history_tool_calls(request.messages[unit[0]])
            }
            for i in unit:
                if i in hidden:
                    continue
                message = deepcopy(request.messages[i])
                if message["role"] == "tool" and unit[0] in hidden and not message.get("name"):
                    message["name"] = calls[message["tool_call_id"]]
                visible.append(message)
        if not visible:
            raise ValueError("Text drop must retain at least one message")
        messages = tuple(visible)
    return ContextDecision(
        messages=messages,
        drop_message=(schedule or None) if mode == "kv_drop" else None,
        reposition=reposition,
    )


def rolling_message_schedule(
    request: PolicyRequest, tool_schedule: Mapping[int, Sequence[int]], target: str
) -> dict[int, tuple[int, ...]]:
    """Delete an assistant once all results from its batch have expired."""
    expired = {i: trigger for trigger, ids in tool_schedule.items() for i in ids}
    schedule = {trigger: list(ids) for trigger, ids in tool_schedule.items()} if target == "both" else {}
    pending = []
    for unit in atomic_message_units(request.messages, 0):
        assistant, *results = unit
        if request.messages[assistant]["role"] != "assistant":
            continue
        pending.append(assistant)
        if results:
            if all(i in expired for i in results):
                schedule.setdefault(max(expired[i] for i in results), []).extend(pending)
            pending = []
    return {trigger: tuple(sorted(ids)) for trigger, ids in sorted(schedule.items())}


def triggered_drop_schedule(
    request: PolicyRequest,
    *,
    drop_target_responses: int | None,
    after_call: int | None = None,
    interval_calls: int | None = None,
    trigger_tokens: int | None = None,
    token_counter: ChatTokenCounter | None = None,
    target_tokens: int | None = None,
    layouts: dict[int, ChatMessageTokenLayout] | None = None,
) -> tuple[dict[int, tuple[int, ...]], JsonDict]:
    """Drop oldest results toward a response-count or token target.

    Length is the exact prompt span minus all hidden tool-result spans, including
    system, assistant reasoning, tool declarations and the generation header.
    This attention-length metric is independent of reposition's position metric.
    A result batch is atomic for triggering, but each result is a deletion unit.
    Replay each boundary using its own request prefix so later messages cannot
    change past token decisions. Rendering inputs must remain unchanged.
    """
    boundaries = tuple(request.completed_call_message_ids)
    if (
        len(boundaries) != request.call_index
        or any(
            type(b) is not int
            or b <= 0
            or b >= len(request.messages)
            or request.messages[b].get("role") != "tool"
            for b in boundaries
        )
        or any(a >= b for a, b in itertools.pairwise(boundaries))
    ):
        raise ValueError("triggered drop requires complete ordered logical-call boundaries")
    needs_tokens = trigger_tokens is not None or target_tokens is not None
    if needs_tokens and token_counter is None:
        raise ValueError("token drop requires a token counter")
    layouts = {} if layouts is None else layouts
    schedule, events, hidden = {}, [], set()
    tool_ids = [i for i, message in enumerate(request.messages) if message.get("role") == "tool"]
    for call, boundary in enumerate(boundaries, 1):
        visible = [i for i in tool_ids if i <= boundary and i not in hidden]
        visible_tokens = None
        if needs_tokens:
            end = request.prompt_end_message_ids.get(boundary, boundary)
            layout = token_layout(request, token_counter, end, layouts)
            visible_tokens = layout.prompt_tokens - sum(message_tokens(layout, i) for i in hidden)
        if trigger_tokens is not None:
            triggered = visible_tokens > trigger_tokens
        else:
            triggered = (
                after_call is not None
                and call >= after_call
                and (
                    call == after_call
                    or interval_calls is not None
                    and (call - after_call) % interval_calls == 0
                )
            )
        selected = visible[:-drop_target_responses] if triggered and target_tokens is None else []
        if triggered and target_tokens is not None:
            selected = []
            remaining = visible_tokens
            for message_id in visible:
                if remaining <= target_tokens:
                    break
                selected.append(message_id)
                remaining -= message_tokens(layout, message_id)
        if selected:
            schedule[boundary] = tuple(selected)
            hidden.update(selected)
            events.append(
                {
                    "call": call,
                    "message_id": boundary,
                    "dropped_message_ids": selected,
                    "retained_message_ids": [i for i in visible if i not in hidden],
                    "visible_response_count_before": len(visible),
                    "visible_response_count_after": len(visible) - len(selected),
                    "visible_tokens_before": visible_tokens,
                    "visible_tokens_after": (
                        visible_tokens - sum(message_tokens(layout, i) for i in selected)
                    )
                    if needs_tokens
                    else None,
                    "target_tokens": target_tokens,
                    "target_reached": remaining <= target_tokens if target_tokens is not None else True,
                }
            )
    return schedule, {
        "drop_after_call": after_call,
        "drop_interval_calls": interval_calls,
        "drop_trigger_tokens": trigger_tokens,
        "drop_token_metric": "exact_prompt_minus_hidden_tool_spans" if needs_tokens else None,
        "drop_trigger_events": events,
    }
