<div align="center">
  <img src="resource/logo.gif" alt="AKV animated memory bot" width="160" />
  <h1>AKV</h1>
  <h3>Cache Engineering for Agents</h3>

  <p>
    <a href="https://fuvty.github.io/thinking_yard_project_page/projects/akv/">🌐 <b>Project Page</b></a>
  </p>
</div>

AKV brings KV cache engineering to LLM agents. It combines agent-side policies for deciding what to drop and when to compact positions with an SGLang-based inference system that executes those operations while reusing compatible KV cache.

- `akv.agent` is the algorithm component that decides which messages to drop from the KV cache and when to reposition retained tokens.
- `akv.system` executes cache operations through SGLang's OpenAI-compatible Chat Completions API.


## Environment Setup

### 1. Install Rust and Cargo if they are not already available

```bash
curl -o install-rust.sh https://sh.rustup.rs
sh install-rust.sh -y
source "$HOME/.cargo/env"
```

### 2. Create the `akv` environment and install the agent and system together

```bash
git clone https://github.com/thu-nics/AKV.git
cd AKV
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

The environment is stored in `.venv` to avoid conflicting with the `akv/` source directory. See the [system README](akv/system/README.md) and the [agent README](akv/agent/README.md) for dependencies.

## Quick Start

These examples use GPT-OSS-120B, served as `gpt-oss-120b`.

### 1. Launch the system

```bash
python -m akv.system.launch_server \
  --model-path /path/to/gpt-oss-120b \
  --reasoning-parser gpt-oss \
  --tool-call-parser gpt-oss \
  --tp-size 2 \
  --port 8000
```

For additional launch options, see the [system README](akv/system/README.md).

### 2. Apply an AKV policy

This payload contains two completed tool calls, with results at message indices 2 and 4. Apply a policy that keeps the latest batch and triggers Reposition:

```python
from akv.agent import CachePolicy
from akv.agent.policies import BatchTrigger, KeepBatches, BatchReposition

payload = {
    "model": "gpt-oss-120b",
    "messages": [
        {
            "role": "user", 
            "content": "Compare records A and B."
        },
        {
            "role": "assistant", 
            "tool_calls": [{
                "id": "a", "type": "function",
                "function": {"name": "read_record", "arguments": '{"record_id":"A"}'},
            }],
        },
        {
            "role": "tool", 
            "tool_call_id": "a", 
            "content": "Record A"
        },
        {
            "role": "assistant", 
            "tool_calls": [{
                "id": "b", "type": "function",
                "function": {"name": "read_record", "arguments": '{"record_id":"B"}'},
            }],
        },
        {
            "role": "tool", 
            "tool_call_id": "b", 
            "content": "Record B"
        },
    ],
    "stream": True,
    "max_completion_tokens": 2048,
}

policy = CachePolicy(
    trigger=BatchTrigger(),
    selector=KeepBatches(1),
    reposition=BatchReposition(after=2, every=2),
)
akv_payload, state = policy.process(payload, api_style="sglang")
```

Here, `payload` is OpenAI Compatible, and `akv_payload` preserves the entire original payload and adds:

```json
{
  "drop_message": {"4": [2]},
  "reposition": [4]
}
```

Where `messages` contains the conversation history. `drop_message` selects which messages to drop the model's active context and where; `reposition` compacts the positions of the remaining tokens.

### 3. Send request to system

After applying the policy, you can send `akv_payload` directly to the system as the request body. 

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=akv_payload,
    timeout=120,
)
response.raise_for_status()
```

<details>
<summary>Explanation to the request</summary>

We number the indices of messages (which is from 0), including system, user, assistant, and tool messages.
In the example above:

| Index | Role | Content |
| --- | --- | --- |
| 0 | user | Compare records A and B. |
| 1 | assistant | "tool_calls": `read_record(A)` |
| 2 | tool | Record A |
| 3 | assistant | "tool_calls": `read_record(B)` |
| 4 | tool | Record B |

| Field | Format | Example | Meaning |
| --- | --- | --- | --- |
| `messages` | OpenAI-Compatible messages | The four messages above | The full conversation history, including messages selected for dropping. |
| `drop_message` | `{"<trigger_message_index>" : [<message_to_drop_index>, ...]}` | `{"4": [2]}` | After processing message 4, drop the KV cache of messages 2  from the active context. |
| `reposition` | `[<trigger_message_index>, ...]` | `[4]` | After processing message 4, compact the remaining token positions to remove gaps. |

We can read the example as: **process message 4 $\to$ drop messages 2 $\to$ compact token positions $\to$ generate the reply.**

Hints:

- Do not remove messages 2 from the request yourself.
- `drop_message` and `reposition` are optional and independent. If at one message, both two operations are triggered, the system always drop the message **first**, then reposition.
- Repositioning does not change message indices. Keep the full history and its original numbering; the server applies the operations.

</details>


### Run a tool-calling Agent example

The [demo code](examples/demo.py) works as an agent, asking the model to find a parcel’s final delivery location. 

When the agent works, the model can call `drop_context` to drop `tool_call` that are no longer need. Before the next model request, AKV converts successful drop requests into `drop_message` operations. Reposition is scheduled after every three completed tool-call batches.

Run the demo with:

```bash
python examples/demo.py \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --output local/demo-tool-01
```

As the demo runs, you can watch the model follow the chain of delivery records, manage its context by discarding results it no longer needs, and ultimately identify the parcel's final delivery location.

## Integrate with an agent harness

Integrating AKV requires only one step: **process the final payload before sending request to the system**. After the harness serializes its history and tools, pass the payload to [CachePolicy.process](akv/agent/policy.py), then send the returned payload to AKV System.

**Optional: register cache tools.** If your policy uses a [CacheTool](akv/agent/policies/tools.py), register its schema and `execute` handler with the harness before starting the agent loop.

<details>
<summary>Workflow Diagram</summary>

```mermaid
flowchart LR
    register["Optional: register cache tool<br/>schema + execute handler"]
    payload["Harness builds payload"]
    policy["AKV policy.process"]
    system["AKV System"]
    response["Harness handles response<br/>and executes tools"]

    register -.-> payload
    payload --> policy --> system --> response
    response -->|Next request| payload
```

</details>

## Details

### Agent: policy composition

Compose three independent components with `CachePolicy`:


| Component | Responsibility | Built-in choices |
| --- | --- | --- |
| `trigger` | Decide when to select messages for deletion. | [BatchTrigger, TokenTrigger, ToolTrigger](akv/agent/policies/triggers.py) |
| `selector` | Decide which canonical messages to delete. | [KeepBatches, KeepTokens, ToolSelection](akv/agent/policies/selectors.py) |
| `reposition` | Decide when to compact retained positions. | [BatchReposition, TokenReposition, ToolReposition](akv/agent/policies/reposition.py) |


The built-in components support three approaches:

- **Batch-based:** schedule Drop and Reposition by completed tool batches and retain the latest batches. A batch includes all results from one assistant tool-calling message.
- **Token-based:** use token thresholds to trigger Drop or Reposition, and token budgets to select old results for removal. These components require a token counter matching the model.
- **Tool-based:** let the model select messages through a registered cache tool. The policy reads successful tool results to schedule Drop and, optionally, Reposition.

To customize a policy, supply your own `trigger`, `selector`, or `reposition` function to `CachePolicy`. Each function reads a `PolicyContext` supplied by AKV and returns its decision. The context provides the current request and earlier cache decisions, including:


| Context field | Meaning |
| --- | --- |
| `context.request.messages` | The full message list from the payload. Each message's zero-based index is its ID. |
| `context.hidden` | IDs already selected for deletion by earlier policy decisions. These messages remain in the payload. |
| `context.new_batch` | Whether a new complete tool batch has appeared since the previous processed request. |


The three functions have the following contracts:

| Function | Return value | When it runs |
| --- | --- | --- |
| `trigger(context)` | `True` to run the selector. | When the policy evaluates new history. |
| `selector(context)` | A `set[int]` of message IDs to delete. | Only when the trigger returns `True`. |
| `reposition(context, selected)` | `True` to compact retained positions.  | Independently of the trigger; `selected` is empty when no messages are selected. |


For example, this policy checks after each completed batch, keeps the latest visible tool result, and compacts positions whenever it deletes older results:

```python
from akv.agent import CachePolicy, PolicyContext


def after_tool_batch(context: PolicyContext) -> bool:
    return context.new_batch


def select_old_results(context: PolicyContext) -> set[int]:
    visible_results = [
        i for i, message in enumerate(context.request.messages)
        if message["role"] == "tool" and i not in context.hidden
    ]
    return set(visible_results[:-1])  # Keep the latest result.


def reposition_after_drop(context: PolicyContext, selected: set[int]) -> bool:
    return bool(selected)


policy = CachePolicy(
    trigger=after_tool_batch,
    selector=select_old_results,
    reposition=reposition_after_drop,
)
```

With the Quick Start payload, this policy selects message 2 for deletion, keeps message 4 and triggers reposition. It adds `drop_message={"4": [2]}` and `reposition=[4]` to the request.

### System: Drop and Reposition

AKV extends standard chat interface with three optional request fields for drop and reposition. An AKV policy can generate these fields, or you can write them directly, alongside `message` with `JSON` body.

| Field | How to write it | Meaning |
| --- | --- | --- |
| `drop_message` | `{"3": [1, 2]}` | After processing message 3, drop the KV cache of messages 1 and 2. The key specifies where; the list specifies what to drop. |
| `drop_rule` | `{"type": "message_drop", "drop_messages": {"3": [1, 2]}}` | Equivalent to the `drop_message` example above. **Use either form, not both.** |
| `drop_rule` | `{"type": "thinking_drop"}` | Drop each historical assistant message’s reasoning KV after processing that message, while keeping its final answer. Do not use it with `drop_message` or non-thinking mode. |
| `reposition` | `[3]` | After processing message 3, compact the remaining token positions to remove gaps. If Drop is also scheduled there, Drop runs first. |

For complete requests, see:
- [Message Drop](examples/message_drop.py): two equivalent ways to specify which messages to drop, combined with Reposition.
- [Thinking Drop](examples/thinking_drop.py): remove historical reasoning while retaining final answers, combined with Reposition.

Context requests report `cached_tokens`, `drop_skipped_tokens`, and `repos_tokens` in `usage.prompt_tokens_details`, including zero values. See the [system README](akv/system/README.md) for rule details and usage accounting.

## Supported Models

Drop/Reposition is currently enabled for these text-model architectures:


| Model family                                  | Architecture                              |
| --------------------------------------------- | ----------------------------------------- |
| Qwen                                          | `QWenLMHeadModel`                         |
| Qwen1.5, Qwen2, Qwen2.5                       | `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` |
| Qwen3, including MoE variants and AgenticQwen | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` |
| GPT-OSS                                       | `GptOssForCausalLM`                       |
| MiniMax-M2 family, including M2.7             | `MiniMaxM2ForCausalLM`                    |


See the [system README](akv/system/README.md) for runtime requirements and compatibility details.
