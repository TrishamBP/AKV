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
  --tp-size 2 \
  --reasoning-parser gpt-oss \
  --tool-call-parser gpt-oss \
  --port 8000
```

For additional launch options, see the [system README](akv/system/README.md).

### 2. Send a simple request

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The capital of France is Paris."},
            {"role": "assistant", "content": "I have read the note."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "drop_message": {"3": [1, 2]},
        "reposition": [3],
        "temperature": 0,
        "max_tokens": 2048,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json())
```

Here, `messages` contains the conversation history. `drop_message` selects which messages to drop the model's active context and where; `reposition` compacts the positions of the remaining tokens.

<details>
<summary>Explanation to the request</summary>

In this example, we number the indices of messages (which is from 0), including system, user, assistant, and tool messages.
In the example above:

| Index | Role | Content |
| --- | --- | --- |
| 0 | system | You are a helpful assistant. |
| 1 | user | The capital of France is Paris. |
| 2 | assistant | I have read the note. |
| 3 | user | What is the capital of France? |

| Field | Format | Example | Meaning |
| --- | --- | --- | --- |
| `messages` | OpenAI-Compatible messages | The four messages above | The full conversation history, including messages selected for dropping. |
| `drop_message` | `{"<trigger_message_index>" : [<message_to_drop_index>, ...]}` | `{"3": [1, 2]}` | After processing message 3, drop the KV cache of messages 1 and 2 from the active context. |
| `reposition` | `[<trigger_message_index>, ...]` | `[3]` | After processing message 3, compact the remaining token positions to remove gaps. |

We can read the example as: **process message 3 $\to$ drop messages 1 and 2 $\to$ compact token positions $\to$ generate the reply.**

Hints:

- Do not remove messages 1 and 2 from the request yourself.
- `drop_message` and `reposition` are optional and independent. If at one message, both two operations are triggered, the system always drop the message **first**, then reposition.
- Repositioning does not change message indices. Keep the full history and its original numbering; the server applies the operations.

</details>

### 3. Apply an AKV policy

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
prepared, state = policy.process(payload, api_style="sglang")
```

`prepared` preserves the entire original payload and adds:

```json
{
  "drop_message": {"4": [2]},
  "reposition": [4]
}
```

Send `prepared` as the request body. After message 4, AKV System drops the earlier result at message 2 and compacts the remaining positions.

### 4. Putting it all together

Run the [complete agent demo](examples/demo.py) with GPT-OSS-120B. Reposition is triggered every three turns, starting with the third turn.

```bash
python examples/demo.py \
  --base-url http://localhost:8000/v1 \
  --model gpt-oss-120b \
  --output local/demo-tool-01
```

As the demo runs, you can watch the model follow the chain of delivery records, manage its context by discarding results it no longer needs, and ultimately identify the parcel's final delivery location.

## Integrate with an agent harness

Integrate AKV in two steps:

1. **Apply the policy before sending each model request.** Once the harness has serialized its history and tools, pass the final payload to [CachePolicy.process](akv/agent/policy.py). Send the returned payload to AKV System. After a successful request, retain the returned AKV state and pass it to the next `process` call.
2. **Register a cache tool only for tool-based policies.** Before starting the agent loop, register the schema and `execute` handler of your [CacheTool](akv/agent/policies/tools.py) implementation alongside the harness's existing tools. Label tool results with their zero-based indices in the final serialized history so the model can select them for deletion.

```mermaid
flowchart TB
    register["Optional: register cache tool<br/>schema + execute handler"]
    payload["Harness builds payload"]
    policy["AKV policy.process"]
    system["AKV System"]
    response["Harness handles response<br/>and executes tools"]

    register -.-> payload
    payload --> policy --> system --> response
    response -->|Next request| payload
```


## Usage

### System: Drop and Reposition


| Request field | Usage |
| --- | --- |
| `drop_message` | A clean field to use `message_drop`.Maps a trigger message index to the message indices whose KV should be dropped after that trigger. |
| `drop_rule` | Selects a structured Drop rule: 1. `message_drop`, which is equivalent to `drop_message`; 2. `thinking_drop`, removes historical assistant reasoning KV while retaining the final answer. |
| `reposition` | Lists message indices after which to compact retained KV positions. Optional and independent of Drop; indices must be unique and strictly increasing. |


Drop alone preserves the surviving positions; Reposition removes the gaps. Thinking Drop acts on historical reasoning and does not disable thinking generation for the new reply. Note that: **do not combine `drop_rule` with `drop_message`**. See the details to use `drop_rule` in [examples](akv/examples).

Context requests report `cached_tokens`, `drop_skipped_tokens`, and `repos_tokens` in `usage.prompt_tokens_details`, including zero values. See the [system README](akv/system/README.md) for rule details and usage accounting.

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


Treat these inputs as read-only. The three functions have the following contracts:

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
