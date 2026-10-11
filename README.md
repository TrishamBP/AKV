<div align="center">
  <img src="resource/logo.gif" alt="AKV animated memory bot" width="160" />
  <h1>AKV</h1>
  <h3>Cache Engineering for Agents</h3>

  <p>
    <a href="https://fuvty.github.io/thinking_yard_project_page/projects/akv/">🌐 <b>Project Page</b></a>
  </p>
</div>

AKV brings KV cache engineering to LLM agents. It combines agent-side policies for deciding what to drop and when to compact positions with an SGLang-based inference system that executes those operations while reusing compatible KV cache.

- **`akv.agent`** derives Drop/Reposition instructions from the full conversation history and integrates with existing agent workflows.
- **`akv.system`** executes cache operations through SGLang's OpenAI-compatible Chat Completions API.

## Environment Setup

1. Install Rust and Cargo if they are not already available:

```bash
curl -o install-rust.sh https://sh.rustup.rs
sh install-rust.sh -y
source "$HOME/.cargo/env"
```

2. Create the `akv` environment and install the agent and system together:

```bash
git clone https://github.com/thu-nics/AKV.git
cd AKV
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

The environment is stored in `.venv` to avoid conflicting with the `akv/` source directory. See the [system README](akv/system/README.md) and the [agent README](akv/agent/README.md) for dependencies.

## Quick Start

1. Launch the system

```bash
python -m akv.system.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --port 8000
```

For additional launch options, see the [system README](akv/system/README.md).

2. Send a simple request

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
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

Message indices are **zero-based**. After processing message 3, the server drops the KV of messages 1 and 2 and compacts the retained positions before generating the response.

3. Apply an AKV policy

Prepare a Chat Completions request, then apply the policy before sending it:

```python
from akv.agent import AKVPolicy, adapt_request, apply_context

policy = AKVPolicy(
    keep_tool_responses=12,
    reposition_after_call=4,
    reposition_interval_calls=4,
)


def prepare_request(payload):
    policy_input = adapt_request(payload, api_style="sglang")
    decision = policy.decide(policy_input)
    return apply_context(payload, decision)
```

`payload` contains the full message history, tools, and generation options, without existing cache instructions. `apply_context` returns a copy with cumulative `drop_message` and `reposition` instructions. Send the prepared request without changing the message order; the caller manages history and tool execution. See the [agent README](akv/agent/README.md) for client integration.

## Usage

### System: Drop and Reposition

| Request field | Usage |
| --- | --- |
| `drop_message` | Maps a trigger message index to the message indices whose KV should be dropped after that trigger. |
| `drop_rule` | Selects a structured Drop rule. `message_drop` is equivalent to `drop_message`; `thinking_drop` removes historical assistant reasoning KV while retaining the final answer. Do not combine `drop_rule` with `drop_message`. |
| `reposition` | Lists message indices after which to compact retained KV positions. Optional and independent of Drop; indices must be unique and strictly increasing. |

Drop alone preserves the surviving positions; Reposition removes the gaps. Thinking Drop acts on historical reasoning and does not disable thinking generation for the new reply.

Context requests report `cached_tokens`, `drop_skipped_tokens`, and `repos_tokens` in `usage.prompt_tokens_details`, including zero values. See the [system README](akv/system/README.md) for rule details and usage accounting.

### Agent: policy configuration

`AKVPolicy` uses a rolling window by default. It counts individual tool responses for retention and complete tool batches for call-based triggers.

| Option | Usage |
| --- | --- |
| `keep_tool_responses` | Number of recent individual tool responses to retain. |
| `rolling_drop_target` | Select `tool`, `assistant`, or `both` for rolling deletion. |
| `reposition_after_call` | First complete tool batch to trigger repositioning. |
| `reposition_interval_calls` | Repeat interval for repositioning. |
| `reposition_trigger_tokens` | Trigger repositioning by token position length instead of batch count. |
| `drop_after_call`, `drop_interval_calls` | Schedule drops by complete tool batches. |
| `drop_trigger_tokens`, `drop_trigger_responses` | Trigger drops by visible prompt length or tool-response count. |
| `drop_target_tokens` | Target visible prompt length after a triggered drop. |
| `token_counter` | Counter used by token-based policies. |

Drop trigger modes are mutually exclusive. Repositioning requires `kv_drop`; token thresholds and targets require a matching counter. See the [agent README](akv/agent/README.md) for configuration rules and token counting.

## Supported Models

Drop/Reposition is currently enabled for these text-model architectures:

| Model family | Architecture |
| --- | --- |
| Qwen | `QWenLMHeadModel` |
| Qwen1.5, Qwen2, Qwen2.5 | `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` |
| Qwen3, including MoE variants and AgenticQwen | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` |
| GPT-OSS | `GptOssForCausalLM` |
| MiniMax-M2 family, including M2.7 | `MiniMaxM2ForCausalLM` |

See the [system README](akv/system/README.md) for runtime requirements and compatibility details.
