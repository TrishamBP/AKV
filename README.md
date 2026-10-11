<div align="center">
  <img src="resource/logo.gif" alt="AKV animated memory bot" width="160" />
  <h1>AKV</h1>
  <h3>Cache Engineering for Agents</h3>

  <p>
    <a href="https://fuvty.github.io/thinking_yard_project_page/projects/akv/">🌐 <b>Project Page</b></a>
  </p>
</div>

AKV brings KV cache engineering to LLM agents. It combines agent-side policies for deciding what to drop and when to compact positions with an SGLang-based inference system that executes those operations while reusing compatible KV cache.

- **`akv.agent`** is the algorithm component that decides which messages to drop from the KV cache and when to reposition retained tokens.
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

Apply the policy to a Chat Completions `payload` containing the full history, tools, and generation options, without existing cache instructions:

```python
from akv.agent import CachePolicy, CacheState

policy = CachePolicy(
    rolling_drop_target="tool",
    drop_after_call=1,
    drop_interval_calls=1,
    drop_target_responses=12,
    reposition_after_call=4,
    reposition_interval_calls=4,
)
state = CacheState()

prepared, next_state = policy.process(payload, state=state, api_style="sglang")
```

Send `prepared` as the request body. After a successful response, reuse `next_state` with the full updated history for the next request. See the [agent README](akv/agent/README.md) and [complete agent demo](examples/demo.py) for usage examples.

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

`CachePolicy` supports rules or agent-selected actions. `CachePolicy()` applies no automatic operations.

<table>
  <thead>
    <tr><th>Configuration</th><th>Parameter</th><th>Usage</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>What to drop</td>
      <td><code>rolling_drop_target</code></td>
      <td>Select <code>tool</code>, <code>assistant</code>, or <code>both</code> for deletion based on the retention window.</td>
    </tr>
    <tr>
      <td rowspan="3">When to drop tools</td>
      <td><code>drop_after_call=N</code></td>
      <td>First check after complete tool batch N.</td>
    </tr>
    <tr>
      <td><code>drop_interval_calls=M</code></td>
      <td>Repeat every M batches; requires <code>drop_after_call</code>. Omit for a single check.</td>
    </tr>
    <tr>
      <td><code>drop_trigger_tokens=T</code></td>
      <td>Check when the visible prompt exceeds T tokens.</td>
    </tr>
    <tr>
      <td rowspan="2">How much to retain</td>
      <td><code>drop_target_responses=K</code></td>
      <td>Window size in individual tool results.</td>
    </tr>
    <tr>
      <td><code>drop_target_tokens=U</code></td>
      <td>Drop oldest tool results toward a visible prompt length of U tokens.</td>
    </tr>
    <tr>
      <td rowspan="3">When to reposition</td>
      <td><code>reposition_after_call=N</code></td>
      <td>First reposition after complete tool batch N.</td>
    </tr>
    <tr>
      <td><code>reposition_interval_calls=M</code></td>
      <td>Repeat every M batches; requires <code>reposition_after_call</code>. Omit for a single operation.</td>
    </tr>
    <tr>
      <td><code>reposition_trigger_tokens=T</code></td>
      <td>Reposition when token position length exceeds T and dropped content leaves gaps.</td>
    </tr>
    <tr>
      <td>Token counting</td>
      <td><code>token_counter</code></td>
      <td>Required for token thresholds and token retention targets.</td>
    </tr>
  </tbody>
</table>

For each operation, choose either batch or token triggers. Choose one retention target; a token target must be below the token Drop trigger when both are set. The rolling example above checks after every batch and drops only when more than 12 tool results are retained.

Pass `options` to `process` to override rules for one request, or `actions` to supply agent-selected message IDs and a reposition decision. See the [agent README](akv/agent/README.md#policy-options) for examples and configuration details.

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
