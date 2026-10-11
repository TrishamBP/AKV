# AKV Agent

AKV Agent is the algorithm component of AKV for managing the KV cache of LLM agents. Its configurable policies decide which messages to drop from the cache and when to reposition retained tokens.

## Environment Setup

Create a uv environment and install in editable mode:

```bash
uv venv --python 3.12 akvagent
source akvagent/bin/activate
uv pip install -e ./akv/agent
```

Optional: install token counting dependencies for token-based policies:

```bash
uv pip install -e "./akv/agent[tokenization]"
```

## Quick Start

Use your running server's URL and model name:

```python
from akv.agent import CachePolicy, CacheState
from akv.agent.backends import send_chat_request

payload = {
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Say hello."}],
    "stream": True,
    "max_completion_tokens": 2048,
}
policy = CachePolicy(
    drop_target_responses=1,
    reposition_after_call=2,
    reposition_interval_calls=1,
)
state = CacheState()

prepared, next_state = policy.process(payload, state=state, api_style="sglang")
result = send_chat_request("http://127.0.0.1:30000/v1", prepared)
state = next_state
print(result.message.content)
```

For a complete agent loop with tool calls and cache operations, run [the demo](../../examples/demo.py):

```bash
python examples/demo.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model gpt-oss-120b \
  --output local/demo-01
```

## Policy Options

`CachePolicy` controls KV Drop and Reposition through rules or agent-selected actions.

### What to drop

Set `rolling_drop_target` to choose which messages to remove from the KV cache:

| Value | Messages to drop |
| --- | --- |
| `"tool"` (default) | Tool results outside the retention window. |
| `"assistant"` | Assistant messages associated with expired tool batches. |
| `"both"` | Both tool results and their associated assistant messages. |

### 1. Managed By Harness

Configure Drop and Reposition independently. A batch is complete when all results from one assistant tool-calling message have returned.

#### Drop

For `tool` Drop, configure when to check and how much to retain.

**When to drop**

| Parameter | Behavior |
| --- | --- |
| `drop_after_call=N` | First check after batch N. |
| `drop_interval_calls=M` | Repeat the check every M batches; requires `drop_after_call`. Omit for a single check. |
| `drop_trigger_tokens=T` | Check when the visible prompt exceeds T tokens; requires `token_counter`. |

Choose either the batch parameters or the token trigger.

**How much to retain**

| Parameter | Retention target |
| --- | --- |
| `drop_target_responses=K` | Keep the latest K tool results. |
| `drop_target_tokens=U` | Drop oldest tool results toward a prompt length of U tokens; requires `token_counter`. |

Choose one target. With a token trigger and token target, U must be less than T.

A **rolling window** is a batch configuration that checks from the first batch, then every batch. For example, retain the latest 12 tool results:

```python
policy = CachePolicy(
    rolling_drop_target="tool",
    drop_after_call=1,
    drop_interval_calls=1,
    drop_target_responses=12,
)
```

Nothing is dropped until the number of retained tool results exceeds 12. Each check drops the excess oldest results.

#### Reposition

| Parameter | Behavior |
| --- | --- |
| `reposition_after_call=N` | First reposition after batch N. |
| `reposition_interval_calls=M` | Repeat every M batches; requires `reposition_after_call`. Omit for a single operation. |
| `reposition_trigger_tokens=T` | Reposition when the token position length exceeds T and dropped content leaves gaps; requires `token_counter`. |

Choose either the batch parameters or the token trigger. Set `reposition_interval_calls=1` to reposition after every batch from N onward.

Override rules for one request with `options`, reusing the same state:

```python
prepared, next_state = policy.process(
    payload, state=state, options={"drop_target_responses": 2}, api_style="sglang"
)
```

### 2. Managed By Agent

The agent can choose message IDs to drop and whether to reposition:

```python
prepared, next_state = CachePolicy().process(
    payload,
    state=state,
    actions={"drop_message_ids": [1, 2], "reposition": True},
    api_style="sglang",
)
```

IDs are zero-based indices in `payload["messages"]`. `actions` replaces automatic rules for that request; use it without `options`.

## Token Counting

New models need a tokenizer and chat template matching the server:

- For Jinja models, provide `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` (or a template in `tokenizer_config.json`).
- For custom counting, follow [models/](models/) and register the model's `model_type` in [models/__init__.py](models/__init__.py).

Load the model directory and pass the counter as `token_counter` in your policy:

```python
from akv.agent.tokenization import ModelTokenCounter

counter = ModelTokenCounter("/path/to/model")
```

## License

AKV is licensed under [Apache-2.0](../../LICENSE).
