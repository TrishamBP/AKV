# AKV System

AKV System brings cache engineering to SGLang, letting agents drop selected KV states and reposition retained ones through a stateless API while reusing corresponding cache to reduce long-horizon reasoning costs.

## Environment Setup

1. Install *Rust* and *Cargo* if they are not already available:

```bash
curl -o install-rust.sh https://sh.rustup.rs
sh install-rust.sh -y
source "$HOME/.cargo/env"
```

2. From the AKV repository root, create a uv environment and install the agent and system in editable mode:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Drop/Repos additionally compiles Context CPU kernels during startup warmup. Keep a working C++20 compiler and a writable compilation cache available.

## Quick Start

AKVSGLang extends SGLang's OpenAI-compatible chat completion API.

1. Launch the server:

```bash
python -m akv.system.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --port 8000
```

<details>
<summary>Hints</summary>

Some arguments you might use for launching the system:

| Server argument | Usage |
| --- | --- |
| `--page-size` | Only support page-size = 1 (default) for Drop/Repos. |
| `--attention-backend` | Use `flashinfer` (default), `triton`, or `fa3`, subject to the model and GPU's native support. Both prefill and decode backends must be supported. |
| `--disable-drop-aware-eviction` | New optional flag. Drop-aware eviction is enabled by default; pass this flag to disable it. |

For PD deployment, follow [SGLang's PD guide](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) and use the same system revision on both workers, with `--disaggregation-transfer-backend mooncake` (default). AKV PD currently supports only Mooncake and excludes staging buffers, KV offload, and KV checksum mode. Here is an example to quick start:

```bash
MC_FORCE_TCP=1 CUDA_VISIBLE_DEVICES=0 \
python -m akv.system.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --port 30000 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \

MC_FORCE_TCP=1 CUDA_VISIBLE_DEVICES=1 \
python -m akv.system.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --port 30001 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \

python -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port 8000 \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 8998 \
  --decode http://127.0.0.1:30001
```
</details>


2. Send a request with cache operations:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The capical of France is Paris."},
            {"role": "assistant", "content": "I have read the note."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "drop_message": {"3": [1, 2]},
        "reposition": [3],
        "temperature": 0,
        "max_tokens": 2048
    },
    timeout=120,
)
response.raise_for_status()
print(response.json())
```

Message indices are **zero-based**. Here, after message `3`, the server drops message `1, 2`'s KV and compacts the retained positions before processing message `4`. Drop alone preserves the surviving positions; `reposition` explicitly removes the gaps.

| Request field | Usage |
| --- | --- |
| `drop_message` | Map a trigger message index to the message indices whose KV should be dropped **after** that trigger. |
| `reposition` | List message indices after which to compact retained KV positions. Optional; independent of `drop_message`. |
| `drop_rule` | Alternative structured rule: `message_drop` or `thinking_drop`. Do not combine it with `drop_message`. (We will further release more drop rules, see how to use it in examples)|

Context requests report `cached_tokens`, `drop_skipped_tokens`, and `repos_tokens` in `response.usage.prompt_tokens_details`, including zero values.

## Supported Models

Drop/Reposition is currently enabled for these text-model architectures:

| Model family | Architecture |
| --- | --- |
| Qwen | `QWenLMHeadModel` |
| Qwen1.5, Qwen2, Qwen2.5 | `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` |
| Qwen3, including MoE variants, AgenticQwen | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` |
| GPT-OSS | `GptOssForCausalLM` |
| MiniMax-M2 family, including M2.7 | `MiniMaxM2ForCausalLM` |

TP is supported. Context requests currently exclude PP/CP/DCP, speculative decoding, LoRA, HiCache, external/session caches, and beam search. Unsupported Context requests are rejected at the API; model startup is not restricted by this Context model list.
