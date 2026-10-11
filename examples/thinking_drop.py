"""Drop historical thinking KV while keeping the assistant's final answer."""

import requests

body = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 15 + 27?"},
        {
            "role": "assistant",
            "reasoning_content": "15 + 27 = 42.",
            "content": "42.",
        },
        {"role": "user", "content": "Now multiply that result by 3."},
    ],
    # thinking_drop removes KV entries for each historical assistant message's
    # reasoning_content after that message is processed (message 2 here).
    # It keeps the final answer and other messages. It does not disable thinking
    # generation for the new reply.
    "drop_rule": {"type": "thinking_drop"},
    "reposition": [3],
    "temperature": 0,
    "max_tokens": 64,
}

response = requests.post(
    "http://localhost:30000/v1/chat/completions", json=body, timeout=120
)
response.raise_for_status()
print(response.json())
