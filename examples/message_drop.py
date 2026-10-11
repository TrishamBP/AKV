"""Two equivalent ways to drop messages; send only one form per request."""

import requests

body = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 15 + 27?"},
        {"role": "assistant", "content": "42."},
        {"role": "user", "content": "Now multiply that result by 3."},
    ],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
    "reposition": [3],  # Compact positions after message 3.
}

# After message 3, drop the KV of messages 1 and 2 (zero-based indices).
for operation in (
    {"drop_message": {"3": [1, 2]}},
    {"drop_rule": {"type": "message_drop", "drop_messages": {"3": [1, 2]}}},
):
    response = requests.post(
        "http://localhost:30000/v1/chat/completions",
        json={**body, **operation},
        timeout=120,
    )
    response.raise_for_status()
    print(response.json())
