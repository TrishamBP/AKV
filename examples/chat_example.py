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
    "drop_message": {"3": [1, 2]},
    "reposition": [3],  # Compact positions after message 3.
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={**body},
    timeout=120,
)
response.raise_for_status()
print(response.json())
