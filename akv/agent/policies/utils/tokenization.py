"""Optional local token counting from the final request's rendering inputs."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from ...types import ChatMessageTokenLayout, ChatTokenCounter, JsonDict
from .models import load_counter


class ModelTokenCounter:
    """Load the model's counter and count the final request's messages and tools.

    Every implementation follows ChatTokenCounter. Model selection and shared
    rendering details stay behind this interface.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._counter = load_counter(model_path)
        self._count_lock = threading.Lock()

    def message_layout(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
        chat_template_kwargs: JsonDict | None = None,
    ) -> ChatMessageTokenLayout:
        if not messages:
            raise ValueError("Token counting requires nonempty messages")
        if api_style not in {"minisglang", "sglang"}:
            raise ValueError("Token counting requires minisglang or sglang rendering")
        with self._count_lock:
            return self._counter.message_layout(
                deepcopy(list(messages)),
                deepcopy(list(tools)),
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
                api_style=api_style,
                chat_template_kwargs=deepcopy(chat_template_kwargs),
            )


__all__ = ["ChatMessageTokenLayout", "ChatTokenCounter", "ModelTokenCounter"]
