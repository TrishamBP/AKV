"""Facts and temporary calculations for one policy decision."""

from dataclasses import dataclass, field
from typing import Any

from ...types import CacheOperationHistory, ChatMessageTokenLayout, ChatTokenCounter, PolicyRequest


@dataclass
class PolicyContext:
    request: PolicyRequest
    cache_operation_history: CacheOperationHistory
    layouts: dict[int, ChatMessageTokenLayout] = field(default_factory=dict)
    values: dict[object, Any] = field(default_factory=dict)

    @property
    def hidden(self) -> set[int]:
        return {i for op in self.cache_operation_history.operations for i in op.drop_message_ids}

    @property
    def new_turn(self) -> bool:
        return any(
            i >= self.cache_operation_history.message_count for i in self.request.completed_call_message_ids
        )

    def layout(self, counter: ChatTokenCounter) -> ChatMessageTokenLayout:
        key = id(counter)
        if key not in self.layouts:
            request = self.request
            rendering = (
                {"chat_template_kwargs": request.chat_template_kwargs} if request.chat_template_kwargs else {}
            )
            self.layouts[key] = counter.message_layout(
                request.messages,
                request.tools,
                enable_thinking=request.enable_thinking,
                reasoning_effort=request.reasoning_effort,
                api_style=request.api_style,
                **rendering,
            )
        return self.layouts[key]

    def tokens(self, counter: ChatTokenCounter, ids: set[int]) -> int:
        layout = self.layout(counter)
        return sum(message_tokens(layout, i) for i in ids)


def message_tokens(layout: ChatMessageTokenLayout, message_id: int) -> int:
    if layout.message_token_counts is not None:
        return layout.message_token_counts[message_id]
    return layout.message_ends[message_id] - (layout.message_ends[message_id - 1] if message_id else 0)
