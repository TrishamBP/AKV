"""Position-compaction decisions, independent of Drop triggers and selectors."""

from dataclasses import dataclass

from ..types import ChatTokenCounter
from .tools import CacheTool, tool_actions
from .utils.common import check_schedule, positive_integer, scheduled
from .utils.context import PolicyContext


@dataclass(frozen=True)
class TurnReposition:
    after: int = 1
    every: int | None = 1

    def __post_init__(self) -> None:
        check_schedule(self.after, self.every)

    def __call__(self, context: PolicyContext, selected: set[int]) -> bool:
        return context.new_turn and scheduled(context.request.call_index, self.after, self.every)


@dataclass(frozen=True)
class TokenReposition:
    threshold: int
    counter: ChatTokenCounter

    def __post_init__(self) -> None:
        positive_integer(self.threshold, "threshold")

    def __call__(self, context: PolicyContext, selected: set[int]) -> bool:
        last = max(
            (op.message_id for op in context.cache_operation_history.operations if op.reposition), default=-1
        )
        compacted = {
            i
            for op in context.cache_operation_history.operations
            if op.message_id <= last
            for i in op.drop_message_ids
        }
        compacted_tokens = context.tokens(self.counter, compacted)
        return (
            context.layout(self.counter).prompt_tokens - compacted_tokens > self.threshold
            and context.tokens(self.counter, context.hidden | selected) > compacted_tokens
        )


@dataclass(frozen=True)
class ModelReposition:
    tool: CacheTool

    def __call__(self, context: PolicyContext, selected: set[int]) -> bool:
        return tool_actions(context, self.tool).reposition
