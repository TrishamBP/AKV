"""Compose independent Drop triggers, message selectors, and Reposition rules."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .policies.utils.context import PolicyContext
from .policies.utils.request import check_state, parse_actions, prefix_digest, prepare_payload, read_request
from .types import CacheAction, CacheOperation, CacheState, JsonDict, PolicyRequest

DropTrigger = Callable[[PolicyContext], bool]
MessageSelector = Callable[[PolicyContext], set[int]]
RepositionRule = Callable[[PolicyContext, set[int]], bool]


@dataclass(frozen=True, kw_only=True)
class CachePolicy:
    """Combine three callables; omitted components choose no operation.

    The caller owns history and cumulative state. Request parsing and validation
    use shared helpers; this class owns the decision and state update flow.
    """

    trigger: DropTrigger | None = None
    selector: MessageSelector | None = None
    reposition: RepositionRule | None = None

    def __post_init__(self) -> None:
        for name in ("trigger", "selector", "reposition"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")

    def current_actions(self, request: PolicyRequest, state: CacheState) -> CacheAction:
        context = PolicyContext(request, state)
        selected = set()
        if self.trigger is not None and self.trigger(context) and self.selector is not None:
            selected = self.selector(context)
        reposition = self.reposition(context, selected) if self.reposition is not None else False
        return CacheAction(tuple(selected), reposition)

    def process(
        self,
        payload: Mapping[str, Any],
        *,
        state: CacheState | None = None,
        options: Mapping[str, Any] | None = None,
        actions: CacheAction | Mapping[str, Any] | None = None,
        api_style: str | None = None,
        completed_call_message_ids: Sequence[int] | None = None,
    ) -> tuple[JsonDict, CacheState]:
        """Run the three decisions for new history and preserve earlier operations."""
        if actions is not None and options is not None:
            raise ValueError("Pass either component options or explicit actions, not both")
        current = replace(self, **options) if options is not None else self
        state = CacheState() if state is None else state
        request = read_request(
            payload,
            api_style=api_style,
            completed_call_message_ids=completed_call_message_ids,
        )
        context_digest = check_state(payload, request, state)
        size = len(request.messages)
        if actions is None:
            actions = current.current_actions(request, state) if size > state.message_count else CacheAction()
        hidden = {i for op in state.operations for i in op.drop_message_ids}
        selected, reposition = parse_actions(actions, size, hidden)
        if size > state.message_count:
            operations = state.operations
            if selected or reposition:
                operations = (*operations, CacheOperation(size - 1, tuple(selected), reposition))
            state = CacheState(operations, size, prefix_digest(request, size), context_digest)
        return prepare_payload(payload, state), state
