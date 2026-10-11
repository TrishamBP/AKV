"""Shared policy, message, token-layout and HTTP result types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class BackendTraceContext:
    """Filesystem destination and identity for one physical backend call.

    ``artifact_dir`` must be unique for the call. The backend creates it and
    refuses to reuse an existing directory, which prevents concurrent calls or
    retries from overwriting one another. ``metadata`` is audit-only and is
    never included in the provider request.
    """

    artifact_dir: Path
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_openai(self) -> JsonDict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class GeneratedMessage:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    role: str = "assistant"
    reasoning_content: str | None = None

    def as_openai(self, *, include_reasoning_content: bool = False) -> JsonDict:
        message: JsonDict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.as_openai() for call in self.tool_calls]
        if include_reasoning_content and isinstance(self.reasoning_content, str):
            message["reasoning_content"] = self.reasoning_content
        return message


@dataclass(frozen=True)
class GenerationResult:
    message: GeneratedMessage
    finish_reason: str | None
    latency_seconds: float
    usage: JsonDict | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyRequest:
    """Canonical input facts for one drop/reposition decision."""

    messages: Sequence[JsonDict]
    call_index: int = 0
    tools: Sequence[JsonDict] = ()
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    api_style: str | None = None
    completed_call_message_ids: Sequence[int] = ()
    prompt_end_message_ids: Mapping[int, int] = field(default_factory=dict)
    chat_template_kwargs: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ContextDecision:
    """A wire-history replacement or cumulative model-side context instructions."""

    drop_message: dict[int, tuple[int, ...]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: tuple[JsonDict, ...] | None = None
    reposition: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CacheOperation:
    """Chosen drops and optional reposition at one message boundary."""

    message_id: int
    drop_message_ids: tuple[int, ...] = ()
    reposition: bool = False

    def __post_init__(self) -> None:
        ids = tuple(self.drop_message_ids)
        if type(self.message_id) is not int or self.message_id < 0:
            raise ValueError("Cache operation requires a nonnegative message ID")
        if any(type(i) is not int or not 0 <= i <= self.message_id for i in ids):
            raise ValueError("Cache operation cannot drop invalid or future message IDs")
        if len(ids) != len(set(ids)) or type(self.reposition) is not bool:
            raise ValueError("Cache operation requires unique IDs and a boolean reposition")
        if not ids and not self.reposition:
            raise ValueError("Cache operation must contain a drop or reposition")
        object.__setattr__(self, "drop_message_ids", tuple(sorted(ids)))


@dataclass(frozen=True)
class CacheState:
    """Caller-owned operation history and the last processed message prefix.

    This contains chosen instructions, not server acknowledgements or GPU data.
    Keep a separate state for each conversation branch. Policy changes are not
    recorded: each call supplies the rules or actions for its current boundary.
    """

    operations: tuple[CacheOperation, ...] = ()
    message_count: int = 0
    history_digest: str = ""
    context_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.message_count) is not int or self.message_count < 0:
            raise ValueError("Cache state message_count must be nonnegative")
        operations = tuple(self.operations)
        previous, dropped = -1, set()
        for operation in operations:
            if not isinstance(operation, CacheOperation):
                raise ValueError("Cache state requires CacheOperation entries")
            if not previous < operation.message_id < self.message_count:
                raise ValueError("Cache operations must have increasing processed boundaries")
            if dropped.intersection(operation.drop_message_ids):
                raise ValueError("Cache state cannot delete the same message twice")
            previous = operation.message_id
            dropped.update(operation.drop_message_ids)
        for digest in (self.history_digest, self.context_digest):
            valid = isinstance(digest, str) and (
                len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                if self.message_count
                else digest == ""
            )
            if not valid:
                raise ValueError("Cache state digests must match its processed prefix")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> JsonDict:
        """Return a private JSON-serializable snapshot."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheState:
        data = dict(value)
        data["operations"] = tuple(CacheOperation(**item) for item in data.get("operations", ()))
        return cls(**data)


@dataclass(frozen=True)
class ChatMessageTokenLayout:
    """Token ownership for final request messages and the generation prefix."""

    message_ends: tuple[int, ...]
    generation_tokens: int
    message_token_counts: tuple[int, ...] | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.message_ends[-1] + self.generation_tokens


class ChatTokenCounter(Protocol):
    def count(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
        chat_template_kwargs: JsonDict | None = None,
    ) -> int: ...

    def message_layout(
        self,
        messages: Sequence[JsonDict],
        tools: Sequence[JsonDict],
        *,
        enable_thinking: bool | None,
        reasoning_effort: str | None,
        api_style: str | None,
        chat_template_kwargs: JsonDict | None = None,
    ) -> ChatMessageTokenLayout: ...


class BackendError(RuntimeError):
    """The inference service returned invalid data or an HTTP error."""

    def __init__(
        self,
        message: str,
        *,
        response_body: bytes | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.response_body = response_body
        self.http_status = http_status


class ContextLengthError(BackendError):
    """The service rejected input length, output budget, or execution positions
    outside the model context window."""
