"""Model-facing cache tools; the caller registers and executes their handlers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import CacheAction, CacheOperationHistory, JsonDict
from .utils.context import PolicyContext


class CacheTool(Protocol):
    """The harness executes; the policy only interprets successful receipts.

    Pass canonical history before the calling assistant message and the adopted
    cache operation history used to generate that call. Argument and receipt
    formats belong to each implementation. read_result must never execute the
    tool again.
    """

    name: str

    @property
    def schema(self) -> JsonDict: ...

    def execute(
        self,
        arguments: str | Mapping[str, Any],
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> JsonDict: ...

    def read_result(
        self,
        arguments: str | Mapping[str, Any],
        content: Any,
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> CacheAction | None: ...


class ToolArgumentError(ValueError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = {"code": code, "message": message, **details}


@dataclass(frozen=True)
class DropContextTool:
    """Schedule deletion of visible earlier tool results, without changing history."""

    name: str = "drop_context"
    protected_prefix_messages: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be nonempty")
        if type(self.protected_prefix_messages) is not int or self.protected_prefix_messages < 0:
            raise ValueError("protected_prefix_messages must be a nonnegative integer")

    @property
    def schema(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Schedule deletion of visible EARLIER tool results from the KV cache on the next "
                    "model request. Copy the canonical message_id supplied with each result; do not "
                    "use source or document IDs. Each ID deletes the entire tool result, including "
                    "all sources inside it; other results and assistant messages remain. Select only "
                    "information no longer needed. No summary is created, and deletion cannot be undone. "
                    "Any invalid ID rejects the entire call; use available_message_ids in the error "
                    "to correct it. Continue the original task after a successful call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {"type": "integer", "minimum": 0, "maximum": 9223372036854775807},
                            "description": "Canonical message IDs of visible earlier tool results.",
                        },
                        "reposition": {
                            "type": "boolean",
                            "description": "Also request position compaction, if enabled by the policy.",
                        },
                    },
                    "required": ["message_ids"],
                    "additionalProperties": False,
                },
            },
        }

    def available_ids(
        self, messages: Sequence[JsonDict], cache_operation_history: CacheOperationHistory
    ) -> list[int]:
        if cache_operation_history.message_count > len(messages):
            raise ValueError("Cache operation history extends beyond the history before this tool call")
        hidden = {i for op in cache_operation_history.operations for i in op.drop_message_ids}
        return [
            i
            for i, message in enumerate(messages)
            if i >= self.protected_prefix_messages and message["role"] == "tool" and i not in hidden
        ]

    def execute(
        self,
        arguments: str | Mapping[str, Any],
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> JsonDict:
        """Return a receipt; do not contact the server or claim deletion has occurred."""
        available = self.available_ids(messages, cache_operation_history)
        try:
            return scheduled_result(self.action(arguments, available))
        except ToolArgumentError as error:
            return {
                "ok": False,
                "status": "error",
                "error": error.details,
                "available_message_ids": available,
            }

    @staticmethod
    def action(arguments: str | Mapping[str, Any], available: Sequence[int]) -> CacheAction:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError as error:
            raise ToolArgumentError("invalid_json", "Arguments must be a JSON object") from error
        if (
            not isinstance(args, Mapping)
            or "message_ids" not in args
            or set(args) - {"message_ids", "reposition"}
        ):
            raise ToolArgumentError("invalid_arguments", "Expected message_ids and optional reposition only")
        ids = args["message_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 64:
            raise ToolArgumentError("invalid_message_ids", "message_ids must be an array of 1 to 64 IDs")
        reposition = args.get("reposition", False)
        if type(reposition) is not bool:
            raise ToolArgumentError("invalid_reposition", "reposition must be boolean")
        for index, message_id in enumerate(ids):
            if type(message_id) is not int or not 0 <= message_id <= 9223372036854775807:
                raise ToolArgumentError(
                    "invalid_message_id", "Each message ID must be a nonnegative int64", item_index=index
                )
            if message_id not in available:
                raise ToolArgumentError(
                    "unavailable_message_id",
                    "ID must identify a visible, unprotected earlier tool result",
                    item_index=index,
                    message_id=message_id,
                )
        return CacheAction(tuple(ids), reposition)

    def read_result(
        self,
        arguments: str | Mapping[str, Any],
        content: Any,
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> CacheAction | None:
        """Interpret a matching success receipt without executing the tool."""
        try:
            outcome = json.loads(content)
            action = self.action(arguments, self.available_ids(messages, cache_operation_history))
        except (ValueError, TypeError):
            return None
        return action if outcome == scheduled_result(action) else None


def scheduled_result(action: CacheAction) -> JsonDict:
    return {
        "ok": True,
        "status": "scheduled",
        "selected_message_ids": list(action.drop_message_ids),
        "reposition": action.reposition,
        "next_turn_instruction": "Continue the original task. Cache changes take effect on the next model request.",
    }


def tool_actions(context: PolicyContext, tool: CacheTool) -> CacheAction:
    """Read each new receipt once per request, shared by components using this tool."""
    key = ("tool_actions", id(tool))
    if key in context.values:
        return context.values[key]
    request, cache_operation_history = context.request, context.cache_operation_history
    selected, reposition = set(), False
    for unit in request.message_units:
        assistant, *results = unit
        if not results:
            continue
        calls = {call["id"]: call for call in request.messages[assistant]["tool_calls"]}
        for i in results:
            if i < cache_operation_history.message_count:
                continue
            result = request.messages[i]
            call = calls[result["tool_call_id"]]
            if call["function"]["name"] != tool.name:
                continue
            action = tool.read_result(
                call["function"].get("arguments", ""),
                result.get("content"),
                messages=request.messages[:assistant],
                cache_operation_history=cache_operation_history,
            )
            if action is None:
                continue
            if not isinstance(action, CacheAction):
                raise TypeError("Cache tool read_result must return CacheAction or None")
            if any(i >= assistant for i in action.drop_message_ids):
                raise ValueError("Actions require valid current message IDs")
            selected.update(action.drop_message_ids)
            reposition |= action.reposition
    context.values[key] = CacheAction(tuple(selected), reposition)
    return context.values[key]
