"""Request parsing, continuity checks, and cumulative payload rendering."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ...types import CacheAction, CacheOperationHistory, JsonDict, PolicyRequest
from .messages import atomic_message_units


def check_cache_operation_history(
    payload: Mapping[str, Any], request: PolicyRequest, cache_operation_history: CacheOperationHistory
) -> str:
    """Validate continuity against the adopted prefix and rendering settings."""
    context_digest = digest(
        {
            **{
                key: payload.get(key)
                for key in ("model", "tools", "chat_template_kwargs", "reasoning_effort")
            },
            "api_style": request.api_style,
        }
    )
    if cache_operation_history.message_count and (
        len(request.messages) < cache_operation_history.message_count
        or prefix_digest(request, cache_operation_history.message_count)
        != cache_operation_history.history_digest
        or context_digest != cache_operation_history.context_digest
    ):
        raise ValueError(
            "Cache operation history requires the same history prefix, model and rendering settings"
        )
    return context_digest


def check_payload(payload: Mapping[str, Any]) -> None:
    if any(payload.get(name) is not None for name in ("drop_message", "drop_rule", "reposition")):
        raise ValueError("Pass a canonical request without existing context instructions")
    if "extra_body" in payload:
        raise ValueError("Pass the final HTTP payload, with SDK extra_body already merged")


def read_request(
    payload: Mapping[str, Any],
    *,
    api_style: str | None = None,
    completed_call_message_ids: Sequence[int] | None = None,
) -> PolicyRequest:
    """Read cache facts from the exact outgoing message order.

    Native tool batches provide their own boundaries. Other observation formats
    use caller-supplied historical boundary metadata.
    """
    check_payload(payload)
    messages = deepcopy(tuple(payload["messages"]))
    tools = deepcopy(tuple(payload.get("tools") or ()))
    template = payload.get("chat_template_kwargs") or {}
    if not messages:
        raise ValueError("Canonical history must be nonempty")
    units = atomic_message_units(messages)
    native = {unit[-1] for unit in units if len(unit) > 1}
    boundaries = (
        tuple(sorted(native)) if completed_call_message_ids is None else tuple(completed_call_message_ids)
    )
    if any(
        type(b) is not int
        or not 0 <= b < len(messages)
        or messages[b]["role"] not in ("tool", "user")
        or messages[b]["role"] == "tool"
        and b not in native
        for b in boundaries
    ) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Logical boundaries must be complete, ordered tool batches or user observations")
    if not native <= set(boundaries):
        raise ValueError("Logical boundaries must include every completed native tool batch")
    return PolicyRequest(
        call_index=len(boundaries),
        messages=messages,
        tools=tools,
        enable_thinking=template.get("enable_thinking"),
        reasoning_effort=payload.get("reasoning_effort"),
        chat_template_kwargs=deepcopy(template),
        api_style=api_style,
        completed_call_message_ids=boundaries,
        message_units=units,
    )


def prepare_payload(payload: Mapping[str, Any], cache_operation_history: CacheOperationHistory) -> JsonDict:
    """Render validated cumulative cache operations without changing any messages."""
    check_payload(payload)
    if cache_operation_history.message_count > len(payload["messages"]):
        raise ValueError("Cache operation history cannot reference messages beyond the current payload")
    prepared = deepcopy(dict(payload))
    schedule = {
        str(op.message_id): list(op.drop_message_ids)
        for op in cache_operation_history.operations
        if op.drop_message_ids
    }
    reposition = [op.message_id for op in cache_operation_history.operations if op.reposition]
    if schedule:
        prepared["drop_message"] = schedule
    if reposition:
        prepared["reposition"] = reposition
    return prepared


def parse_actions(
    actions: CacheAction | Mapping[str, Any], size: int, hidden: set[int]
) -> tuple[list[int], bool]:
    """Normalize actions, then check their IDs against the current request."""
    if not isinstance(actions, CacheAction):
        if not isinstance(actions, Mapping) or set(actions) - {"drop_message_ids", "reposition"}:
            raise ValueError("Actions must contain only drop_message_ids and reposition")
        ids = actions.get("drop_message_ids", ())
        if not isinstance(ids, (tuple, list)):
            raise ValueError("Actions require a list or tuple of message IDs")
        actions = CacheAction(tuple(ids), actions.get("reposition", False))
    if any(i >= size for i in actions.drop_message_ids):
        raise ValueError("Actions require valid current message IDs")
    return [i for i in actions.drop_message_ids if i not in hidden], actions.reposition


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prefix_digest(request: PolicyRequest, size: int) -> str:
    return digest(
        {
            "messages": request.messages[:size],
            "boundaries": [i for i in request.completed_call_message_ids if i < size],
        }
    )
