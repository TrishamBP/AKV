"""Canonical roles and complete atomic units shared by context policies."""

from collections.abc import Sequence

from akv.agent.types import JsonDict

CANONICAL_ROLES = ("system", "developer", "user", "assistant", "tool")


def atomic_message_units(messages: Sequence[JsonDict]) -> tuple[tuple[int, ...], ...]:
    """Validate history once and group each assistant with its complete tool results."""
    if any(message.get("role") not in CANONICAL_ROLES for message in messages):
        raise ValueError("history contains an unknown message role")
    units: list[tuple[int, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message["role"]
        if role == "tool":
            raise ValueError(f"history contains orphan tool message at {index}")
        tool_calls = message.get("tool_calls") or ()
        if role != "assistant" or not tool_calls:
            units.append((index,))
            index += 1
            continue
        expected = {
            call["id"]
            for call in tool_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]
        }
        if len(expected) != len(tool_calls):
            raise ValueError("assistant tool_calls must have unique nonempty string IDs")
        grouped = [index]
        seen: list[str] = []
        index += 1
        while index < len(messages) and messages[index]["role"] == "tool":
            tool_id = messages[index].get("tool_call_id")
            if tool_id not in expected:
                raise ValueError(f"tool result {index} does not match the preceding tool calls")
            grouped.append(index)
            seen.append(tool_id)
            index += 1
        if len(seen) != len(expected) or set(seen) != expected:
            raise ValueError("assistant tool-call batch has incomplete tool results")
        units.append(tuple(grouped))
    return tuple(units)
