"""Validation and schedule arithmetic shared by decision components."""


def positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def check_schedule(after: int, every: int | None) -> None:
    positive_integer(after, "after")
    if every is not None:
        positive_integer(every, "every")


def scheduled(call: int, after: int, every: int | None) -> bool:
    """Use absolute batch counts; missed triggers are not replayed."""
    return call >= after and (call == after or every is not None and (call - after) % every == 0)
