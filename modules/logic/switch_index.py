"""Choosing one of a switch's growing inputs by number."""

from __future__ import annotations

__all__ = ["MAX_SLOTS", "OUT_OF_RANGE", "SLOT_NAMES", "resolve"]

#: Inputs an index switch grows to, matching the batching nodes beside it.
MAX_SLOTS = 26

#: The slot names, in the order they are drawn.
SLOT_NAMES: tuple[str, ...] = tuple(f"input_{letter}" for letter in "abcdefghijklmnopqrstuvwxyz")

#: What an index past either end does. ``empty`` is not offered, unlike the list readers: a
#: switch answers a value of the type it was wired, and there is no empty image or model to
#: hand back in its place.
OUT_OF_RANGE: tuple[str, ...] = ("wrap", "clamp", "error")


def resolve(index: int, length: int, out_of_range: str, node: str) -> int:
    """Turn a requested index into a connected slot's position.

    Args:
        index: The requested position, counting from 0. Negative counts back from the end.
        length: How many inputs are connected, which is one or more.
        out_of_range: One of :data:`OUT_OF_RANGE`.
        node: The node's name, for the message where the index is refused.

    Returns:
        A position from 0 to ``length - 1``.

    Raises:
        ValueError: The index is past either end and ``out_of_range`` is ``error``.
    """
    wanted = int(index)
    position = wanted + length if wanted < 0 else wanted
    if 0 <= position < length:
        return position
    if out_of_range == "wrap":
        return position % length
    if out_of_range == "clamp":
        return 0 if position < 0 else length - 1
    raise ValueError(
        f"{node} was asked for input {wanted} and {length} input(s) are connected, numbered "
        f"0 to {length - 1}. Connect another input, or set out_of_range to wrap or clamp"
    )
