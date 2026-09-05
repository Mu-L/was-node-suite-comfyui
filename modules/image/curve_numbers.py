"""Reading a drawn curve back out as plain numbers over a range.

A curve is control points on a 0-255 scale, written as ``0,0;128,200;255,255``.
:func:`read` walks ``minimum`` to ``maximum`` by ``step`` and answers ``(positions, values)``
on that same range.
"""

from __future__ import annotations

import numpy as np

from ..util.numbers import whole_steps
from . import curves

__all__ = [
    "MAX_DECIMALS",
    "MAX_RANGE",
    "MAX_VALUES",
    "MIN_STEP",
    "count_of",
    "points_of",
    "read",
    "text_of",
]

#: How far a range may reach in either direction.
MAX_RANGE = 1e9

#: The smallest increment a range may be walked by.
MIN_STEP = 1e-6

#: How many values one walk may produce.
MAX_VALUES = 4096

#: How many decimal places a value may be rounded to.
MAX_DECIMALS = 12


def points_of(text: str) -> tuple[tuple[int, int], ...]:
    """The control points a curve widget's text holds.

    Args:
        text: ``0,0;128,200;255,255`` on a 0-255 scale, or the
            ``rgb:0,0;255,255|r:...`` an Image Curves node stores, whose composite curve is
            the one read.

    Returns:
        The points, ``(input, output)``, sorted by input. Empty and malformed text answer
        the straight line.
    """
    body = str(text or "").strip()
    if not body:
        return curves.DEFAULT_POINTS
    named = any(
        block.partition(curves.NAME)[0].strip().lower() in curves.CHANNELS and curves.NAME in block
        for block in body.split(curves.BLOCK)
    )
    if not named:
        body = f"{curves.CHANNELS[0]}{curves.NAME}{body}"
    return curves.parse(body)[curves.CHANNELS[0]]


def count_of(minimum: float, maximum: float, step: float) -> int:
    """How many values a walk of a range produces.

    Args:
        minimum: Where the walk starts.
        maximum: Where it stops, at or above ``minimum``.
        step: The increment, clamped to :data:`MIN_STEP`.

    Returns:
        One for every whole step inside the range, plus the one at ``minimum``.
        ``maximum`` is counted only where the step divides the range exactly.
    """
    span = max(0.0, float(maximum) - float(minimum))
    return whole_steps(span, max(abs(float(step)), MIN_STEP)) + 1


def read(
    text: str,
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.1,
    decimals: int = 6,
) -> tuple[list[float], list[float]]:
    """A curve as two lists of numbers over a range.

    Args:
        text: The curve widget's text, as :func:`points_of` reads it.
        minimum: Where the walk starts, and the value a curve output of 0 answers.
        maximum: Where it stops, and the value a curve output of 1 answers.
        step: The increment along the range, clamped to :data:`MIN_STEP`.
        decimals: Places each number is rounded to, clamped to 0 and :data:`MAX_DECIMALS`.

    Returns:
        ``(positions, values)``, the same length, in rising order of position. A position is
        a point along the range, and its value is the curve read at that point, on the same
        range.
    """
    low = float(minimum)
    high = max(low, float(maximum))
    size = max(abs(float(step)), MIN_STEP)
    count = count_of(low, high, size)
    span = high - low

    positions = np.minimum(low + size * np.arange(count, dtype=np.float64), high)
    fractions = (positions - low) / span if span > 0 else np.zeros(count, dtype=np.float64)
    table = curves.channel_lut(points_of(text))
    outputs = np.interp(
        fractions * curves.MAX_LEVEL, np.arange(curves.LUT_SIZE, dtype=np.float64), table
    )

    places = min(max(int(decimals), 0), MAX_DECIMALS)
    return _rounded(positions, places), _rounded(low + outputs * span, places)


def text_of(values: list[float]) -> str:
    """The numbers on one line, separated by commas, as ``0, 0.5, 1``."""
    return ", ".join(_written(value) for value in values)


def _rounded(values: np.ndarray, places: int) -> list[float]:
    """One array as rounded python floats."""
    return [round(float(value), places) for value in values]


def _written(value: float) -> str:
    """One number as the shortest text that reads back as the same number."""
    number = float(value)
    if number == 0:
        return "0"
    text = repr(number)
    return text[:-2] if text.endswith(".0") else text
