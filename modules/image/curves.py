"""Tone curves: control points in, a mapped image out.

A curve is ``(input, output)`` points on a 0-255 scale, one list per key in
:data:`CHANNELS`. :func:`parse` and :func:`serialise` carry them to and from widget text;
:func:`apply` maps an image.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "CHANNELS",
    "DEFAULT_POINTS",
    "LUT_SIZE",
    "MAX_POINTS",
    "apply",
    "channel_lut",
    "identity",
    "is_identity",
    "parse",
    "serialise",
    "through",
]

#: Curve keys, in the order they are applied. ``rgb`` runs over all three channels first,
#: then each named channel runs over its own.
CHANNELS = ("rgb", "r", "g", "b")

#: Entries in a channel's lookup table, one per 8-bit input level.
LUT_SIZE = 256

#: Highest input or output value a control point may carry.
MAX_LEVEL = 255

#: Control points one channel may hold. Two are needed to describe a line.
MIN_POINTS = 2
MAX_POINTS = 16

#: The straight line, which leaves a channel untouched.
DEFAULT_POINTS = ((0, 0), (MAX_LEVEL, MAX_LEVEL))

#: Separates channel blocks, and the points inside one.
BLOCK = "|"
POINT = ";"
PAIR = ","
NAME = ":"


def identity() -> dict[str, tuple[tuple[int, int], ...]]:
    """A curve set that changes nothing.

    Returns:
        Every key in :data:`CHANNELS` mapped to :data:`DEFAULT_POINTS`.
    """
    return {name: DEFAULT_POINTS for name in CHANNELS}


def is_identity(curves: dict[str, tuple[tuple[int, int], ...]]) -> bool:
    """Whether a curve set would leave an image unchanged.

    Args:
        curves: Control points per channel.

    Returns:
        True when every channel is the straight line, so the caller can skip the mapping.
    """
    return all(tuple(curves.get(name, DEFAULT_POINTS)) == DEFAULT_POINTS for name in CHANNELS)


def _clean(points) -> tuple[tuple[int, int], ...]:
    """One channel's control points, sorted and made usable.

    Args:
        points: Pairs of numbers, in any order.

    Returns:
        Pairs clamped to 0-255, sorted by input, with duplicate inputs dropped and the last
        of a duplicate kept. Fewer than :data:`MIN_POINTS` survivors answer
        :data:`DEFAULT_POINTS`, and no more than :data:`MAX_POINTS` are returned.
    """
    seen: dict[int, int] = {}
    for pair in points:
        try:
            x, y = pair
            x = int(round(float(x)))
            y = int(round(float(y)))
        except (TypeError, ValueError):
            continue
        seen[min(max(x, 0), MAX_LEVEL)] = min(max(y, 0), MAX_LEVEL)
    if len(seen) < MIN_POINTS:
        return DEFAULT_POINTS
    ordered = tuple(sorted(seen.items()))[:MAX_POINTS]
    return ordered


def parse(text: str) -> dict[str, tuple[tuple[int, int], ...]]:
    """Read a widget's text back into control points.

    Args:
        text: ``rgb:0,0;255,255|r:...`` as :func:`serialise` writes it. Empty, malformed
            and unknown channel names all fall back rather than raise, so a hand-edited
            field cannot stop a run.

    Returns:
        Every key in :data:`CHANNELS`, each with at least :data:`MIN_POINTS` points.
    """
    curves = identity()
    for block in str(text or "").split(BLOCK):
        name, _, body = block.partition(NAME)
        name = name.strip().lower()
        if name not in CHANNELS or not body:
            continue
        pairs = []
        for item in body.split(POINT):
            x, _, y = item.partition(PAIR)
            pairs.append((x.strip(), y.strip()))
        curves[name] = _clean(pairs)
    return curves


def serialise(curves: dict[str, tuple[tuple[int, int], ...]]) -> str:
    """Write control points as the text a widget stores.

    Args:
        curves: Control points per channel.

    Returns:
        ``rgb:0,0;255,255|r:0,0;255,255|g:...|b:...``, every channel present so the text
        round-trips through :func:`parse` unchanged.
    """
    blocks = []
    for name in CHANNELS:
        points = _clean(curves.get(name, DEFAULT_POINTS))
        body = POINT.join(f"{x}{PAIR}{y}" for x, y in points)
        blocks.append(f"{name}{NAME}{body}")
    return BLOCK.join(blocks)


def _slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson tangents for a monotone cubic through the points.

    Args:
        xs: Input levels, strictly increasing.
        ys: Output levels.

    Returns:
        One tangent per point, limited so the spline cannot overshoot between two control
        points and cannot reverse direction inside a rising or falling run.
    """
    deltas = np.diff(ys) / np.diff(xs)
    tangents = np.empty_like(ys, dtype=np.float64)
    tangents[0] = deltas[0]
    tangents[-1] = deltas[-1]
    if len(deltas) > 1:
        tangents[1:-1] = (deltas[:-1] + deltas[1:]) / 2.0

    for index, delta in enumerate(deltas):
        if delta == 0:
            tangents[index] = 0.0
            tangents[index + 1] = 0.0
            continue
        alpha = tangents[index] / delta
        beta = tangents[index + 1] / delta
        size = alpha * alpha + beta * beta
        if size > 9.0:
            scale = 3.0 / np.sqrt(size)
            tangents[index] = scale * alpha * delta
            tangents[index + 1] = scale * beta * delta
    return tangents


def channel_lut(points) -> np.ndarray:
    """One channel's lookup table, as a monotone cubic through its control points.

    Args:
        points: Control points, as :func:`parse` answers them.

    Returns:
        :data:`LUT_SIZE` values in 0-1, one per input level, clipped to that range. Two
        points give a straight line, and the curve never overshoots between points.
    """
    cleaned = _clean(points)
    xs = np.array([x for x, _ in cleaned], dtype=np.float64)
    ys = np.array([y for _, y in cleaned], dtype=np.float64)
    grid = np.arange(LUT_SIZE, dtype=np.float64)

    if len(cleaned) == MIN_POINTS:
        out = np.interp(grid, xs, ys)
        return np.clip(out / MAX_LEVEL, 0.0, 1.0)

    tangents = _slopes(xs, ys)
    slot = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, len(xs) - 2)
    span = xs[slot + 1] - xs[slot]
    step = (grid - xs[slot]) / span
    step2 = step * step
    step3 = step2 * step

    out = (
        (2 * step3 - 3 * step2 + 1) * ys[slot]
        + (step3 - 2 * step2 + step) * span * tangents[slot]
        + (-2 * step3 + 3 * step2) * ys[slot + 1]
        + (step3 - step2) * span * tangents[slot + 1]
    )
    out = np.where(grid <= xs[0], ys[0], out)
    out = np.where(grid >= xs[-1], ys[-1], out)
    return np.clip(out / MAX_LEVEL, 0.0, 1.0)


def _mapped(values: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Values 0-1 read through a lookup table, interpolating between its entries.

    Args:
        values: Any shape, expected in 0-1 and clamped to it.
        table: :data:`LUT_SIZE` entries, matching ``values`` in dtype and device.

    Returns:
        The same shape as ``values``. Reading between two entries interpolates rather than
        rounding, so a 16-bit or float image keeps its precision.
    """
    scaled = torch.clamp(values, 0.0, 1.0) * (LUT_SIZE - 1)
    low = torch.floor(scaled)
    frac = scaled - low
    low = low.long().clamp_(0, LUT_SIZE - 1)
    high = (low + 1).clamp_(max=LUT_SIZE - 1)
    return torch.lerp(table[low], table[high], frac)


def through(image: torch.Tensor, table) -> torch.Tensor:
    """Map an image's three colour channels through one lookup table.

    Args:
        image: ``(..., 3)`` or ``(..., 4)`` in 0-1. A fourth channel is carried through
            untouched.
        table: :data:`LUT_SIZE` output levels in 0-1, one per input level.

    Returns:
        A new tensor the same shape, dtype and device as ``image``.
    """
    if image.shape[-1] < 3:
        return image.clone()
    out = image.clone()
    rgb = out[..., :3]
    dtype = rgb.dtype if rgb.is_floating_point() else torch.float32
    levels = torch.as_tensor(np.asarray(table, dtype=np.float64), dtype=dtype).to(rgb.device)
    out[..., :3] = _mapped(rgb.to(dtype), levels).to(out.dtype)
    return out


def apply(image: torch.Tensor, curves: dict[str, tuple[tuple[int, int], ...]]) -> torch.Tensor:
    """Map an image through a set of tone curves.

    Args:
        image: ``(..., 3)`` or ``(..., 4)`` in 0-1. A fourth channel is carried through
            untouched.
        curves: Control points per channel, as :func:`parse` answers them.

    Returns:
        A new tensor the same shape, dtype and device as ``image``. The ``rgb`` curve runs
        first over all three channels, then each named curve runs over its own.
    """
    if image.shape[-1] < 3:
        return image.clone()

    out = image.clone()
    rgb = out[..., :3]
    dtype = rgb.dtype if rgb.is_floating_point() else torch.float32

    composite = torch.as_tensor(channel_lut(curves.get("rgb", DEFAULT_POINTS)), dtype=dtype)
    composite = composite.to(rgb.device)
    if tuple(curves.get("rgb", DEFAULT_POINTS)) != DEFAULT_POINTS:
        rgb = _mapped(rgb.to(dtype), composite)

    for index, name in enumerate(("r", "g", "b")):
        points = curves.get(name, DEFAULT_POINTS)
        if tuple(points) == DEFAULT_POINTS:
            continue
        table = torch.as_tensor(channel_lut(points), dtype=dtype).to(rgb.device)
        rgb[..., index] = _mapped(rgb[..., index].to(dtype), table)

    out[..., :3] = rgb.to(out.dtype)
    return out
