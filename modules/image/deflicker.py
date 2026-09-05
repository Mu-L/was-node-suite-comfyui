"""Evening out exposure and colour drift across the frames of a sequence.

Images are float tensors shaped ``(frames, height, width, channels)`` in ``[0, 1]``. Each frame
is remapped by a monotone curve of its own.
"""

from __future__ import annotations

import torch

__all__ = ["BINS", "PROBES", "equalize"]

#: Levels the value range is measured in. 256 matches an 8 bit source and is fine enough that
#: the curve built from it is smooth once interpolated between bins.
BINS = 256

#: Points the quantile function is sampled at. Higher than :data:`BINS` so the inverse is
#: resolved more finely than the histogram that produced it.
PROBES = 512


def _weights(radius: int, device, dtype) -> torch.Tensor:
    """Gaussian temporal weights over ``2 * radius + 1`` frames, summing to one."""
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    # Two standard deviations across the half window, so the ends contribute without dominating.
    sigma = max(radius / 2.0, 1e-6)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def _quantiles(values: torch.Tensor, probes: int) -> torch.Tensor:
    """The quantile function of each row's distribution, sampled on a uniform grid.

    Args:
        values: ``(rows, bins)`` cumulative distributions, each ending at 1.
        probes: How many points to sample the inverse at.

    Returns:
        ``(rows, probes)`` levels in ``[0, 1]``.
    """
    rows, bins = values.shape
    grid = torch.linspace(0, 1, probes, device=values.device, dtype=values.dtype)
    grid = grid.expand(rows, probes).contiguous()
    # The first bin whose cumulative weight reaches each probability is that quantile's level.
    found = torch.searchsorted(values.contiguous(), grid).clamp(max=bins - 1)
    return found.to(values.dtype) / (bins - 1)


def _interpolate(table: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Read ``table`` at fractional indices, linearly between its entries.

    Args:
        table: ``(rows, entries)`` to read from.
        position: ``(rows, points)`` indices in ``[0, entries - 1]``.

    Returns:
        ``(rows, points)`` interpolated values.
    """
    entries = table.shape[1]
    lower = position.floor().clamp(0, entries - 1)
    upper = (lower + 1).clamp(max=entries - 1)
    fraction = (position - lower).clamp(0, 1)
    low = table.gather(1, lower.long())
    high = table.gather(1, upper.long())
    return low + (high - low) * fraction


def equalize(images, radius: int = 4, strength: float = 1.0, per_channel: bool = True):
    """Even out exposure across a sequence, one monotone curve per frame.

    Args:
        images: An ``IMAGE`` tensor, ``(frames, height, width, channels)`` in ``[0, 1]``.
        radius: Frames either side that a frame's reference is averaged over. 0 leaves the
            sequence alone, since a frame's reference would be itself.
        strength: How far each frame is moved towards its reference, 0 to 1.
        per_channel: Correct each channel on its own curve, which follows a colour cast as
            well as a brightness change. False derives one curve from luminance and applies it
            to every channel, which moves no colour that was not already moving.

    Returns:
        A tensor of the same shape and dtype. The input is handed back untouched for a single
        frame, a radius of 0, or a strength of 0, none of which have anything to do.
    """
    frames = int(images.shape[0])
    if frames < 2 or int(radius) < 1 or float(strength) <= 0.0:
        return images

    original = images
    working = images.to(torch.float32).clamp(0.0, 1.0)
    height, width, channels = (int(working.shape[1]), int(working.shape[2]),
                               int(working.shape[3]))

    # Rows are (frame, channel) pairs, or one row per frame when a single curve is shared.
    if per_channel:
        planes = working.permute(0, 3, 1, 2).reshape(frames * channels, height * width)
        lanes = channels
    else:
        # Rec. 709 luminance, which is what a viewer reads as the brightness of the frame.
        luma = (working[..., 0] * 0.2126 + working[..., 1] * 0.7152 + working[..., 2] * 0.0722
                if channels >= 3 else working[..., 0])
        planes = luma.reshape(frames, height * width)
        lanes = 1

    levels = (planes * (BINS - 1)).round().clamp(0, BINS - 1).long()
    histogram = torch.zeros(planes.shape[0], BINS, device=planes.device, dtype=planes.dtype)
    histogram.scatter_add_(1, levels, torch.ones_like(planes))
    cumulative = histogram.cumsum(1) / planes.shape[1]
    quantiles = _quantiles(cumulative, PROBES)

    # Average each frame's quantile function with its neighbours', per lane, with the ends
    # holding their edge frame so a sequence is not darkened at its start and finish.
    shaped = quantiles.reshape(frames, lanes, PROBES).permute(1, 2, 0)
    weights = _weights(int(radius), planes.device, planes.dtype)
    padded = torch.nn.functional.pad(shaped, (int(radius), int(radius)), mode="replicate")
    reference = torch.nn.functional.conv1d(
        padded.reshape(lanes * PROBES, 1, frames + 2 * int(radius)),
        weights.reshape(1, 1, -1),
    ).reshape(lanes, PROBES, frames).permute(2, 0, 1).reshape(frames * lanes, PROBES)

    # One curve per row: where a level sits in this frame's distribution, read off the average.
    bins = torch.linspace(0, 1, BINS, device=planes.device, dtype=planes.dtype)
    bins = bins.expand(planes.shape[0], BINS)
    curve = _interpolate(reference, cumulative * (PROBES - 1))
    curve = torch.maximum(curve.cummax(1).values, torch.zeros_like(curve))

    if per_channel:
        corrected = _interpolate(curve, planes * (BINS - 1))
        corrected = corrected.reshape(frames, channels, height, width).permute(0, 2, 3, 1)
    else:
        gains = _interpolate(curve, planes * (BINS - 1)).reshape(frames, height, width)
        # The same curve on every channel, applied as a ratio so a hue is not moved.
        safe = planes.reshape(frames, height, width).clamp(min=1e-6)
        corrected = working * (gains / safe).unsqueeze(-1)

    corrected = corrected.clamp(0.0, 1.0)
    blended = working + (corrected - working) * float(min(max(strength, 0.0), 1.0))
    return blended.clamp(0.0, 1.0).to(original.dtype)
