"""Guided filtering: smoothing one image while following the edges of another.

Images are float tensors shaped ``(batch, height, width, channels)`` in ``[0, 1]``. A
three-channel guide is solved against its own covariance.
"""

from __future__ import annotations

import torch

__all__ = ["MAX_RADIUS", "filter_with_guide"]

#: Widest window the filter offers. The cost is flat in the radius, so this bounds what is
#: meaningful rather than what is affordable: once the window covers the picture the result
#: stops changing.
MAX_RADIUS = 256

#: Smallest epsilon accepted. Zero divides by the variance alone, which is zero across a flat
#: region and would answer infinities there.
MIN_EPSILON = 1e-8


def _windowed_sum(values, radius: int, dim: int):
    """The sum over each ``2 * radius + 1`` window along ``dim``, at every position.

    Args:
        values: The tensor to sum over.
        radius: Half the window.
        dim: The axis to run along.

    Returns:
        A tensor of the same shape. Windows are clipped at the edges rather than padded, so an
        edge entry sums the part of its window that exists.
    """
    length = values.shape[dim]
    cumulative = values.cumsum(dim)
    positions = torch.arange(length, device=values.device)

    upper = cumulative.index_select(dim, torch.clamp(positions + radius, max=length - 1))
    lower_at = positions - radius - 1
    lower = cumulative.index_select(dim, torch.clamp(lower_at, min=0))
    # Below the first entry there is nothing to subtract, and a clamped index would subtract
    # the first entry instead of zero.
    shape = [1] * values.dim()
    shape[dim] = length
    inside = (lower_at >= 0).reshape(shape).to(values.dtype)
    return upper - lower * inside


def _counts(height: int, width: int, radius: int, like):
    """How many pixels each window covers, which is fewer at the edges."""
    ones = torch.ones((1, 1, height, width), dtype=like.dtype, device=like.device)
    return _windowed_sum(_windowed_sum(ones, radius, 2), radius, 3)


def _mean(values, radius: int, counts):
    """The window mean of a channels-last tensor, ``(batch, height, width, channels)``."""
    planes = values.permute(0, 3, 1, 2)
    summed = _windowed_sum(_windowed_sum(planes, radius, 2), radius, 3)
    return (summed / counts).permute(0, 2, 3, 1)


def _matched(source, guide):
    """``source`` and ``guide`` brought to one batch length and one size.

    Raises:
        ValueError: The two hold different numbers of frames and neither holds a single one.
    """
    if source.shape[0] != guide.shape[0]:
        if guide.shape[0] == 1:
            guide = guide.expand(source.shape[0], -1, -1, -1)
        elif source.shape[0] == 1:
            source = source.expand(guide.shape[0], -1, -1, -1)
        else:
            raise ValueError(
                f"the source holds {source.shape[0]} frame(s) and the guide {guide.shape[0]}; "
                f"they must match, or one of them must hold a single frame"
            )
    if source.shape[1:3] != guide.shape[1:3]:
        # Lifting the source to the guide's size here is what makes a small mask or depth map
        # come back at full size with the guide's edges, in one step.
        source = torch.nn.functional.interpolate(
            source.permute(0, 3, 1, 2), size=tuple(guide.shape[1:3]),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)
    return source, guide


def filter_with_guide(source, guide, radius: int = 8, epsilon: float = 0.01):
    """Smooth ``source`` while holding the edges that ``guide`` has.

    ``source`` is resized to the guide when the two differ.

    Args:
        source: The tensor being smoothed, ``(batch, height, width, channels)``.
        guide: The tensor whose edges are followed, ``(batch, height, width, channels)``. One
            channel is solved as a scalar, three as a full covariance, and a fourth is ignored.
        radius: Half the window, in pixels of the guide.
        epsilon: How much variance still counts as flat. Squared intensity units, so 0.01 is a
            change of 0.1 in ``[0, 1]``.

    Returns:
        A tensor at the guide's height and width and the source's channel count.

    Raises:
        ValueError: Either input is not a batch of images, or the guide carries no usable
            channels.
    """
    if getattr(source, "ndim", 0) != 4 or getattr(guide, "ndim", 0) != 4:
        raise ValueError(
            "guided filtering takes two batches shaped (batch, height, width, channels)"
        )

    radius = max(1, min(int(radius), MAX_RADIUS))
    epsilon = max(float(epsilon), MIN_EPSILON)

    # Where the guide is flat the source is averaged over the window; where the guide has an
    # edge the source is allowed to change with it. Resizing here rather than asking the caller
    # to is what lets a small mask or depth map arrive at full size with the guide's edges.
    working = source.to(torch.float32)
    lead = guide.to(torch.float32)
    if lead.shape[3] >= 3:
        # An alpha channel says nothing about where an edge is, so it is left out.
        lead = lead[..., :3]
    elif lead.shape[3] != 1:
        raise ValueError(
            f"the guide carries {int(guide.shape[3])} channel(s); it must be greyscale or RGB"
        )
    working, lead = _matched(working, lead)

    height, width = int(lead.shape[1]), int(lead.shape[2])
    counts = _counts(height, width, radius, lead)
    mean_guide = _mean(lead, radius, counts)
    mean_source = _mean(working, radius, counts)

    if lead.shape[3] == 1:
        # A scalar slope per pixel: how much the source moves for a move in the guide.
        covariance = _mean(lead * working, radius, counts) - mean_guide * mean_source
        variance = _mean(lead * lead, radius, counts) - mean_guide * mean_guide
        slope = covariance / (variance + epsilon)
        offset = mean_source - slope * mean_guide
        result = _mean(slope, radius, counts) * lead + _mean(offset, radius, counts)
        return result.clamp(0.0, 1.0).to(source.dtype)

    channels = int(working.shape[3])
    # (batch, height, width, source channels, 3): each source channel against each guide channel.
    covariance = torch.stack(
        [
            _mean(lead[..., k:k + 1] * working, radius, counts)
            - mean_guide[..., k:k + 1] * mean_source
            for k in range(3)
        ],
        dim=-1,
    )
    # The guide's own 3x3 covariance at every pixel.
    sigma = torch.stack(
        [
            torch.stack(
                [
                    (_mean(lead[..., i:i + 1] * lead[..., j:j + 1], radius, counts)
                     - mean_guide[..., i:i + 1] * mean_guide[..., j:j + 1]).squeeze(-1)
                    for j in range(3)
                ],
                dim=-1,
            )
            for i in range(3)
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=sigma.dtype, device=sigma.device)
    # Solved rather than inverted: the same answer without forming the inverse.
    slope = torch.linalg.solve(sigma + epsilon * eye, covariance.transpose(-1, -2))
    slope = slope.transpose(-1, -2)
    offset = mean_source - (slope @ mean_guide.unsqueeze(-1)).squeeze(-1)

    slope_mean = _mean(
        slope.reshape(slope.shape[0], height, width, channels * 3), radius, counts,
    ).reshape(slope.shape[0], height, width, channels, 3)
    offset_mean = _mean(offset, radius, counts)
    result = (slope_mean @ lead.unsqueeze(-1)).squeeze(-1) + offset_mean
    return result.clamp(0.0, 1.0).to(source.dtype)
