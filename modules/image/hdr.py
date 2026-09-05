"""Reconstructing what a finished picture lost: the levels between its codes.

Images are ``(batch, height, width, 3)``. :func:`dequantise` answers a picture on the same
0 to 1 scale; the reconstruction never leaves a sample by more than half a code.
"""

from __future__ import annotations

import torch
from torch.nn import functional

__all__ = ["dequantise"]

#: Codes an 8-bit picture is stored with, less one, so a level is 1/255.
EIGHT_BIT = 255

#: Passes of smoothing and projection. Each pass moves the estimate towards a smooth
#: signal, and the projection after it holds the result inside the source's own interval.
ROUNDS = 6

#: Widest span smoothed over, in pixels, and the smallest that still spans a band.
LARGEST_RADIUS = 24.0
SMALLEST_RADIUS = 1.0


def _weights(span: int, radius: float, image: torch.Tensor) -> torch.Tensor:
    """A normalised Gaussian of ``span`` taps, on the image's device and type."""
    grid = torch.arange(span, device=image.device, dtype=image.dtype) - span // 2
    kernel = torch.exp(-(grid * grid) / (2.0 * radius * radius))
    return kernel / kernel.sum()


def _blurred(image: torch.Tensor, radius: float) -> torch.Tensor:
    """One separable Gaussian pass over a batch of images."""
    span = int(radius * 3) | 1
    planes = image.permute(0, 3, 1, 2)
    height, width = int(planes.shape[2]), int(planes.shape[3])
    # A reflection reaches no further than the edge it turns at, so each axis takes the
    # widest odd kernel its own side has room for.
    across = min(span, 2 * width - 1)
    down = min(span, 2 * height - 1)
    row = _weights(across, radius, image).view(1, 1, 1, -1)
    column = _weights(down, radius, image).view(1, 1, -1, 1)
    planes = functional.conv2d(
        functional.pad(planes, (across // 2, across // 2, 0, 0), mode="reflect"),
        row.expand(3, 1, 1, across), groups=3,
    )
    planes = functional.conv2d(
        functional.pad(planes, (0, 0, down // 2, down // 2), mode="reflect"),
        column.expand(3, 1, down, 1), groups=3,
    )
    return planes.permute(0, 2, 3, 1)


def dequantise(
    images: torch.Tensor, levels: int = EIGHT_BIT, radius: float = 8.0, rounds: int = ROUNDS
) -> torch.Tensor:
    """Reconstruct the gradient a quantiser flattened into steps.

    Args:
        images: ``(batch, height, width, 3)`` on a 0 to 1 scale.
        levels: Codes the picture was stored with, less one. ``255`` for 8-bit.
        radius: Pixels the reconstruction smooths over, from
            :data:`SMALLEST_RADIUS` to :data:`LARGEST_RADIUS`.
        rounds: Passes of smoothing and projection.

    Returns:
        A tensor of the same shape on a 0 to 1 scale, within half a code of ``images`` and
        never below it at the top code.

    Raises:
        ValueError: ``levels`` is below one.
    """
    if levels < 1:
        raise ValueError(f"a picture is stored with at least one level, not {levels}")

    held = max(min(float(radius), LARGEST_RADIUS), SMALLEST_RADIUS)
    half = 0.5 / float(levels)
    planes = images[..., :3]
    low, high = planes - half, planes + half
    # A sample at the top code is clipped rather than quantised, so its own value is the
    # floor the projection holds it at.
    low = torch.where(planes >= 1.0, planes, low)

    working = planes.clone()
    for _ in range(max(int(rounds), 1)):
        working = torch.maximum(torch.minimum(_blurred(working, held), high), low)
    return working.clamp(0.0, 1.0)
