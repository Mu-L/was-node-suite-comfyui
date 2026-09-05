"""Backward warping, as EMA-VFI's ``model/warplayer.py``, with the device taken from the input.

The arithmetic is upstream's unchanged: the sampling grid is sized from the flow while the flow
is normalised against the input, so warping a feature map with a larger flow answers at the
flow's resolution. Only where the grid is built has moved.
"""

from __future__ import annotations

import torch

__all__ = ["warp"]


def warp(tenInput, tenFlow):
    """Sample ``tenInput`` at positions displaced by ``tenFlow``.

    Args:
        tenInput: What to read from, ``(batch, channels, height, width)``.
        tenFlow: Displacement in pixels, ``(batch, 2, height, width)``. Its height and width
            decide the result's.

    Returns:
        ``tenInput`` sampled onto ``tenFlow``'s grid, edges held rather than faded.
    """
    # Upstream chose the device once at import from `torch.cuda.is_available()` and kept the
    # grids in a global dict keyed by shape. Both are wrong inside ComfyUI: the device is the
    # caller's to decide and a CPU-only install got a CUDA grid, while a cache never emptied
    # grows with every new resolution a workflow feeds it. Two linspaces against a grid_sample
    # are not worth keeping, so the grid is built per call.
    device, dtype = tenFlow.device, tenFlow.dtype
    batch, _, height, width = tenFlow.shape
    horizontal = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(
        1, 1, 1, width).expand(batch, -1, height, -1)
    vertical = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(
        1, 1, height, 1).expand(batch, -1, -1, width)
    grid = torch.cat([horizontal, vertical], 1)

    # Pixels to the -1..1 the sampler wants, against the size actually being read.
    normalised = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(grid + normalised).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
