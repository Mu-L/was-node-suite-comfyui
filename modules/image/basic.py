"""Whole-image smoothing and resampling.

:func:`medianFilter` and :func:`resizeImage` both take and return a PIL image, and neither
has any notion of a batch, so a caller iterates a batch itself.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .convolve import bilateral_blur

__all__ = ["medianFilter", "resizeImage"]


def medianFilter(img: Image.Image, diameter: float, sigmaColor: float,
                 sigmaSpace: float) -> Image.Image:
    """Smooth an image while preserving its edges, with a bilateral filter.

    Args:
        img: Source image. Converted to ``RGB`` first, so a mode with alpha loses it.
        diameter: Neighbourhood diameter in pixels, truncated to an int.
        sigmaColor: Colour-space sigma, truncated to an int. Larger values mix colours
            that differ more.
        sigmaSpace: Coordinate-space sigma, truncated to an int. Larger values let more
            distant pixels take part.

    Returns:
        An ``RGB`` image the same size as the source.
    """
    pixels = torch.from_numpy(np.array(img.convert('RGB'))).float().permute(2, 0, 1)[None]
    settings = (int(diameter), int(sigmaColor), int(sigmaSpace))
    filtered = _on_device(pixels, settings)
    array = filtered[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode='RGB')


def _on_device(pixels: torch.Tensor, settings: tuple) -> torch.Tensor:
    """Run the filter on the compute device, falling back to the CPU when VRAM runs out."""
    from ..model import compute_device

    device = compute_device()
    if device.type != "cuda":
        return bilateral_blur(pixels, *settings)
    try:
        return bilateral_blur(pixels.to(device), *settings)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return bilateral_blur(pixels, *settings)


def resizeImage(image: Image.Image, max_size: int) -> Image.Image:
    """Scale an image down until its longer side is ``max_size``, preserving aspect.

    Args:
        image: Source image.
        max_size: Length in pixels of the longer side of the result.

    Returns:
        The resized image, or the source image when its longer side already fits.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / longest
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
