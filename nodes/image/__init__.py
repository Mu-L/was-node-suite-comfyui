"""Image nodes, and the batch handling they share.

An ``IMAGE`` reaching one of these nodes is ``(batch, height, width, channels)``.
"""

from __future__ import annotations

import torch
from PIL import Image

from ...modules.convert.tensors import pil2tensor

__all__ = ["image_planes", "quantises_exactly", "stack_images"]


def image_planes(images: torch.Tensor) -> list[torch.Tensor]:
    """Split an image tensor into one image per item of its batch.

    Args:
        images: Image tensor. Four axes or more are read as a batch and iterated; three are
            read as a single unbatched image; fewer are read as one image of one row.

    Returns:
        One ``(height, width, channels)`` view per image, in batch order. Never empty.
    """
    if images.ndim >= 4:
        return list(images)
    if images.ndim == 3:
        return [images]
    if images.ndim == 2:
        return [images.unsqueeze(-1)]
    return [images.reshape(1, -1, 1)]


def stack_images(images: list[Image.Image]) -> torch.Tensor:
    """Assemble PIL images into the ``(batch, height, width, channels)`` tensor a node emits.

    Args:
        images: One PIL image per item of the batch, in batch order, all in the same mode.

    Returns:
        A tensor shaped ``(len(images), height, width, channels)``.

    Raises:
        ValueError: No image was given, or the images do not share a channel count.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    planes = [pil2tensor(image)[0] for image in images]
    channels = {plane.shape[2] if plane.ndim > 2 else 1 for plane in planes}
    if len(channels) > 1:
        raise ValueError(f"All images must share a channel count, got {sorted(channels)}.")
    height = max(plane.shape[0] for plane in planes)
    width = max(plane.shape[1] for plane in planes)
    padded = []
    for plane in planes:
        if plane.shape[:2] != (height, width):
            canvas = plane.new_zeros((height, width, *plane.shape[2:]))
            top = (height - plane.shape[0]) // 2
            left = (width - plane.shape[1]) // 2
            canvas[top:top + plane.shape[0], left:left + plane.shape[1]] = plane
            plane = canvas
        padded.append(plane)
    return torch.stack(padded, dim=0)


def quantises_exactly(plane: torch.Tensor) -> bool:
    """Whether a PIL round trip on this image is nothing but an 8-bit quantisation.

    Args:
        plane: One ``(height, width, channels)`` image of a batch.

    Returns:
        Whether tensor arithmetic on this image is equivalent to the PIL round trip.
    """
    return plane.ndim == 3 and plane.shape[0] > 1 and plane.shape[1] > 1 and plane.shape[2] in (3, 4)
