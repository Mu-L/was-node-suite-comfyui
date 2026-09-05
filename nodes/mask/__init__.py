"""Mask nodes, and the tensor handling they share.

A ``MASK`` arrives as ``(batch, height, width)``, ``(batch, 1, height, width)``,
``(batch, height, width, 1)`` or ``(height, width)``. ``mask_images`` reduces all four to
one greyscale image per mask.
"""

from __future__ import annotations

import torch

from ...modules.convert.tensors import mask_images, mask_planes, pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import morph_region

__all__ = [
    "float_mask",
    "mask_images",
    "mask_planes",
    "morph_masks",
    "same_size_or_refuse",
    "stack_masks",
]


def same_size_or_refuse(named, operation: str) -> None:
    """Reject masks that cannot be paired pixel by pixel.

    Args:
        named: ``(input id, tensor)`` pairs in slot order. A pair whose mask is None is
            skipped, so an optional slot left empty costs nothing.
        operation: The node's display name, opening the message.

    Raises:
        ValueError: At least two of them carry different shapes.
    """
    present = [(name, mask) for name, mask in named if mask is not None]
    if len({tuple(mask.shape) for _, mask in present}) <= 1:
        return
    listed = ", ".join(f"{name} is {tuple(mask.shape)}" for name, mask in present)
    # A height or width apart takes one remedy, a frame count apart takes the other.
    if len({(int(m.shape[-2]), int(m.shape[-1])) for _, m in present}) > 1:
        remedy = (
            "One of them has usually been through a crop. To bring a mask to another size, run "
            "it through Convert Mask to Image, then Upscale Image, then Convert Image to Mask."
        )
    else:
        remedy = (
            "They cover the same area but carry different numbers of frames, so there is "
            "nothing to pair the extra frames with. Feed both from the same batch, or combine "
            "them a frame at a time."
        )
    raise ValueError(
        f"{operation} pairs its masks pixel by pixel, so every connected mask must be the "
        f"same size and hold the same number of frames. These do not match: {listed}. {remedy}"
    )


def stack_masks(masks: list[torch.Tensor]) -> torch.Tensor:
    """Assemble 2D masks into the ``(batch, 1, height, width)`` tensor a mask node emits.

    Args:
        masks: One ``(height, width)`` mask per item of the batch, in batch order.

    Returns:
        A tensor shaped ``(len(masks), 1, height, width)``.

    Raises:
        ValueError: No mask was given.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")
    height = max(mask.shape[0] for mask in masks)
    width = max(mask.shape[1] for mask in masks)
    padded = []
    for mask in masks:
        if mask.shape != (height, width):
            canvas = mask.new_zeros((height, width))
            top = (height - mask.shape[0]) // 2
            left = (width - mask.shape[1]) // 2
            canvas[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
            mask = canvas
        padded.append(mask)
    return torch.stack(padded, dim=0).unsqueeze(1)


def float_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a mask that arithmetic works on.

    Args:
        mask: Mask tensor. A boolean one supports neither subtraction nor negation in
            torch, and reaches a ``MASK`` socket from any node that thresholds without
            casting back.

    Returns:
        The mask itself, or a float32 copy of a boolean one.
    """
    return mask.to(torch.float32) if mask.dtype == torch.bool else mask


def morph_masks(
    masks: torch.Tensor, iterations: int, grow: bool, blur: float = 0.0
) -> torch.Tensor:
    """Reshape every mask of a batch by binary morphology and report the change.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries.
        iterations: Morphology passes. Zero and below run until the result stops changing.
        grow: Dilate the set area when true, erode it when false.
        blur: Gaussian radius in pixels for the edge. 0 leaves the hard binary edge.

    Returns:
        A tensor shaped ``(batch, 1, height, width)``.
    """
    shaped = [morph_region(image, iterations, grow, blur) for image in mask_images(masks)]
    stacked = stack_masks([pil2mask(image) for image in shaped])
    mask_report.publish(masks, stacked)
    return stacked
