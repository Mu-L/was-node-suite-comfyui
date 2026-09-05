"""Bringing images of different sizes to one size, so they can share a batch.

Images are float tensors shaped ``(batch, height, width, channels)`` in ``[0, 1]``. Every method
answers a tensor of the requested height and width.
"""

from __future__ import annotations

import torch

__all__ = ["METHODS", "PAD_LEVEL", "RGBA", "fit_to", "pad_to", "target_size"]

# `resize` scales to the target and ignores the shape it had, so nothing is lost or added but a
# picture of another shape is stretched. `crop` scales until the target is covered, keeping the
# shape, then takes the middle, so what falls outside the frame is gone. `pad` scales until the
# target contains it and centres it on a flat field, keeping the whole picture.

#: The methods, in the order a node offers them.
METHODS = ("resize", "crop", "pad")

#: What ``pad`` fills the rest of the frame with, as a level in ``[0, 1]``.
PAD_LEVEL = 0.0

#: Channels a frame carrying alpha holds.
RGBA = 4


def target_size(tensor) -> tuple[int, int]:
    """The height and width of an image batch.

    Args:
        tensor: An ``IMAGE`` tensor.

    Returns:
        ``(height, width)``.
    """
    return int(tensor.shape[1]), int(tensor.shape[2])


def _scaled(tensor, height: int, width: int):
    """``tensor`` resampled to exactly ``height`` by ``width``."""
    # Interpolation wants the channels next to the batch, and an image carries them last.
    planes = tensor.permute(0, 3, 1, 2)
    # `antialias` only applies when scaling down, where without it a large reduction drops
    # detail between samples instead of averaging it.
    resampled = torch.nn.functional.interpolate(
        planes, size=(height, width), mode="bilinear", align_corners=False, antialias=True,
    )
    return resampled.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _centre_crop(tensor, height: int, width: int):
    """The middle ``height`` by ``width`` of ``tensor``, which must be at least that large."""
    top = max(0, (int(tensor.shape[1]) - height) // 2)
    left = max(0, (int(tensor.shape[2]) - width) // 2)
    return tensor[:, top:top + height, left:left + width, :]


def pad_to(
    tensor, height: int, width: int, level: float = PAD_LEVEL, transparent: bool = False,
):
    """One image batch centred on a larger field, with nothing resampled.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)``, no larger than the
            field on either axis.
        height: Height of the field.
        width: Width of the field.
        level: What the field around the frame holds, as a level in ``[0, 1]``.
        transparent: Answer 4 channels, the field fully transparent and the frame opaque. A
            frame that already carries alpha keeps its own.

    Returns:
        A tensor of exactly that height and width, at the source channel count, or at
        :data:`RGBA` channels where ``transparent`` is set.
    """
    height, width = int(height), int(width)
    batch, current_height, current_width, channels = (int(axis) for axis in tensor.shape)
    depth = RGBA if transparent else channels
    field = torch.full(
        (batch, height, width, depth), level,
        dtype=tensor.dtype, device=tensor.device,
    )
    if transparent:
        field[..., 3] = 0.0
    top = max(0, (height - current_height) // 2)
    left = max(0, (width - current_width) // 2)
    carried = min(channels, depth)
    rows = slice(top, top + current_height)
    columns = slice(left, left + current_width)
    field[:, rows, columns, :carried] = tensor[..., :carried]
    if transparent and channels < RGBA:
        field[:, rows, columns, 3] = 1.0
    return field


def fit_to(tensor, height: int, width: int, method: str = "resize"):
    """One image batch brought to ``height`` by ``width``.

    Args:
        tensor: An ``IMAGE`` tensor, ``(batch, height, width, channels)``.
        height: The height to answer with.
        width: The width to answer with.
        method: One of :data:`METHODS`.

    Returns:
        A tensor of exactly that height and width. The same tensor is handed back untouched
        when it is already that size, whatever the method.

    Raises:
        ValueError: ``method`` is not one of :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(
            f"resize_method is {method!r}, which is not one of {', '.join(METHODS)}."
        )
    current_height, current_width = target_size(tensor)
    if (current_height, current_width) == (int(height), int(width)):
        return tensor

    if method == "resize":
        return _scaled(tensor, int(height), int(width))

    # Both of the shape-keeping methods scale by a single factor and then square the frame up,
    # one by taking the middle and one by filling around it.
    if method == "crop":
        factor = max(height / current_height, width / current_width)
    else:
        factor = min(height / current_height, width / current_width)
    scaled_height = max(1, round(current_height * factor))
    scaled_width = max(1, round(current_width * factor))
    scaled = _scaled(tensor, scaled_height, scaled_width)

    if method == "crop":
        return _centre_crop(scaled, int(height), int(width))
    return pad_to(scaled, int(height), int(width))
