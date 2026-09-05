"""The mask drawn on a node, and how it joins the computed mask.

A drawing is held in one ``STRING`` input as ``was-mask-1 <width>x<height> <base64 png>``,
and decodes to a float tensor shaped ``(height, width)`` in ``[0, 1]``.
"""

from __future__ import annotations

import base64
from io import BytesIO

import torch

from .. import log

__all__ = [
    "COMBINE_MODES",
    "DEFAULT_COMBINE",
    "MASK_MAX_EDGE",
    "VALUE_TAG",
    "apply",
    "combine",
    "decode",
    "resample",
]

logger = log.get_logger("mask.drawn")

#: The tag every value starts with. A later format takes a later tag.
VALUE_TAG = "was-mask-1"

#: Longest edge the interface stores a drawing at, in pixels. Stated here so the decoder
#: rejects a value large enough to have come from somewhere else.
MASK_MAX_EDGE = 2048

#: What the drawing does to the computed mask, in the order the combo offers them.
COMBINE_MODES = ("union", "subtract", "intersect", "off")

#: The mode a node ships with. Union of an empty drawing is the computed mask.
DEFAULT_COMBINE = "union"


def decode(value: str) -> torch.Tensor | None:
    """Read a stored drawing as a mask.

    Args:
        value: What the ``drawn_mask`` input holds.

    Returns:
        A float32 tensor shaped ``(height, width)`` with values in ``[0, 1]``, where 1 is
        drawn. ``None`` when the value is empty, is not this format, or cannot be decoded.
    """
    header = _header(value)
    if header is None:
        return None
    width, height, data = header

    try:
        from PIL import Image

        import numpy as np

        with Image.open(BytesIO(base64.b64decode(data, validate=True))) as picture:
            # The luminance, not the alpha. The interface composites its strokes onto black
            # before encoding, so a soft edge is a grey level and the alpha channel, where
            # the browser's encoder writes one at all, is opaque everywhere.
            plane = picture.convert("L")
            if plane.size != (width, height):
                plane = plane.resize((width, height), Image.BILINEAR)
            array = np.asarray(plane, dtype=np.float32) / 255.0
    except Exception as error:
        logger.warning("the drawn mask could not be decoded (%s), so it was ignored", error)
        return None

    return torch.from_numpy(array.copy())


def resample(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a mask to a given shape.

    Args:
        mask: Float tensor shaped ``(height, width)``.
        height: Rows wanted.
        width: Columns wanted.

    Returns:
        A float32 tensor shaped ``(height, width)``, bilinearly resampled and clamped to
        ``[0, 1]``. The input is answered unchanged when it is already that shape.
    """
    rows = max(1, int(height))
    columns = max(1, int(width))
    if mask.shape[0] == rows and mask.shape[1] == columns:
        return mask

    resized = torch.nn.functional.interpolate(
        mask.reshape(1, 1, mask.shape[0], mask.shape[1]).to(torch.float32),
        size=(rows, columns),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(rows, columns).clamp(0.0, 1.0)


def combine(computed: torch.Tensor, drawn: torch.Tensor, mode: str) -> torch.Tensor:
    """Join a drawing with a computed mask.

    Args:
        computed: The mask the node worked out, any shape.
        drawn: The drawing, broadcastable onto ``computed``.
        mode: One of :data:`COMBINE_MODES`. An unknown mode answers the computed mask.

    Returns:
        The joined mask, the same shape as ``computed``, with values in ``[0, 1]``.
    """
    if mode == "union":
        return torch.maximum(computed, drawn)
    if mode == "subtract":
        return torch.clamp(computed - drawn, 0.0, 1.0)
    if mode == "intersect":
        return torch.minimum(computed, drawn)
    if mode != "off":
        logger.warning("drawn_combine was %r, which is not a combine, so the drawing was left out", mode)
    return computed


def apply(computed: torch.Tensor, value: str, mode: str) -> torch.Tensor:
    """Join whatever a node's ``drawn_mask`` holds with the mask it computed.

    Args:
        computed: A ``MASK`` tensor shaped ``(batch, height, width)``.
        value: What the ``drawn_mask`` input holds.
        mode: What the ``drawn_combine`` input holds, one of :data:`COMBINE_MODES`.

    Returns:
        The mask to hand on, the same shape and dtype family as ``computed``. The input is
        answered unchanged when nothing was drawn or the mode is ``off``.
    """
    if mode == "off":
        return computed

    drawn = decode(value)
    if drawn is None:
        return computed

    if computed.ndim < 2:
        logger.warning(
            "the computed mask is %d dimensional, so the drawing could not be placed on it",
            computed.ndim,
        )
        return computed

    drawn = resample(drawn, computed.shape[-2], computed.shape[-1]).to(computed.dtype)
    return combine(computed, drawn, mode)


def _header(value: str) -> tuple[int, int, str] | None:
    """Split a stored drawing into its size and its base64 body.

    Args:
        value: What the ``drawn_mask`` input holds.

    Returns:
        ``(width, height, base64)``, or ``None`` when the value is empty or is not this
        format.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    parts = text.split()
    if len(parts) != 3 or parts[0] != VALUE_TAG:
        logger.warning("drawn_mask does not hold a %s value, so it was ignored", VALUE_TAG)
        return None

    size = parts[1].split("x")
    if len(size) != 2 or not all(part.isdigit() for part in size):
        logger.warning("drawn_mask states its size as %r, which is not <width>x<height>", parts[1])
        return None

    width, height = int(size[0]), int(size[1])
    if not 0 < width <= MASK_MAX_EDGE or not 0 < height <= MASK_MAX_EDGE:
        logger.warning(
            "drawn_mask states %dx%d, which is outside 1 to %d pixels, so it was ignored",
            width, height, MASK_MAX_EDGE,
        )
        return None

    return width, height, parts[2]
