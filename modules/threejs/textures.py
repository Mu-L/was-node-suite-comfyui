"""Carrying an ``IMAGE`` batch's first frame to the browser as a texture."""

from __future__ import annotations

__all__ = ["COLOR_SPACES", "WRAP_MODES", "as_png", "texture_url"]

import io as _io

from ..interface import three_asset

#: Colour spaces a texture may be tagged with, in the order a menu lists them.
COLOR_SPACES = ("srgb", "linear-srgb", "none")

#: How a texture repeats past its edges, in the order a menu lists them.
WRAP_MODES = ("clamp", "repeat", "mirrored-repeat")


def as_png(image) -> bytes:
    """One ``IMAGE`` batch's first frame encoded as PNG.

    Args:
        image: An ``IMAGE`` tensor or array, ``(batch, height, width, channels)`` or
            ``(height, width, channels)``, with 1, 3 or 4 channels.

    Returns:
        The PNG bytes.

    Raises:
        ValueError: Nothing was given, the shape is not an image, or the channel count is
            not 1, 3 or 4.
    """
    import numpy as np
    from PIL import Image

    if image is None:
        raise ValueError("No image arrived, so there is nothing to make a texture from.")

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"A texture wants an image shaped (height, width, channels), or a batch of them, "
            f"and this one is shaped {tuple(array.shape)}."
        )
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"A texture wants 1, 3 or 4 channels and this image has {array.shape[-1]}."
        )

    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    buffer = _io.BytesIO()
    Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB").save(
        buffer, format="PNG", optimize=True
    )
    return buffer.getvalue()


def texture_url(image) -> str:
    """Hold one frame for the browser and answer where to fetch it.

    Args:
        image: An ``IMAGE`` tensor or array, as :func:`as_png` takes.

    Returns:
        The route and key the browser fetches the encoded frame from.

    Raises:
        ValueError: Nothing was given, the shape is not an image, or the channel count is
            not 1, 3 or 4.
    """
    return "%s?key=%s" % (three_asset.ROUTE, three_asset.keep(as_png(image)))
