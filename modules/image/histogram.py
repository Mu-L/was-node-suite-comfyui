"""Counting an image's tones into the 256 bins a HISTOGRAM socket carries.

Bins are read from one frame. ``luminance`` is BT.709 and ``rgb`` is the mean of the three
colour channels, which is the composite a levels chart draws.
"""

from __future__ import annotations

__all__ = ["BINS", "Bins", "bins"]

from typing import NamedTuple

#: Bins a histogram holds, one per code an 8-bit channel can take.
BINS = 256

#: BT.709 luminance weights, red, green and blue.
LUMA = (0.2126, 0.7152, 0.0722)


class Bins(NamedTuple):
    """The five counts a histogram socket set carries.

    Attributes:
        rgb: Mean of the three colour channels, bin for bin.
        luminance: BT.709 luminance.
        red: Red channel.
        green: Green channel.
        blue: Blue channel.
    """

    rgb: list[int]
    luminance: list[int]
    red: list[int]
    green: list[int]
    blue: list[int]


def bins(image) -> Bins:
    """Count one frame's tones into 256 bins per channel.

    Args:
        image: ``(height, width, channels)`` float array or tensor in ``[0, 1]``, or a
            ``(frames, height, width, channels)`` batch, of which the first frame is read.
            A single channel is read as all three.

    Returns:
        The five counts, each a list of 256 integers.

    Raises:
        ValueError: The array carries no height, width and channel axes.
    """
    import numpy as np

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(
            f"a histogram is counted from a (height, width, channels) frame, not an array "
            f"of shape {tuple(array.shape)}"
        )

    frame = array.astype(np.float32)
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    frame = frame[:, :, :3]

    codes = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    def count(plane):
        return np.bincount(plane.ravel(), minlength=BINS)[:BINS]

    red, green, blue = (count(codes[:, :, index]) for index in range(3))
    luma = frame[:, :, 0] * LUMA[0] + frame[:, :, 1] * LUMA[1] + frame[:, :, 2] * LUMA[2]
    luminance = count(np.clip(luma * 255.0, 0, 255).astype(np.uint8))

    return Bins(
        rgb=((red + green + blue) // 3).tolist(),
        luminance=luminance.tolist(),
        red=red.tolist(),
        green=green.tolist(),
        blue=blue.tolist(),
    )
