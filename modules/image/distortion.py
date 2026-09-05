"""Analogue and digital display artefacts.

Three functions backing the three modes of the monitor-distortion node, each taking a PIL
image with a channel axis and returning one of the same size and mode.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

__all__ = ["digital_distortion", "signal_distortion", "tv_vhs_distortion"]


def digital_distortion(image: Image.Image, amplitude: int = 5,
                       line_width: int = 2) -> Image.Image:
    """Shear columns along a sine wave and punch random scan lines through the result.

    Args:
        image: Source image, read as an array, so it must have a channel axis.
        amplitude: Peak column offset in pixels, truncated to an int by the sine table.
            It also scales the scan-line mask, which saturates at 1 for any amplitude at
            or above 0.02.
        line_width: Row stride of the scan lines. Every ``line_width``-th row is replaced,
            so 1 replaces the whole image.

    Returns:
        An image the same size and mode as the source.
    """
    im = np.array(image)

    x, y, z = im.shape
    sine_wave = amplitude * np.sin(np.linspace(-np.pi, np.pi, y))
    sine_wave = sine_wave.astype(int)

    left_distortion = np.zeros((x, y, z), dtype=np.uint8)
    right_distortion = np.zeros((x, y, z), dtype=np.uint8)
    for i in range(y):
        left_distortion[:, i, :] = np.roll(im[:, i, :], -sine_wave[i], axis=0)
        right_distortion[:, i, :] = np.roll(im[:, i, :], sine_wave[i], axis=0)

    distorted_image = np.maximum(left_distortion, right_distortion)
    scan_lines = np.zeros((x, y), dtype=np.float32)
    scan_lines[::line_width, :] = 1
    scan_lines = np.minimum(scan_lines * amplitude*50.0, 1)  # Scale scan line values
    scan_lines = np.tile(scan_lines[:, :, np.newaxis], (1, 1, z))  # Add channel dimension
    distorted_image = np.where(scan_lines > 0, np.random.permutation(im), distorted_image)
    distorted_image = np.roll(distorted_image, np.random.randint(0, y), axis=1)

    distorted_image = Image.fromarray(distorted_image)

    return distorted_image


def signal_distortion(image: Image.Image, amplitude: int) -> Image.Image:
    """Shift every row horizontally by a random amount plus a sawtooth ramp.

    Args:
        image: Source image, read as an array.
        amplitude: Half-width of the random shift, in pixels, and the period of the ramp.

    Returns:
        An image the same size and mode as the source.

    Raises:
        ValueError: ``amplitude`` is negative, which leaves ``randint`` an empty range.
        ZeroDivisionError: ``amplitude`` is 0, which leaves the ramp a modulo by zero.
    """
    img_array = np.array(image)
    row_shifts = np.random.randint(-amplitude, amplitude + 1, size=img_array.shape[0])
    distorted_array = np.zeros_like(img_array)

    for y in range(img_array.shape[0]):
        x_shift = row_shifts[y]
        x_shift = x_shift + y % (amplitude * 2) - amplitude
        distorted_array[y,:] = np.roll(img_array[y,:], x_shift, axis=0)

    distorted_image = Image.fromarray(distorted_array)

    return distorted_image


def tv_vhs_distortion(image: Image.Image, amplitude: int = 10) -> Image.Image:
    """Tear rows, add an interference pattern, and overlay the result on the source.

    Args:
        image: Source image, read as an array.
        amplitude: Divisor of the image height. Larger values give *smaller* shifts, so
            this argument runs opposite to the one :func:`signal_distortion` takes.

    Returns:
        An image the same size and mode as the source.

    Raises:
        ZeroDivisionError: ``amplitude`` is larger than the image height, which floors the
            shift range to 0 and leaves the ramp with a modulo by zero.
    """
    np_image = np.array(image)
    offset_variance = int(image.height / amplitude)
    row_shifts = np.random.randint(-offset_variance, offset_variance + 1, size=image.height)
    distorted_array = np.zeros_like(np_image)

    for y in range(np_image.shape[0]):
        x_shift = row_shifts[y]
        x_shift = x_shift + y % (offset_variance * 2) - offset_variance
        distorted_array[y,:] = np.roll(np_image[y,:], x_shift, axis=0)

    h, w, c = distorted_array.shape
    x_scale = np.linspace(0, 1, w)
    y_scale = np.linspace(0, 1, h)
    x_idx = np.broadcast_to(x_scale, (h, w))
    y_idx = np.broadcast_to(y_scale.reshape(h, 1), (h, w))
    noise = np.random.rand(h, w, c) * 0.1
    distortion = np.sin(x_idx * 50) * 0.5 + np.sin(y_idx * 50) * 0.5
    distorted_array = distorted_array + distortion[:, :, np.newaxis] + noise

    distorted_image = Image.fromarray(np.uint8(distorted_array))
    distorted_image = distorted_image.resize((image.width, image.height))

    image_enhance = ImageEnhance.Color(image)
    image = image_enhance.enhance(0.5)

    effect_image = ImageChops.overlay(image, distorted_image)
    result_image = ImageChops.overlay(image, effect_image)
    result_image = ImageChops.blend(image, result_image, 0.25)

    return result_image
