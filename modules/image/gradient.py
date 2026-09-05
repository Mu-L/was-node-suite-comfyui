"""Colour gradient generation and gradient mapping.

:func:`parse_gradient_stops` reads ``position:r,g,b`` text into the dict :func:`gradient`
takes. :func:`gradient_map` recolours an image by looking its luminance up in a gradient.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


__all__ = ["gradient", "gradient_map", "parse_gradient_stops"]


def parse_gradient_stops(text: str) -> dict[int, list[int]]:
    """Read ``position:r,g,b`` lines into a mapping of stop position to RGB channels.

    Args:
        text: One stop per line as ``position:r,g,b``.

    Returns:
        Mapping of stop position to a list of three channel values.

    Raises:
        ValueError: Not one line could be read as a stop.
    """
    stops: dict[int, list[int]] = {}
    for line in text.strip().replace(' ', '').splitlines():
        parts = line.split(':')
        if len(parts) < 2:
            continue
        channels = parts[1].split(',')
        if len(channels) < 3:
            continue
        try:
            stops[int(parts[0])] = [int(channel) for channel in channels[:3]]
        except ValueError:
            continue
    if not stops:
        raise ValueError(
            "gradient_stops holds no readable stop. Every line is 'position:r,g,b', for "
            "example '0:255,0,0', with the position running 0 to 100 across the image and "
            "each channel 0 to 255."
        )
    return stops


def gradient(
    size: tuple[int, int],
    mode: str = 'horizontal',
    colors: dict | str | None = None,
    tolerance: int = 0,
) -> Image.Image:
    """Draw a linear gradient through a set of colour stops.

    Args:
        size: ``(width, height)`` in pixels.
        mode: ``'horizontal'`` or ``'vertical'``. Any other value leaves the image black.
        colors: Mapping of stop position to ``[r, g, b]``, or the same as a JSON string.
            ``None`` selects a red-green-blue default. Keys and channels are coerced to
            ``int``.
        tolerance: Quantisation step applied to each channel. 0 disables it.

    Returns:
        An ``RGB`` image of ``size``.

    Raises:
        ZeroDivisionError: ``size`` has a 1-pixel extent along the gradient axis.
    """

    if isinstance(colors, str):
        colors = json.loads(colors)

    if colors is None:
        colors = {0: [255, 0, 0], 50: [0, 255, 0], 100: [0, 0, 255]}

    colors = {int(k): [int(c) for c in v] for k, v in colors.items()}

    colors[0] = colors[min(colors.keys())]
    colors[255] = colors[max(colors.keys())]

    img = Image.new('RGB', size, color=(0, 0, 0))

    color_stop_positions = sorted(colors.keys())
    color_stop_count = len(color_stop_positions)
    spectrum = []
    for i in range(256):
        start_pos = max(p for p in color_stop_positions if p <= i)
        end_pos = min(p for p in color_stop_positions if p >= i)
        start = colors[start_pos]
        end = colors[end_pos]

        if start_pos == end_pos:
            factor = 0
        else:
            factor = (i - start_pos) / (end_pos - start_pos)

        r = round(start[0] + (end[0] - start[0]) * factor)
        g = round(start[1] + (end[1] - start[1]) * factor)
        b = round(start[2] + (end[2] - start[2]) * factor)
        spectrum.append((r, g, b))

    draw = ImageDraw.Draw(img)
    if mode == 'horizontal':
        for x in range(size[0]):
            pos = int(x * 100 / (size[0] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([round(c / tolerance) * tolerance for c in color])
            draw.line((x, 0, x, size[1]), fill=color)
    elif mode == 'vertical':
        for y in range(size[1]):
            pos = int(y * 100 / (size[1] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([round(c / tolerance) * tolerance for c in color])
            draw.line((0, y, size[0], y), fill=color)

    # The blur hides the banding that ``tolerance`` and the integer spectrum introduce.
    blur = 1.5
    if size[0] > 512 or size[1] > 512:
        multiplier = max(size[0], size[1]) / 512
        if multiplier < 1.5:
            multiplier = 1.5
        blur = blur * multiplier

    img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    return img


def gradient_map(
    image: Image.Image,
    gradient_map_input: Image.Image | None = None,
    reverse: bool = False,
    stops: dict[int, list[int]] | None = None,
) -> Image.Image:
    """Recolour an image by looking each pixel's luminance up in a gradient.

    Args:
        image: Source image. Converted to ``L`` to index the map.
        gradient_map_input: Gradient to read. Used when ``stops`` is not given.
        reverse: Turn the ramp end for end, so its last colour lands in the shadows.
        stops: Colour stops to build the ramp from, as :func:`parse_gradient_stops` reads
            them.

    Returns:
        An ``RGB`` image the size of ``image``.

    Raises:
        ValueError: Neither a gradient nor a stop was given.
    """
    table = _table(gradient_map_input, stops, _resample)
    if reverse:
        table = table[::-1]
    return Image.fromarray(table[np.array(image.convert('L'))])


def _table_from_stops(stops: dict[int, list[int]]) -> np.ndarray:
    """The 256 entry lookup table a set of colour stops describes.

    Args:
        stops: Mapping of stop position, 0 to 100, to ``[r, g, b]``.

    Returns:
        A ``(256, 3)`` uint8 table, interpolated linearly between neighbouring stops.
    """
    positions = sorted(stops)
    levels = np.arange(256, dtype=np.float32) * (100.0 / 255.0)
    channels = [
        np.interp(levels, positions, [stops[position][channel] for position in positions])
        for channel in range(3)
    ]
    return np.rint(np.stack(channels, axis=1)).astype(np.uint8)


def _table(gradient, stops, resample) -> np.ndarray:
    """The 256 entry lookup table a map reads, from stops or from a gradient.

    Args:
        gradient: Gradient to read, or None when ``stops`` is given.
        stops: Colour stops, or None when ``gradient`` is given.
        resample: Callable taking an ``(n, 3)`` line of colours and answering 256 of them.

    Returns:
        A ``(256, 3)`` uint8 table, one colour per greyscale level.

    Raises:
        ValueError: Both were None.
    """
    if stops:
        return _table_from_stops(stops)
    if gradient is None:
        raise ValueError(
            "a gradient map needs either a gradient to read or stops to build one from"
        )
    return resample(_ramp_line(gradient))


def _span(line: np.ndarray) -> float:
    """How far a line of colours travels, summed over the three channels."""
    return float((line.max(axis=0) - line.min(axis=0)).sum())


def _ramp_line(gradient: Image.Image) -> np.ndarray:
    """The line of colours a gradient runs along.

    Args:
        gradient: The gradient to read. Whichever of its two axes travels furthest is the
            one taken, averaged along the other.

    Returns:
        An ``(n, 3)`` float32 line of colours.
    """
    pixels = np.asarray(gradient.convert('RGB')).astype(np.float32)
    across, down = pixels.mean(axis=0), pixels.mean(axis=1)
    return down if _span(down) > _span(across) else across


def _resample(line: np.ndarray) -> np.ndarray:
    """A line of colours as 256 of them, sampled at pixel centres."""
    width = line.shape[0]
    position = np.clip(
        (np.arange(256, dtype=np.float32) + 0.5) * (width / 256.0) - 0.5, 0, width - 1
    )
    left = np.floor(position).astype(np.int32)
    right = np.minimum(left + 1, width - 1)
    across = (position - left)[:, None]
    return np.rint(line[left] * (1.0 - across) + line[right] * across).astype(np.uint8)
