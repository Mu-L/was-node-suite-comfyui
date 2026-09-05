"""Displacement warping.

One function: a per-pixel gather that offsets each source coordinate by the brightness of
a displacement map.
"""

from __future__ import annotations

from PIL import Image

__all__ = ["displace_image"]


def displace_image(
    image: Image.Image,
    displacement_map: Image.Image,
    amplitude: float,
) -> Image.Image:
    """Warp an image by a greyscale displacement map.

    Args:
        image: Source image, converted to ``RGB`` internally.
        displacement_map: Displacement map, converted to ``L`` internally. Must be at
            least as large as ``image``.
        amplitude: Maximum displacement in pixels, at map value 255. Negative values
            displace toward the top left.

    Returns:
        An ``RGB`` image the size of ``image``.

    Raises:
        IndexError: A mirrored coordinate still falls outside the image, which happens
            once ``abs(amplitude)`` exceeds the image's own width or height.
    """

    image = image.convert('RGB')
    displacement_map = displacement_map.convert('L')
    width, height = image.size
    result = Image.new('RGB', (width, height))

    # Cost is one getpixel/putpixel pair per pixel, so it grows with the pixel count
    # rather than with amplitude.
    for y in range(height):
        for x in range(width):

            displacement = displacement_map.getpixel((x, y))
            displacement_amount = amplitude * (displacement / 255)
            new_x = x + int(displacement_amount)
            new_y = y + int(displacement_amount)

            if new_x < 0:
                new_x = abs(new_x)
            elif new_x >= width:
                new_x = 2 * width - new_x - 1

            if new_y < 0:
                new_y = abs(new_y)
            elif new_y >= height:
                new_y = 2 * height - new_y - 1

            if new_x < 0:
                new_x = abs(new_x)
            if new_y < 0:
                new_y = abs(new_y)

            if new_x >= width:
                new_x = 2 * width - new_x - 1
            if new_y >= height:
                new_y = 2 * height - new_y - 1

            pixel = image.getpixel((new_x, new_y))
            result.putpixel((x, y), pixel)

    return result
