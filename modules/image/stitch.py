"""Compositing two images with a feathered seam.

:func:`stitch_image` joins two images along one edge through a gradient mask.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

__all__ = ["stitch_image"]


def stitch_image(
    image_a: Image.Image,
    image_b: Image.Image,
    mode: str = 'right',
    fuzzy_zone: int = 50,
) -> Image.Image:
    """Join two images along one edge, blended across a feathered seam.

    Args:
        image_a: First image, converted to ``RGB`` internally.
        image_b: Second image, converted to ``RGB`` internally.
        mode: Which side of ``image_a`` ``image_b`` is placed on: ``'top'``,
            ``'bottom'``, ``'left'`` or ``'right'``. Any other value leaves both masks
            unset.
        fuzzy_zone: Width of the blended overlap in pixels, held one pixel short of the two
            images' combined length along the joining axis, since the seam is cut out of the
            pair and a wider zone would leave no canvas.

    Returns:
        An ``RGB`` image holding both inputs.

    Raises:
        AttributeError: ``mode`` is not one of the four supported values, which leaves
            the paste masks at ``None``.
    """

    def linear_gradient(start_color, end_color, size, start, end, mode='horizontal'):
        width, height = size
        gradient = Image.new('RGB', (width, height), end_color)
        draw = ImageDraw.Draw(gradient)

        for i in range(0, start):
            if mode == "horizontal":
                draw.line((i, 0, i, height-1), start_color)
            elif mode == "vertical":
                draw.line((0, i, width-1, i), start_color)

        for i in range(start, end):
            if mode == "horizontal":
                curr_color = (
                    int(start_color[0] + (float(i - start) / (end - start)) * (end_color[0] - start_color[0])),
                    int(start_color[1] + (float(i - start) / (end - start)) * (end_color[1] - start_color[1])),
                    int(start_color[2] + (float(i - start) / (end - start)) * (end_color[2] - start_color[2]))
                )
                draw.line((i, 0, i, height-1), curr_color)
            elif mode == "vertical":
                curr_color = (
                    int(start_color[0] + (float(i - start) / (end - start)) * (end_color[0] - start_color[0])),
                    int(start_color[1] + (float(i - start) / (end - start)) * (end_color[1] - start_color[1])),
                    int(start_color[2] + (float(i - start) / (end - start)) * (end_color[2] - start_color[2]))
                )
                draw.line((0, i, width-1, i), curr_color)

        for i in range(end, width if mode == 'horizontal' else height):
            if mode == "horizontal":
                draw.line((i, 0, i, height-1), end_color)
            elif mode == "vertical":
                draw.line((0, i, width-1, i), end_color)

        return gradient

    image_a = image_a.convert('RGB')
    image_b = image_b.convert('RGB')

    # The seam is cut out of the pair, so a zone as wide as both of them together asks for a
    # canvas of nothing. Held one pixel short of that and no further: any tighter bound would
    # move a two-image stitch, which has answered the same size since v2.
    across = mode in ('right', 'left')
    span = (image_a.size[0] + image_b.size[0]) if across else (image_a.size[1] + image_b.size[1])
    fuzzy_zone = max(0, min(int(fuzzy_zone), span - 1))

    offset = int(fuzzy_zone / 2)
    canvas_width = int(image_a.size[0] + image_b.size[0] - fuzzy_zone) if across else image_a.size[0]
    canvas_height = int(image_a.size[1] + image_b.size[1] - fuzzy_zone) if mode == 'top' or mode == 'bottom' else image_a.size[1]
    canvas = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))

    im_ax = 0
    im_ay = 0
    im_bx = 0
    im_by = 0

    image_a_mask = None
    image_b_mask = None

    if mode == 'top':

        image_a_mask = linear_gradient((0, 0, 0), (255, 255, 255), image_a.size, 0, fuzzy_zone, 'vertical')
        image_b_mask = linear_gradient((255, 255, 255), (0, 0, 0), image_b.size, int(image_b.size[1] - fuzzy_zone), image_b.size[1], 'vertical')
        im_ay = image_b.size[1] - fuzzy_zone

    elif mode == 'bottom':

        image_a_mask = linear_gradient((255, 255, 255), (0, 0, 0), image_a.size, int(image_a.size[1] - fuzzy_zone), image_a.size[1], 'vertical')
        image_b_mask = linear_gradient((0, 0, 0), (255, 255, 255), image_b.size, 0, fuzzy_zone, 'vertical').convert('L')
        im_by = image_a.size[1] - fuzzy_zone

    elif mode == 'left':

        image_a_mask = linear_gradient((0, 0, 0), (255, 255, 255), image_a.size, 0, fuzzy_zone, 'horizontal')
        image_b_mask = linear_gradient((255, 255, 255), (0, 0, 0), image_b.size, int(image_b.size[0] - fuzzy_zone), image_b.size[0], 'horizontal')
        im_ax = image_b.size[0] - fuzzy_zone

    elif mode == 'right':

        image_a_mask = linear_gradient((255, 255, 255), (0, 0, 0), image_a.size, int(image_a.size[0] - fuzzy_zone), image_a.size[0], 'horizontal')
        image_b_mask = linear_gradient((0, 0, 0), (255, 255, 255), image_b.size, 0, fuzzy_zone, 'horizontal')
        im_bx = image_a.size[0] - fuzzy_zone

    Image.Image.paste(canvas, image_a, (im_ax, im_ay), image_a_mask.convert('L'))
    Image.Image.paste(canvas, image_b, (im_bx, im_by), image_b_mask.convert('L'))

    return canvas
