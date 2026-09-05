"""Black, mid and white point adjustment."""

from __future__ import annotations

import math

import numpy as np
from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


def adjust_levels(image, black_level: float, mid_level: float, white_level: float):
    """Remap an image's tonal range onto new black, mid and white points.

    Args:
        image: Source PIL image.
        black_level: Input level that becomes black, 0-255.
        mid_level: Input level that becomes mid grey, 0-255. At or below ``black_level``
            the gamma curve is skipped.
        white_level: Input level that becomes white, 0-255.

    Returns:
        A PIL image the same size and mode as the source.

    Raises:
        ZeroDivisionError: ``white_level`` equals ``black_level``, so the stretch divides
            by a zero range; or ``mid_level`` sits exactly on ``white_level``, which puts
            the gamma denominator on ``log(1)``.
        ValueError: ``white_level`` is below ``black_level``, which makes the gamma ratio
            negative and takes the logarithm out of its domain.
    """
    from PIL import Image

    im_arr = np.array(image).astype(np.float32)
    im_arr[im_arr < black_level] = black_level
    im_arr = (im_arr - black_level) * (255 / (white_level - black_level))
    im_arr = np.clip(im_arr, 0, 255)

    if mid_level <= black_level:
        gamma = 1.0
    else:
        gamma = math.log(0.5) / math.log((mid_level - black_level) / (white_level - black_level))
    im_arr = np.power(im_arr / 255, gamma) * 255

    return Image.fromarray(im_arr.astype(np.uint8))


class ImageLevelsAdjustment(io.ComfyNode):
    """Stretch an image's tonal range between a new black, mid and white point."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Levels Adjustment",
            display_name="Image Levels Adjustment",
            search_aliases=[
                "Image Levels Adjustment",
                "levels",
                "black point",
                "white point",
                "gamma",
                "contrast",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Photoshop-style levels. Choose which input brightness becomes black, "
                "which becomes white, and where the midtones sit; everything between is "
                "stretched to fill the range."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to adjust. A batch is handled one image at a time.",
                ),
                io.Float.Input(
                    "black_level",
                    default=0.0,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Brightness that becomes pure black, on a 0-255 scale. Anything darker "
                        "is flattened to black. 0.0 keeps the existing black point; raising it "
                        "to 32 deepens the shadows and crushes whatever was below 32."
                    ),
                ),
                io.Float.Input(
                    "mid_level",
                    default=127.5,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Brightness that becomes mid grey, on a 0-255 scale. 127.5 is the "
                        "midpoint and leaves the midtones alone; a lower value such as 90 "
                        "brightens them, a higher one such as 170 darkens them. Values at or "
                        "below black_level skip the midtone curve entirely."
                    ),
                ),
                io.Float.Input(
                    "white_level",
                    default=255,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Brightness that becomes pure white, on a 0-255 scale. Anything "
                        "brighter is flattened to white. 255 keeps the existing white point; "
                        "lowering it to 200 brightens the whole image and blows out the "
                        "highlights. It must stay above black_level."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with the new tonal range."),
            ],
        )

    @classmethod
    def execute(cls, image, black_level, mid_level, white_level) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(
            image, lambda plane: adjust_levels(plane, black_level, mid_level, white_level)
        ))
