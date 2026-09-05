"""Describe an image's aspect ratio numerically and in words."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import NUMBER
from ...modules.convert.tensors import image_planes, tensor2pil


class ImageAspectRatio(io.ComfyNode):
    """Report the aspect ratio of an image, or of an explicit width and height."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Aspect Ratio",
            display_name="Image Aspect Ratio",
            search_aliases=["Image Aspect Ratio", "aspect ratio", "orientation"],
            category="WAS Suite/Number/Operations",
            description=(
                "Measure an image, or a given width and height, and report the shape of it: "
                "the ratio as a number, its common form such as 16:9, and whether it is "
                "landscape, portrait or square."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "The image to measure. Only needed when width and height are not "
                        "both given; with no image and no pair of sizes the node stops with "
                        "an error. A batch is measured at its first image, since each output "
                        "carries one value."
                    ),
                ),
                io.MultiType.Input(
                    "width",
                    [NUMBER, io.Int, io.Float],
                    optional=True,
                    tooltip=(
                        "Width in pixels, measured instead of the image. It is used only "
                        "when height is given too, and a value of 0 counts as not given."
                    ),
                ),
                io.MultiType.Input(
                    "height",
                    [NUMBER, io.Int, io.Float],
                    optional=True,
                    tooltip=(
                        "Height in pixels, measured instead of the image. It is used only "
                        "when width is given too, and a value of 0 counts as not given."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="aspect_number",
                    tooltip=(
                        "Width divided by height: roughly 1.78 for 1920x1080, 0.5625 for "
                        "1080x1920, and exactly 1.0 for a square."
                    ),
                ),
                io.Float.Output(
                    display_name="aspect_float",
                    tooltip="The same width-over-height ratio, on a FLOAT socket.",
                ),
                NUMBER.Output(
                    display_name="is_landscape_bool",
                    tooltip=(
                        "1 when the image is wider than it is tall, 0 when it is taller or "
                        "square. A number rather than a true/false value."
                    ),
                ),
                io.String.Output(
                    display_name="aspect_ratio_common",
                    tooltip=(
                        "The ratio reduced to whole numbers, such as 16:9 for 1920x1080 or "
                        "1:1 for a square. A fractional width or height is cut to a whole "
                        "number before reducing."
                    ),
                ),
                io.String.Output(
                    display_name="aspect_type",
                    tooltip=(
                        "The orientation as a word: 'landscape', 'portrait' or 'square', "
                        "ready to drop into a prompt or a file name."
                    ),
                ),
                io.Float.Output(
                    display_name="is_landscape_float",
                    tooltip="The same landscape flag as 1.0 or 0.0.",
                ),
                io.Int.Output(
                    display_name="is_landscape_int",
                    tooltip="The same landscape flag as 1 or 0, on an INT socket.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image=None, width=None, height=None) -> io.NodeOutput:
        if not (width and height):
            if image is None:
                raise ValueError(
                    "Image Aspect Ratio needs both width and height when no image is connected."
                )
            width, height = tensor2pil(image_planes(image)[0]).size

        aspect_ratio = width / height
        if aspect_ratio > 1:
            aspect_type = "landscape"
        elif aspect_ratio < 1:
            aspect_type = "portrait"
        else:
            aspect_type = "square"
        landscape_bool = 1 if aspect_type == "landscape" else 0

        # The common ratio is whole numbers reduced by their divisor, so a float width or
        # height is truncated rather than handed to math.gcd, which only takes integers.
        whole_width, whole_height = int(width), int(height)
        divisor = math.gcd(whole_width, whole_height)
        aspect_ratio_common = f"{whole_width // divisor}:{whole_height // divisor}"

        return io.NodeOutput(
            aspect_ratio,
            aspect_ratio,
            landscape_bool,
            aspect_ratio_common,
            aspect_type,
            float(landscape_bool),
            landscape_bool,
        )
