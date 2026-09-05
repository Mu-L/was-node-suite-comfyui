"""The pixel dimensions of an image, as numbers."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER
from ...modules.convert.tensors import image_planes, tensor2pil


class ImageSizeToNumber(io.ComfyNode):
    """Emit an image's width and height on NUMBER, FLOAT and INT sockets.

    The size comes from the first image of the batch.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Size to Number",
            display_name="Image Size to Number",
            search_aliases=["Image Size to Number", "width", "height", "dimensions", "size"],
            category="WAS Suite/Number/Operations",
            description=(
                "Measure an image and emit its width and height in pixels, on one pair of "
                "sockets per numeric type, so an existing image's size can drive a latent, "
                "a resize or a crop."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to measure. Nothing about it is changed, and a batch is "
                        "measured at its first image, since each output carries one value."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="width_num",
                    tooltip="Width in pixels, on a NUMBER socket.",
                ),
                NUMBER.Output(
                    display_name="height_num",
                    tooltip="Height in pixels, on a NUMBER socket.",
                ),
                io.Float.Output(
                    display_name="width_float",
                    tooltip="The same width as a float, so 512 leaves here as 512.0.",
                ),
                io.Float.Output(
                    display_name="height_float",
                    tooltip="The same height as a float, so 512 leaves here as 512.0.",
                ),
                io.Int.Output(
                    display_name="width_int",
                    tooltip="The same width as an INT, for a core node's width widget.",
                ),
                io.Int.Output(
                    display_name="height_int",
                    tooltip="The same height as an INT, for a core node's height widget.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames the batch holds. 1 for a single picture, which is "
                        "what makes this answer for a video sequence as well as a still."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        planes = image_planes(image)
        width, height = tensor2pil(planes[0]).size
        return io.NodeOutput(
            width, height, float(width), float(height), width, height, len(planes)
        )
