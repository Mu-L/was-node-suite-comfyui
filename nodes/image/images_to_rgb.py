"""Force a batch of images to three colour channels."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import filtered_planes


class ImagesToRGB(io.ComfyNode):
    """Convert every image in a batch to RGB."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Images to RGB",
            display_name="Images to RGB",
            search_aliases=["Images to RGB", "convert to rgb", "drop alpha", "colorize"],
            category="WAS Suite/Image",
            description=(
                "Convert every image in the batch to three colour channels. A greyscale "
                "image gains two channels holding the same values, and a transparent image "
                "loses its alpha channel, which fixes nodes that reject anything other "
                "than plain RGB."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to convert. Already-RGB images pass through unchanged."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The same images, each with exactly three colour channels.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(images, lambda plane: plane.convert("RGB")))
