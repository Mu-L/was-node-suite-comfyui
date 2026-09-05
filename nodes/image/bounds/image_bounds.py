"""Read the full extent of each image in a batch as bounds."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import IMAGE_BOUNDS


class ImageBounds(io.ComfyNode):
    """Emit one whole-image bounds row per image in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Bounds",
            display_name="Image Bounds",
            search_aliases=["Image Bounds", "full bounds", "image extent", "region"],
            category="WAS Suite/Image/Bound",
            description=(
                "Describe each image's whole area as a bounds value, which is the starting "
                "point for the other bounds nodes: shrink it with Inset Image Bounds, then "
                "crop or blend through that window."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to measure. Only their width and height are read; the "
                        "pixels are not."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    tooltip=(
                        "One row per image, each covering that image edge to edge as "
                        "(top, bottom, left, right) pixel rows and columns."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        image = image.unsqueeze(0) if image.dim() == 3 else image

        return io.NodeOutput(
            [(0, img.shape[0] - 1, 0, img.shape[1] - 1) for img in image]
        )
