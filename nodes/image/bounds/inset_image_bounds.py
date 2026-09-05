"""Shrink a set of image bounds inwards."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import IMAGE_BOUNDS


class InsetImageBounds(io.ComfyNode):
    """Pull each bounds row in from its four edges."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Inset Image Bounds",
            display_name="Inset Image Bounds",
            search_aliases=["Inset Image Bounds", "shrink bounds", "inset", "margin"],
            category="WAS Suite/Image/Bound",
            description=(
                "Move every edge of a bounds inwards by a set number of pixels, which turns "
                "a whole-image bounds from Image Bounds into a smaller window to crop or "
                "blend through."
            ),
            inputs=[
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "The bounds to shrink. Every row is inset by the same four amounts."
                    ),
                ),
                io.Int.Input(
                    "inset_left",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Pixels taken off the left edge. 0 leaves that edge where it is."
                    ),
                ),
                io.Int.Input(
                    "inset_right",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Pixels taken off the right edge.",
                ),
                io.Int.Input(
                    "inset_top",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Pixels taken off the top edge.",
                ),
                io.Int.Input(
                    "inset_bottom",
                    default=64,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Pixels taken off the bottom edge. Insets that meet or cross each "
                        "other leave nothing behind and raise an error, so keep the total "
                        "below the size of the bounds."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    tooltip="The same rows, each pulled in by the four amounts.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_bounds, inset_left, inset_right, inset_top,
                inset_bottom) -> io.NodeOutput:
        inset_bounds = []
        for rmin, rmax, cmin, cmax in image_bounds:
            rmin += inset_top
            rmax -= inset_bottom
            cmin += inset_left
            cmax -= inset_right

            if rmin > rmax or cmin > cmax:
                raise ValueError(
                    "Invalid insets provided. Please make sure the insets do not exceed the "
                    "image bounds."
                )

            inset_bounds.append((rmin, rmax, cmin, cmax))

        return io.NodeOutput(inset_bounds)
