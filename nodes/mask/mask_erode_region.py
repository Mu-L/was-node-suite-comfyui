"""Erode the set area of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import morph_masks


class MaskErodeRegion(io.ComfyNode):
    """Shrink the white area of every mask in the batch by binary erosion."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Erode Region",
            display_name="Mask Erode Region",
            search_aliases=["Mask Erode Region", "erode", "shrink mask", "morphology"],
            category="WAS Suite/Image/Masking",
            description="Shrink the set area of a mask by binary erosion. Any non-zero pixel "
            "counts as set, so grey levels are lost.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to shrink. A batch is handled one mask at a time, all by the "
                        "same amount. A set area running off the edge of the frame keeps its "
                        "border there."
                    ),
                ),
                io.Int.Input(
                    "iterations",
                    default=5,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "How many passes of shrinkage to run. Each pass pulls the mask in by "
                        "about one pixel on every side, so the default 5 narrows it by roughly "
                        "5 pixels and erases anything thinner than about 10 pixels. The widget "
                        "stops at 1; 0 on a wire shrinks until nothing changes, erasing the mask."
                    ),
                ),
                io.Float.Input(
                    "blur",
                    default=0.0,
                    min=0.0,
                    max=128.0,
                    step=0.5,
                    optional=True,
                    tooltip=(
                        "Soften the edge after the shape is settled, as a blur radius in "
                        "pixels. 0 leaves the hard binary edge. A few pixels is what an "
                        "inpaint or a composite wants so the seam does not show. Fractions "
                        "are dropped, so 0.5 blurs the same as 0."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The shrunken mask. Any pixel that was not fully black counted as set, "
                        "so it arrives hard black and white unless blur was above 0."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, iterations, blur=0.0) -> io.NodeOutput:
        return io.NodeOutput(morph_masks(masks, iterations, False, blur))
