"""Dilate the set area of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import morph_masks


class MaskDilateRegion(io.ComfyNode):
    """Grow the white area of every mask in the batch by binary dilation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Dilate Region",
            display_name="Mask Dilate Region",
            search_aliases=["Mask Dilate Region", "dilate", "grow mask", "morphology"],
            category="WAS Suite/Image/Masking",
            description="Grow the set area of a mask by binary dilation. Any non-zero pixel "
            "counts as set, so grey levels are lost.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to grow. A batch is handled one mask at a time, all by the same "
                        "amount."
                    ),
                ),
                io.Int.Input(
                    "iterations",
                    default=5,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "How many passes of growth to run. Each pass expands the mask by about "
                        "one pixel in every direction, so the default 5 widens it by roughly "
                        "5 pixels and closes gaps up to about 10 pixels wide. The widget stops "
                        "at 1; 0 on a wire grows until nothing changes, filling the frame."
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
                        "The grown mask. Any pixel that was not fully black counted as set, so "
                        "it arrives hard black and white unless blur was above 0."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, iterations, blur=0.0) -> io.NodeOutput:
        return io.NodeOutput(morph_masks(masks, iterations, True, blur))
