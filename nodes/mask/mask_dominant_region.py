"""Subtract the largest connected region of a mask's unset area."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report, preview
from ...modules.mask import drawn
from ...modules.mask.regions import dominant_region


class MaskDominantRegion(io.ComfyNode):
    """Clear the largest connected unset region of every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Dominant Region",
            display_name="Mask Dominant Region",
            search_aliases=["Mask Dominant Region", "largest region", "biggest blob"],
            category="WAS Suite/Image/Masking",
            description="Clear the largest connected region of a mask's unset area and set "
            "everything else. The mask is inverted before the regions are labelled and inverted "
            "again on the way out, so the region that is found is subtracted rather than kept: "
            "on a mask whose unset area is one connected region, the result is the input as hard "
            "black and white. A mask painted on the node joins the result in whichever way "
            "drawn_combine names.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to subtract a region from. A batch is handled one mask at a "
                        "time, each losing its own largest unset region."
                    ),
                ),
                io.Int.Input(
                    "threshold",
                    default=128,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Brightness cut-off on a 0-255 scale. The mask is inverted before the "
                        "regions are found, so a pixel joins the search when its own level is "
                        "below 255 minus this value: at the default 128 that is everything dimmer "
                        "than 127, and a higher value takes in less of the mask's unset area."
                    ),
                ),
                io.String.Input(
                    "drawn_mask",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "The mask painted on the node, written by the interface and saved with "
                        "the workflow. Empty means nothing was painted, and the node then "
                        "produces the region mask alone. The node has to have run once before "
                        "there is a mask to paint on. Clear the field to remove the painting."
                    ),
                ),
                io.Combo.Input(
                    "drawn_combine",
                    options=list(drawn.COMBINE_MODES),
                    default=drawn.DEFAULT_COMBINE,
                    optional=True,
                    tooltip=(
                        "How the painting joins the mask this node produced. union keeps "
                        "whichever of the two is brighter at each pixel, subtract takes the "
                        "painting away from the mask, intersect keeps whichever is darker, and "
                        "off ignores the painting without deleting it. The painting is resized "
                        "to the mask when the two differ."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "Everything outside the largest connected region of the input's unset "
                        "area, as hard black and white with no grey levels left. On a mask "
                        "whose unset area is one connected region that is the input mask "
                        "itself; where it is broken into pieces, all but the largest come back "
                        "set. Anything painted on the node is joined in."
                    ),
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        masks,
        threshold,
        drawn_mask="",
        drawn_combine=drawn.DEFAULT_COMBINE,
    ) -> io.NodeOutput:
        regions = [pil2mask(dominant_region(image, threshold)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        # The mask before the painting joins it, which is what the brush is drawn over. The
        # joined mask would put the painting back into its own backdrop.
        preview.publish_mask_output(stacked)
        painted = drawn.apply(stacked, drawn_mask, drawn_combine)
        mask_report.publish(masks, painted)
        return io.NodeOutput(painted)
