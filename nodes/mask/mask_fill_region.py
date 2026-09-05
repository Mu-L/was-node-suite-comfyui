"""Fill enclosed holes in a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report, preview
from ...modules.mask import drawn
from ...modules.mask.regions import fill_region


class MaskFillHoles(io.ComfyNode):
    """Close the enclosed holes of every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Fill Holes",
            display_name="Mask Fill Holes",
            search_aliases=["Mask Fill Holes", "fill region", "close holes"],
            category="WAS Suite/Image/Masking",
            description="Fill the enclosed holes of a mask. A hole touching the image border "
            "is open rather than enclosed and is left alone. A mask painted on the node joins "
            "the result in whichever way drawn_combine names.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to close up, such as an outline that should be a solid shape. A "
                        "batch is handled one mask at a time."
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
                        "produces the filled mask alone. The node has to have run once before "
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
                        "The mask with every fully surrounded gap filled in. The result is hard "
                        "black and white, so soft edges are lost, and a gap that reaches the "
                        "image border is not surrounded and stays open. Anything painted on the "
                        "node is joined in afterwards and keeps its own levels."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, drawn_mask="", drawn_combine=drawn.DEFAULT_COMBINE) -> io.NodeOutput:
        regions = [pil2mask(fill_region(image)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        # The mask before the painting joins it, which is what the brush is drawn over. The
        # joined mask would put the painting back into its own backdrop.
        preview.publish_mask_output(stacked)
        painted = drawn.apply(stacked, drawn_mask, drawn_combine)
        mask_report.publish(masks, painted)
        return io.NodeOutput(painted)
