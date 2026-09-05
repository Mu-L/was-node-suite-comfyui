"""Keep only the brightest band of a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import ceiling_region


class MaskCeilingRegion(io.ComfyNode):
    """Drop everything but the brightest band of every mask in the batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Ceiling Region",
            display_name="Mask Ceiling Region",
            search_aliases=["Mask Ceiling Region", "ceiling", "brightest region"],
            category="WAS Suite/Image/Masking",
            description="Send every level below 225 to black and every level at or above 250 "
            "to white, leaving the band between them at its original level.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to trim down to its strongest parts, which discards weakly "
                        "masked areas such as the outside of a feathered edge. A batch is handled "
                        "one mask at a time."
                    ),
                )
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "Only the brightest sliver of the mask survives: on a 0-255 brightness "
                        "scale, everything below 225 is cleared to black, 250 and up becomes "
                        "solid white, and the narrow 225-249 band keeps whatever strength it had. "
                        "A mask with no near-white areas comes back empty."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks) -> io.NodeOutput:
        regions = [pil2mask(ceiling_region(image)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
