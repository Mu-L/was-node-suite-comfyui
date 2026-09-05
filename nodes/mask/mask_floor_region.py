"""Binarize a mask at its own darkest non-black level."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import floor_region


class MaskFloorRegion(io.ComfyNode):
    """Binarize every mask in the batch against its own floor level."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Floor Region",
            display_name="Mask Floor Region",
            search_aliases=["Mask Floor Region", "floor", "binarize"],
            category="WAS Suite/Image/Masking",
            description="Binarize a mask at the smallest non-zero level present in it, so a "
            "mask holding a single grey level collapses to solid black.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to harden. There is no threshold to set: the cut-off is read "
                        "out of the mask itself, so a batch is handled one mask at a time and "
                        "each may split at a different level."
                    ),
                )
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The mask as pure black and white, split just above its own faintest "
                        "visible level, which keeps every trace of the mask however weak. A mask "
                        "painted in one flat grey has nothing above its faintest level and comes "
                        "back empty."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks) -> io.NodeOutput:
        regions = [pil2mask(floor_region(image)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
