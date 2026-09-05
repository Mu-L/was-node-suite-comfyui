"""Add two masks together."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import float_mask, same_size_or_refuse
from ...modules.interface import mask_report


class MasksAdd(io.ComfyNode):
    """Sum two masks."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Masks Add",
            display_name="Masks Add",
            search_aliases=["Masks Add", "mask add", "union", "mask math"],
            category="WAS Suite/Image/Masking",
            description="Add masks_b to masks_a and hold the result to the 0 to 1 a mask "
            "carries, so an area covered by both comes out fully set rather than twice set.",
            inputs=[
                io.Mask.Input(
                    "masks_a",
                    tooltip=(
                        "The mask to add to. Anything set here stays set in the result, so this "
                        "is the base of the union."
                    ),
                ),
                io.Mask.Input(
                    "masks_b",
                    tooltip=(
                        "The mask added on top. Both masks must be the same width and height, "
                        "and areas set in either one end up set in the result."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The two masks added pixel by pixel and clamped to 0 to 1. Where they "
                        "overlap the values sum, so a half-strength area under another "
                        "half-strength area comes out fully set."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks_a, masks_b) -> io.NodeOutput:
        same_size_or_refuse((("masks_a", masks_a), ("masks_b", masks_b)), "Masks Add")
        if masks_a.ndim > 2 and masks_b.ndim > 2:
            added_masks = masks_a + masks_b
        else:
            added_masks = (masks_a.unsqueeze(1) + masks_b.unsqueeze(1)).squeeze(1)
        # A MASK socket carries 0 to 1, and two masks overlapping sum past it, so the sum is
        # held to the range every reader of a mask assumes.
        added_masks = torch.clamp(float_mask(added_masks), 0.0, 1.0)
        mask_report.publish(masks_a, added_masks, source="masks_a")
        return io.NodeOutput(added_masks)
