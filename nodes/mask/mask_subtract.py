"""Subtract one mask from another."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import float_mask, same_size_or_refuse
from ...modules.interface import mask_report


class MasksSubtract(io.ComfyNode):
    """Subtract one mask from another."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Masks Subtract",
            display_name="Masks Subtract",
            search_aliases=["Masks Subtract", "mask subtract", "difference", "mask math"],
            category="WAS Suite/Image/Masking",
            description="Subtract masks_b from masks_a and clamp the result at zero.",
            inputs=[
                io.Mask.Input("masks_a", tooltip="The mask to cut away from."),
                io.Mask.Input(
                    "masks_b",
                    tooltip=(
                        "The mask to remove. Wherever this is white the same area of masks_a is "
                        "cleared; where it is grey, masks_a is only partly reduced. Both masks "
                        "must be the same width and height."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "masks_a with masks_b taken out of it. A result that would go below zero "
                        "is clamped to black, so removing more than was there simply leaves "
                        "nothing."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks_a, masks_b) -> io.NodeOutput:
        same_size_or_refuse((("masks_a", masks_a), ("masks_b", masks_b)), "Masks Subtract")
        # Held to the range a MASK socket carries at both ends. Only the floor can bite on a
        # mask already inside it, and an upper bound of 255 would not bite on anything.
        subtracted = torch.clamp(float_mask(masks_a) - float_mask(masks_b), 0.0, 1.0)
        mask_report.publish(masks_a, subtracted, source="masks_a")
        return io.NodeOutput(subtracted)
