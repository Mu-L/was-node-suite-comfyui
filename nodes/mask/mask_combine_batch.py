"""Flatten a batch of masks into one mask."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.interface import mask_report


class MasksCombineBatch(io.ComfyNode):
    """Sum every mask in a batch into a single mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Masks Combine Batch",
            display_name="Masks Combine Batch",
            search_aliases=["Masks Combine Batch", "flatten masks", "merge batch"],
            category="WAS Suite/Image/Masking",
            description="Sum every mask in the batch and clamp the total to 0-1.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The batch to flatten, such as the output of Mask Batch. Every mask in it "
                        "must share the same width and height."
                    ),
                )
            ],
            outputs=[
                io.Mask.Output(
                    tooltip=(
                        "A single mask covering every area set in any mask of the batch. The "
                        "values are added and then clamped to 0-1, so overlaps come out solid "
                        "white rather than overflowing."
                    )
                )
            ],
        )

    @classmethod
    def execute(cls, masks) -> io.NodeOutput:
        combined_mask = torch.sum(torch.stack([mask.unsqueeze(0) for mask in masks], dim=0), dim=0)
        combined_mask = torch.clamp(combined_mask, 0, 1)
        mask_report.publish(masks, combined_mask, source="masks")
        return io.NodeOutput(combined_mask)
