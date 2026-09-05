"""Invert a mask."""

from __future__ import annotations

from comfy_api.latest import io

from . import float_mask
from ...modules.interface import mask_report


class MaskInvert(io.ComfyNode):
    """Swap the set and unset areas of a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Invert",
            display_name="Mask Invert",
            search_aliases=["Mask Invert", "invert mask", "negate"],
            category="WAS Suite/Image/Masking",
            description="Invert a mask, so what was masked becomes unmasked.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip="The mask to flip. A batch of masks is inverted all at once.",
                )
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The mask with black and white swapped. Grey levels are mirrored rather "
                        "than dropped, so a half-strength area stays half strength."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks) -> io.NodeOutput:
        inverted = 1.0 - float_mask(masks)
        mask_report.publish(masks, inverted)
        return io.NodeOutput(inverted)
