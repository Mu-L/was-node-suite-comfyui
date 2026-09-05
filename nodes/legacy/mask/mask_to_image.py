"""Convert a batch of masks to greyscale images."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.log import get_logger

REQUIRES = "core_dupes"

logger = get_logger("nodes.legacy.mask")


class ConvertMasksToImages(io.ComfyNode):
    """Broadcast every mask in a batch across three channels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Convert Masks to Images",
            display_name="Convert Masks to Images",
            search_aliases=["Convert Masks to Images", "mask to image"],
            category="WAS Suite/Image/Masking",
            description="Deprecated: use core MaskToImage instead. Converts a batch of "
            "masks to greyscale images, one image per mask, with fully masked areas white "
            "and unmasked areas black.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The masks to render as pictures, so they can be previewed, saved or fed "
                        "to a node that only takes images."
                    ),
                )
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGES",
                    tooltip=(
                        "The masks as grey images, one per mask. Fully masked areas come out "
                        "white and unmasked areas black, with partial strengths in between."
                    ),
                )
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, masks) -> io.NodeOutput:
        if masks.ndim == 4:
            tensor = masks.permute(0, 2, 3, 1)
        elif masks.ndim == 3:
            tensor = masks.unsqueeze(-1)
        elif masks.ndim == 2:
            tensor = masks.unsqueeze(0).unsqueeze(-1)
        else:
            logger.error("Invalid input shape. Expected [N, C, H, W] or [H, W].")
            return io.NodeOutput(masks)

        return io.NodeOutput(torch.cat([tensor] * 3, dim=-1))
