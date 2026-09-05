"""Pulling a mask's edges onto the picture it was drawn for."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import mask_planes, stack_masks
from ...modules.image import guided
from ...modules.interface import mask_report


class MaskGuidedFilter(io.ComfyNode):
    """Refine a mask against an image, so its edge lands where the picture's edge is."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMaskGuidedFilter",
            display_name="Mask Guided Filter",
            search_aliases=[
                "WASMaskGuidedFilter", "Mask Guided Filter",
                "refine mask",
                "feather mask",
                "matting",
                "mask edges",
                "snap mask to image",
                "upscale mask",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Refine a mask against the image it belongs to, so its edge follows the "
                "subject instead of sitting beside it. Good for tidying a rough selection, "
                "softening a hard cut-out into a natural edge, and bringing a small or "
                "low-detail mask up to the image's size with the image's own boundaries."
            ),
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to refine; MASK. Smaller than the image, it is lifted to the "
                        "image's size first, so a low-resolution mask can be used as it is."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The picture the mask belongs to; IMAGE. Its edges are the ones the "
                        "mask is pulled onto, colour included."
                    ),
                ),
                io.Int.Input(
                    "radius",
                    default=12,
                    min=1,
                    max=guided.MAX_RADIUS,
                    tooltip=(
                        "How far the mask may move to find the image's edge, in pixels; INT. "
                        "Costs the same at any size. Roughly how wrong the mask's edge is."
                    ),
                ),
                io.Float.Input(
                    "epsilon",
                    default=0.0001,
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    tooltip=(
                        "How closely the mask follows the image; FLOAT. Small values such as "
                        "0.0001 snap hard to every edge, larger ones such as 0.01 leave a "
                        "softer edge that ignores texture."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="masks",
                    tooltip="The refined mask; MASK, at the image's size.",
                ),
            ],
        )

    @classmethod
    def execute(cls, masks, image, radius=12, epsilon=0.0001) -> io.NodeOutput:
        """Refine every mask of the batch against the image.

        Raises:
            ValueError: The mask holds nothing, or the image is not a batch of images.
        """
        planes = mask_planes(masks)
        if not planes:
            raise ValueError(
                "Mask Guided Filter was given no mask. Connect a mask to refine."
            )
        # One channel each, as the filter takes them, and back to the pack's own mask layout.
        stacked = torch.stack([plane.to(torch.float32) for plane in planes]).unsqueeze(-1)
        refined = guided.filter_with_guide(
            stacked, image, radius=int(radius), epsilon=float(epsilon),
        )
        result = stack_masks([frame[..., 0] for frame in refined])
        mask_report.publish(masks, result)
        return io.NodeOutput(result)
