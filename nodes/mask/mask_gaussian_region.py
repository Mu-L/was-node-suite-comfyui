"""Feather a mask with a Gaussian blur."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_images, stack_masks
from ...modules.convert.tensors import pil2mask
from ...modules.interface import mask_report
from ...modules.mask.regions import gaussian_region


class MaskGaussianRegion(io.ComfyNode):
    """Blur every mask in the batch by a Gaussian radius."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Gaussian Region",
            display_name="Mask Gaussian Region",
            search_aliases=["Mask Gaussian Region", "feather", "blur mask"],
            category="WAS Suite/Image/Masking",
            description="Feather a mask with a Gaussian blur. The radius is truncated to an "
            "integer, so 5.9 blurs as 5.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The mask to soften, so that whatever it is applied to blends in instead "
                        "of showing a hard boundary. A batch is handled one mask at a time."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    default=5.0,
                    min=0.0,
                    max=1024,
                    step=0.1,
                    tooltip=(
                        "How far the edge is spread, in pixels. 0 leaves the mask alone, 5 gives "
                        "a narrow feather, and values in the hundreds smear the mask into a broad "
                        "gradient. Fractions are dropped, so 5.9 blurs the same as 5."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="MASKS",
                    tooltip=(
                        "The mask with its edges faded through grey rather than cut off sharply."
                    ),
                )
            ],
        )

    @classmethod
    def execute(cls, masks, radius) -> io.NodeOutput:
        regions = [pil2mask(gaussian_region(image, radius)) for image in mask_images(masks)]
        stacked = stack_masks(regions)
        mask_report.publish(masks, stacked)
        return io.NodeOutput(stacked)
