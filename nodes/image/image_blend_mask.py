"""Blend two images through a mask."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.image import dynamic
from ...modules.convert.tensors import broadcast_image_planes, tensor2pil
from . import stack_images


class ImageBlendByMask(io.ComfyNode):
    """Composite two images using a third as the blend factor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Blend by Mask",
            display_name="Image Blend by Mask",
            search_aliases=["Image Blend by Mask", "composite", "mask blend"],
            category="WAS Suite/Image",
            description=(
                "Blend two images using a mask to say where, then fade the whole result "
                "back towards image_a. The mask is stretched to the size of image_a, so "
                "it does not have to match it."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "Shows through where the mask is white, and sets the output size."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip="Shows through where the mask is black.",
                ),
                io.Image.Input(
                    "mask",
                    tooltip=(
                        "Greyscale image choosing between the two: white keeps image_a, "
                        "black takes image_b, and mid-grey mixes them. Typed IMAGE rather "
                        "than MASK, so a mask has to be converted to an image first."
                    ),
                ),
                io.Float.Input(
                    "blend_percentage",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the masked result is applied. 1.0 uses it as it is, "
                        "0.0 discards it and returns image_a, 0.5 is an even blend of the "
                        "two."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="image_a with image_b blended into it wherever the mask allows.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, mask, blend_percentage) -> io.NodeOutput:
        scale = dynamic.peak(image_a, image_b)
        first = dynamic.fold(image_a, scale)
        frames = broadcast_image_planes(
            first.images, dynamic.fold(image_b, scale).images, mask
        )
        return io.NodeOutput(dynamic.unfold(
            stack_images([cls.composite(a, b, m, blend_percentage) for a, b, m in frames]),
            first,
        ))

    @staticmethod
    def composite(image_a, image_b, mask, blend_percentage: float):
        """Blend one image into another through one mask.

        Args:
            image_a: Image shown where the mask is white, as ``(height, width, channels)``
                in ``[0, 1]``. Sets the size the mask is stretched to.
            image_b: Image shown where the mask is black.
            mask: Greyscale image choosing between the two, at any size.
            blend_percentage: How strongly the masked result is applied, 0.0 to 1.0.

        Returns:
            The blended PIL image.
        """
        from PIL import Image, ImageOps

        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)
        mask = ImageOps.invert(tensor2pil(mask).convert("L"))

        masked_img = Image.composite(img_a, img_b, mask.resize(img_a.size))

        blend_mask = ImageOps.invert(
            Image.new(mode="L", size=img_a.size, color=(round(blend_percentage * 255)))
        )
        return Image.composite(img_a, masked_img, blend_mask)
