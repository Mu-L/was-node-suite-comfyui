"""Cross-fade two images."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.convert.tensors import broadcast_image_planes, tensor2pil
from ...modules.image import dynamic
from . import quantises_exactly, stack_images


class ImageBlend(io.ComfyNode):
    """Mix two images by a single percentage."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Blend",
            display_name="Image Blend",
            search_aliases=["Image Blend", "cross fade", "mix images", "opacity"],
            category="WAS Suite/Image",
            description=(
                "Fade evenly between two images. When the two are the same size the "
                "result is that size. When they differ, the result takes the size of "
                "image_b, and image_a is laid over its top left corner, so only the "
                "overlapping part is mixed in."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "The image faded from. Where the two images differ in size it is "
                        "laid over the top left corner of image_b, which sets the output "
                        "size."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "The image faded to. A batch here is paired with the image_a batch "
                        "frame by frame, a single image is faded into every frame, and the "
                        "result is as long as the longer of the two."
                    ),
                ),
                io.Float.Input(
                    "blend_percentage",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of image_b to mix in. 0.0 keeps image_a unchanged, 1.0 "
                        "replaces it entirely, 0.5 is an even blend."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The two inputs mixed at the chosen percentage.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, blend_percentage) -> io.NodeOutput:
        scale = dynamic.peak(image_a, image_b)
        first, second = dynamic.fold(image_a, scale), dynamic.fold(image_b, scale)
        pairs = broadcast_image_planes(first.images, second.images)
        if all(quantises_exactly(a) and a.shape == b.shape for a, b in pairs):
            return io.NodeOutput(dynamic.unfold(
                cls.blend(
                    torch.stack([a for a, _ in pairs]),
                    torch.stack([b for _, b in pairs]),
                    blend_percentage,
                ),
                first,
            ))
        return io.NodeOutput(dynamic.unfold(
            stack_images([cls.composite(a, b, blend_percentage) for a, b in pairs]), first
        ))

    @staticmethod
    def blend(images_a: torch.Tensor, images_b: torch.Tensor, blend_percentage: float) -> torch.Tensor:
        """Cross-fade two image batches, frame against frame.

        Args:
            images_a: Batch faded from, as ``(batch, height, width, channels)`` in ``[0, 1]``.
            images_b: Batch faded to, of the same shape.
            blend_percentage: How much of ``images_b`` to mix in, 0.0 to 1.0.

        Returns:
            A float32 batch of the same shape, on the device the inputs are on.
        """
        amount = round(blend_percentage * 255)
        a8 = (255.0 * images_a).clamp(0, 255).to(torch.uint8).to(torch.int32)
        b8 = (255.0 * images_b).clamp(0, 255).to(torch.uint8).to(torch.int32)
        # Integer arithmetic, not a float interpolation: this agrees with Pillow to the
        # value on all 65536 sample pairs at each of the 256 blend amounts, a lerp does not.
        return ((b8 * amount + a8 * (255 - amount) + 127) // 255).to(torch.float32) / 255.0

    @staticmethod
    def composite(image_a: torch.Tensor, image_b: torch.Tensor, blend_percentage: float):
        """Cross-fade one image against another through Pillow.

        Args:
            image_a: Image faded from, as ``(height, width, channels)`` in ``[0, 1]``.
            image_b: Image faded to.
            blend_percentage: How much of ``image_b`` to mix in, 0.0 to 1.0.

        Returns:
            The blended PIL image.
        """
        from PIL import Image, ImageOps

        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)

        blend_mask = Image.new(mode="L", size=img_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        return Image.composite(img_a, img_b, blend_mask)
