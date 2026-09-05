"""Bloom glow around bright areas."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


def apply_bloom_filter(input_image, radius: float, bloom_factor: float):
    """Screen a soft glow drawn from an image's own edges back over it.

    Args:
        input_image: Source PIL image.
        radius: Blur radius in pixels of the first pass. The glow itself is blurred at
            twice this.
        bloom_factor: Strength of the glow, 0.0 to 1.0.

    Returns:
        A PIL image the same size and mode as the source.
    """
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter

    blurred_image = input_image.filter(ImageFilter.GaussianBlur(radius=radius))
    high_pass_filter = ImageChops.subtract(input_image, blurred_image)
    bloom_filter = high_pass_filter.filter(ImageFilter.GaussianBlur(radius=radius * 2))
    bloom_filter = ImageEnhance.Brightness(bloom_filter).enhance(2.0)

    level = int(255 * bloom_factor)
    bloom_filter = ImageChops.multiply(
        bloom_filter, Image.new('RGB', input_image.size, (level, level, level))
    )

    return ImageChops.screen(input_image, bloom_filter)


class ImageBloomFilter(io.ComfyNode):
    """Add a soft glow that spills out of an image's bright areas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Bloom Filter",
            display_name="Image Bloom Filter",
            search_aliases=["Image Bloom Filter", "bloom", "glow", "halation", "light bleed"],
            category="WAS Suite/Image/Filter",
            description=(
                "Add a soft halo of light around the bright, detailed parts of an image, "
                "the way a camera lens blooms when it points at a light source."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to add the glow to. A batch is handled one image at a time.",
                ),
                io.Float.Input(
                    "radius",
                    default=10,
                    min=0.0,
                    max=1024,
                    step=0.1,
                    tooltip=(
                        "How far the glow spreads, in pixels. 2 gives a tight sheen on edges, "
                        "10 a visible halo, 50 a broad wash of light over the whole frame. 0 "
                        "leaves the image unchanged."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=1,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip=(
                        "How bright the glow is. 0.0 adds nothing, 0.3 is a subtle lift, 1.0 is "
                        "the full effect."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with the glow screened over it."),
            ],
        )

    @classmethod
    def execute(cls, image, radius, intensity) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(
            image, lambda plane: apply_bloom_filter(plane, radius, intensity)
        ))
