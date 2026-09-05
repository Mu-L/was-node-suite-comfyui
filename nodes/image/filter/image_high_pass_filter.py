"""Detail extraction by subtracting a blurred copy."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


def apply_hpf(img, radius: int = 10, strength: float = 1.5, color_output: bool = True,
              neutral_background: bool = True):
    """Keep only the fine detail of an image by subtracting a blurred copy of it.

    Args:
        img: Source PIL image, three-channel.
        radius: Blur radius in pixels. Detail larger than this is removed.
        strength: Multiplier applied to the difference before it is clipped to 0-255.
        color_output: Keep the detail in colour rather than averaging the
            channels to grey.
        neutral_background: Screen the detail over mid grey instead of black.

    Returns:
        An ``RGB`` PIL image the same size as the source.
    """
    from PIL import Image, ImageChops, ImageFilter

    img_arr = np.array(img).astype('float')
    blurred_arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=radius))).astype('float')
    hpf_arr = img_arr - blurred_arr
    hpf_arr = np.clip(hpf_arr * strength, 0, 255).astype('uint8')

    if color_output:
        high_pass = Image.fromarray(hpf_arr, mode='RGB')
    else:
        grayscale_arr = np.mean(hpf_arr, axis=2).astype('uint8')
        high_pass = Image.fromarray(grayscale_arr, mode='L')

    if neutral_background:
        neutral_color = (128, 128, 128) if high_pass.mode == 'RGB' else 128
        neutral_bg = Image.new(high_pass.mode, high_pass.size, neutral_color)
        high_pass = ImageChops.screen(neutral_bg, high_pass)

    return high_pass.convert("RGB")


class ImageHighPassFilter(io.ComfyNode):
    """Strip an image down to its fine detail, discarding everything broad."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image High Pass Filter",
            display_name="Image High Pass Filter",
            search_aliases=[
                "Image High Pass Filter",
                "high pass",
                "detail",
                "clarity",
                "frequency separation",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Keep only the fine detail of an image and throw away the broad shapes and "
                "tones. The result is a texture layer, usually blended back over the "
                "original to sharpen it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The image to extract detail from. A batch is handled image by image.",
                ),
                io.Int.Input(
                    "radius",
                    default=10,
                    min=1,
                    max=500,
                    step=1,
                    tooltip=(
                        "Detail finer than this many pixels is kept and everything broader is "
                        "discarded. 2 keeps only grain and pores, 10 keeps skin and fabric "
                        "texture, 100 keeps most of the picture."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.5,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "How far the extracted detail is amplified. 1.0 is the raw difference, "
                        "which is very dark; 1.5 is the usual working level; 10 is extreme and "
                        "clips most of it to white. 0.0 gives a flat result."
                    ),
                ),
                io.Boolean.Input(
                    "color_output",
                    default=True,
                    tooltip=(
                        "Keep the detail in colour, or average it to grey. Grey is the safer "
                        "choice when the layer is going to be blended back for sharpening, "
                        "since coloured detail can tint the result."
                    ),
                ),
                io.Boolean.Input(
                    "neutral_background",
                    default=True,
                    tooltip=(
                        "Put the detail on mid grey instead of black. Grey is what an overlay or "
                        "soft-light blend expects, because mid grey leaves the layer beneath "
                        "unchanged; black gives an add-style layer."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The extracted detail, as an RGB image the size of the source.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, radius, strength, color_output, neutral_background) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(
            images,
            lambda plane: apply_hpf(
                plane, radius, strength, color_output, neutral_background
            ),
        ))
