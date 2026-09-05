"""The Dragan high-contrast portrait look."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


class ImageDraganPhotographyFilter(io.ComfyNode):
    """Apply the heavily textured, high-contrast Dragan portrait treatment."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Dragan Photography Filter",
            display_name="Image Dragan Photography Filter",
            search_aliases=[
                "Image Dragan Photography Filter",
                "dragan",
                "portrait",
                "grunge",
                "high pass",
                "clarity",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "The Dragan portrait look: hard contrast with every pore and wrinkle pulled "
                "out by a high-pass layer laid back over the picture. Heavy-handed by "
                "design, and strongest on faces."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to treat. A batch is handled one image at a time. An alpha "
                        "channel is set aside and put back on the result unchanged."
                    ),
                ),
                io.Float.Input(
                    "saturation",
                    default=1.0,
                    min=0.0,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Colour strength of the result. 0.0 drains it to grey, 1.0 leaves it as "
                        "the recolour made it, 2.0 doubles it. Needs colorize on, since with it "
                        "off the result is monochrome and has no colour to strengthen."
                    ),
                ),
                io.Float.Input(
                    "contrast",
                    default=1.0,
                    min=0.0,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Separation between light and dark. 1.0 leaves it alone, 1.5 is a firm "
                        "push, 3.0 is extreme and starts clipping both ends."
                    ),
                ),
                io.Float.Input(
                    "brightness",
                    default=1.0,
                    min=0.0,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Overall exposure, as a multiplier. 1.0 leaves it alone, 0.8 darkens, "
                        "1.2 lightens."
                    ),
                ),
                io.Float.Input(
                    "sharpness",
                    default=1.0,
                    min=0.0,
                    max=6.0,
                    step=0.01,
                    tooltip=(
                        "Edge crispness before the high-pass layer is built. 1.0 leaves it "
                        "alone, 2.0 sharpens, values below 1.0 soften."
                    ),
                ),
                io.Float.Input(
                    "highpass_radius",
                    default=6.0,
                    min=0.0,
                    max=255.0,
                    step=0.01,
                    tooltip=(
                        "Size of the detail the texture layer picks up, in pixels. 2 catches "
                        "only fine grain, 6 catches skin texture, 30 catches broad shapes and "
                        "starts to look like an HDR halo."
                    ),
                ),
                io.Int.Input(
                    "highpass_samples",
                    default=1,
                    min=0,
                    max=6.0,
                    step=1,
                    tooltip=(
                        "How many extra passes of the texture layer are laid over the picture. "
                        "Each one compounds the effect. 0 is treated as 1, so there is always "
                        "at least one extra pass."
                    ),
                ),
                io.Float.Input(
                    "highpass_strength",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.01,
                    tooltip=(
                        "How much of the textured version is mixed back in. 0.0 keeps the plain "
                        "enhanced image, 1.0 uses the textured one outright, and above 1.0 "
                        "overshoots it."
                    ),
                ),
                io.Boolean.Input(
                    "colorize",
                    default=True,
                    tooltip=(
                        "On = the source's colours laid back over the finished monochrome "
                        "result; off = the result stays monochrome and the textured layer is "
                        "desaturated with it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The treated images, one for each that went in and the same size as "
                        "the source, with the source's colours laid back over the monochrome "
                        "result."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        saturation,
        contrast,
        brightness,
        sharpness,
        highpass_radius,
        highpass_samples,
        highpass_strength,
        colorize,
    ) -> io.NodeOutput:
        from ....modules.image.filters import dragan_filter

        return io.NodeOutput(filtered_planes(
            image,
            lambda plane: dragan_filter(
                plane,
                saturation=saturation,
                contrast=contrast,
                sharpness=sharpness,
                brightness=brightness,
                highpass_radius=highpass_radius,
                highpass_samples=highpass_samples,
                highpass_strength=highpass_strength,
                colorize=colorize,
            ),
        ))
