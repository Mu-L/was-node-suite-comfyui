"""Smoothing an image while holding the edges of a guide."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic, guided


class ImageGuidedFilter(io.ComfyNode):
    """Smooth an image without softening its edges, or transfer another image's edges to it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageGuidedFilter",
            display_name="Image Guided Filter",
            search_aliases=[
                "WASImageGuidedFilter", "Image Guided Filter",
                "edge preserving blur",
                "edge aware smoothing",
                "detail smoothing",
                "denoise keep edges",
                "joint upsample",
                "bilateral alternative",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Smooth an image while keeping its edges crisp, the way a bilateral filter is "
                "meant to but without the halos it leaves. Wire a second image into guide and "
                "that image's edges are the ones kept instead, which also upscales a small "
                "image to the guide's size along the way."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to smooth; IMAGE. Smaller than the guide, it is lifted to "
                        "the guide's size first."
                    ),
                ),
                io.Int.Input(
                    "radius",
                    default=8,
                    min=1,
                    max=guided.MAX_RADIUS,
                    tooltip=(
                        "How far the smoothing reaches, in pixels; INT. Costs the same at any "
                        "size, so a wide radius is as cheap as a narrow one."
                    ),
                ),
                io.Float.Input(
                    "epsilon",
                    default=0.01,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "What still counts as flat, and so gets smoothed; FLOAT. 0.001 keeps "
                        "almost every edge and barely smooths, 0.1 smooths through all but the "
                        "strongest. Squared brightness, so 0.01 is a step of 0.1."
                    ),
                ),
                io.Image.Input(
                    "guide",
                    optional=True,
                    tooltip=(
                        "The image whose edges are kept; IMAGE. Left unconnected the image "
                        "guides itself. Colour guides follow an edge that only changes hue."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The smoothed image; IMAGE, at the guide's size when one is wired.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, radius=8, epsilon=0.01, guide=None) -> io.NodeOutput:
        """Smooth the image against the guide, or against itself.

        Raises:
            ValueError: The image or the guide is not a batch of images.
        """
        folded = dynamic.fold(image)
        led = folded if guide is None else dynamic.fold(guide)
        return io.NodeOutput(dynamic.unfold(
            guided.filter_with_guide(
                folded.images, led.images,
                radius=int(radius), epsilon=float(epsilon),
            ),
            folded,
        ))
