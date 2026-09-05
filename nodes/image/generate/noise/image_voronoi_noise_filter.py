"""Render Worley (Voronoi) cellular noise as an image."""

from __future__ import annotations

from comfy_api.latest import io

from .....modules.convert.tensors import pil2tensor


class ImageVoronoiNoiseFilter(io.ComfyNode):
    """Render a cellular noise field from a random set of feature points."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Voronoi Noise Filter",
            display_name="Image Voronoi Noise Filter",
            search_aliases=[
                "Image Voronoi Noise Filter",
                "voronoi",
                "worley",
                "cellular noise",
                "cracks",
            ],
            category="WAS Suite/Image/Generate",
            description=(
                "Generate Voronoi (Worley) cellular noise: scattered points shaded by how "
                "far each pixel is from them, giving a honeycomb of cells for stone, "
                "cracked earth, scales and organic displacement maps."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Width of the generated image, in pixels. Larger canvases take "
                        "noticeably longer, because every pixel is measured against every "
                        "point."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Height of the generated image, in pixels. The points are scattered "
                        "over a square the width of the image, so a canvas taller than it is "
                        "wide leaves its lower part empty of points and smoothly shaded."
                    ),
                ),
                io.Int.Input(
                    "density",
                    default=50,
                    min=10,
                    max=256,
                    step=2,
                    tooltip=(
                        "How many points are scattered, and so how many cells appear. 10 "
                        "gives a few large cells, 50 a medium honeycomb, 256 a fine mosaic "
                        "that takes five times as long as 50."
                    ),
                ),
                io.Int.Input(
                    "modulator",
                    default=0,
                    min=0,
                    max=8,
                    step=1,
                    tooltip=(
                        "Which nearest point each pixel is shaded by. 0 is the closest one "
                        "and gives round cells; 1 is the second closest and outlines the "
                        "boundaries as bright veins; higher values stack more veins for a "
                        "crystalline look."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Where the points land. The same seed always gives the same "
                        "arrangement; change it to scatter them differently. Any whole number; "
                        "`0` is as good a seed as any."
                    ),
                ),
                io.Boolean.Input(
                    "flat",
                    default=False,
                    optional=True,
                    tooltip=(
                        "`off` = each cell shaded as a smooth gradient away from its point; `on` "
                        "= each cell filled with one flat random colour, which is "
                        "the stained-glass Voronoi look and ignores modulator."
                    ),
                ),
                io.Boolean.Input(
                    "RGB_output",
                    default=True,
                    optional=True,
                    tooltip=(
                        "`on` = a three-channel image, which is what the image nodes expect; "
                        "`off` = a single-channel greyscale image, which suits "
                        "a mask conversion; note that a flat render is colour, so this "
                        "flattens it to grey."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The cellular noise, stretched to fill black-to-white.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, width, height, density, modulator, seed, flat=False, RGB_output=True
    ) -> io.NodeOutput:
        from .....modules.image.noise import WorleyNoise

        noise = WorleyNoise(
            height=height,
            width=width,
            density=density,
            option=modulator,
            use_broadcast_ops=True,
            seed=seed,
            flat=flat,
        )

        image = noise.image
        if RGB_output:
            image = image.convert("RGB")
        else:
            image = image.convert("L")

        return io.NodeOutput(pil2tensor(image))
