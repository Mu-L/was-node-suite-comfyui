"""Render fractal Perlin noise as an image."""

from __future__ import annotations

from comfy_api.latest import io

from .....modules.convert.tensors import pil2tensor


class ImagePerlinNoise(io.ComfyNode):
    """Sum octaves of Perlin noise into a greyscale image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Perlin Noise",
            display_name="Image Perlin Noise",
            search_aliases=[
                "Image Perlin Noise",
                "perlin",
                "noise",
                "clouds",
                "procedural texture",
            ],
            category="WAS Suite/Image/Generate",
            description=(
                "Generate a greyscale Perlin noise image, the soft cloud-like pattern used "
                "for clouds, terrain, displacement maps and noise masks."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=2048,
                    step=1,
                    tooltip="Width of the generated image, in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=2048,
                    step=1,
                    tooltip="Height of the generated image, in pixels.",
                ),
                io.Int.Input(
                    "scale",
                    default=100,
                    min=2,
                    max=2048,
                    step=1,
                    tooltip=(
                        "Size of one blob of the coarsest octave, in pixels. 100 gives "
                        "clouds about a tenth of a 1024-pixel canvas across; 10 gives a "
                        "busy speckle and 1000 a single soft gradient."
                    ),
                ),
                io.Int.Input(
                    "octaves",
                    default=4,
                    min=0,
                    max=8,
                    step=1,
                    tooltip=(
                        "How many levels of detail are added together. 1 is a single smooth "
                        "layer, 4 adds three progressively finer layers over it, 8 is very "
                        "detailed and slower. 0 adds nothing and renders solid black."
                    ),
                ),
                io.Float.Input(
                    "persistence",
                    default=0.5,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How strongly each finer level shows through, as a fraction of the "
                        "level before it. 0.5 halves it each time, which reads as soft "
                        "clouds; 0.9 keeps almost all of it and looks rough and grainy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Which pattern is drawn. The same seed always gives the same image, "
                        "0 included, so change it for a different one. Feed this socket from "
                        "a seed node to draw a fresh pattern each prompt."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The noise, as a greyscale image with all three channels equal.",
                ),
            ],
        )

    @classmethod
    def execute(cls, width, height, scale, octaves, persistence, seed) -> io.NodeOutput:
        from .....modules.image.noise import perlin_noise

        image = perlin_noise(
            width=width,
            height=height,
            octaves=octaves,
            persistence=persistence,
            scale=scale,
            seed=seed,
        )
        return io.NodeOutput(pil2tensor(image))
