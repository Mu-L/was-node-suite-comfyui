"""Render a Perlin power fractal as an image."""

from __future__ import annotations

from comfy_api.latest import io

from .....modules.convert.tensors import pil2tensor


class ImagePerlinPowerFractal(io.ComfyNode):
    """Sum octaves of Perlin noise with a per-octave frequency and amplitude curve."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Perlin Power Fractal",
            display_name="Image Perlin Power Fractal",
            search_aliases=[
                "Image Perlin Power Fractal",
                "perlin",
                "fractal noise",
                "fbm",
                "procedural texture",
            ],
            category="WAS Suite/Image/Generate",
            description=(
                "Generate a greyscale fractal noise image with control over how quickly the "
                "detail levels get finer and fainter. Suits marble, terrain and cloud "
                "textures that need more structure than plain Perlin noise."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=8192,
                    step=1,
                    tooltip="Width of the generated image, in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=8192,
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
                io.Float.Input(
                    "lacunarity",
                    default=2.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How much finer each level is than the one before it. 2.0 halves the "
                        "blob size each time, which is the usual fractal look; 1.0 makes "
                        "every level the same size, and 4.0 jumps straight from big shapes "
                        "to fine grain with nothing between."
                    ),
                ),
                io.Float.Input(
                    "exponent",
                    default=2.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "Sharpens the fall-off between levels. 1.0 leaves persistence as it "
                        "is; higher values fade the fine levels away faster and leave "
                        "smoother, more billowy shapes."
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
    def execute(
        cls, width, height, scale, octaves, persistence, lacunarity, exponent, seed
    ) -> io.NodeOutput:
        from .....modules.image.noise import perlin_power_fractal

        image = perlin_power_fractal(
            width=width,
            height=height,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            exponent=exponent,
            scale=scale,
            seed=seed,
        )
        return io.NodeOutput(pil2tensor(image))
