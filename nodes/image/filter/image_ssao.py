"""Screen-space ambient occlusion from an image and its depth map."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.image import dynamic

logger = log.get_logger("nodes.image.filter")

#: Largest tile the occlusion pass measures, whatever the widget asks for. The widget
#: accepts up to 512, and every value above this measures tiles of this size.
MAX_TILE_SIZE = 8


def create_ambient_occlusion(rgb_image, depth_image, strength: float = 1.0, radius: float = 30,
                             ao_blur: float = 5, spec_threshold: int = 200,
                             enable_specular_masking: bool = False, tile_size: int = 1):
    """Shade an image with the ambient occlusion implied by its depth map.

    Args:
        rgb_image: Source PIL image.
        depth_image: Depth map. Resized to the source when the two differ.
        strength: Multiplier applied to the raw occlusion values before they are clipped
            back into 0-255.
        radius: Neighbourhood half-width in pixels the occlusion is measured over.
        ao_blur: Gaussian radius in pixels applied to the occlusion field.
        spec_threshold: Brightness above which a source pixel counts as specular, 0-255.
        enable_specular_masking: Hold the specular area at full brightness in the occlusion
            field, so the shading does not touch it.
        tile_size: Side in pixels of the tiles the image is measured in. 1 or less measures
            the whole image at once, which is the only setting where a neighbourhood is not
            cut off at a tile edge. Larger values are clamped to :data:`MAX_TILE_SIZE`.

    Returns:
        ``(composited, occlusion, specular_mask)``. The composite is the source with the
        shading multiplied in; the occlusion field and the specular mask are mode ``L``.
        A pixel occluded more strongly than the 0-255 scale holds is shaded fully black.
    """
    import comfy.utils
    from PIL import Image, ImageChops, ImageFilter

    from ....modules.image.occlusion import calculate_ambient_occlusion_factor

    if depth_image.size != rgb_image.size:
        depth_image = depth_image.resize(rgb_image.size)
    rgb_normalized = np.array(rgb_image, dtype=np.float32) / 255.0
    depth_normalized = np.array(depth_image, dtype=np.float32) / 255.0

    height, width, _ = rgb_normalized.shape

    if tile_size <= 1:
        logger.info("Measuring ambient occlusion over the whole image (highest quality) ...")
        occlusion_array = calculate_ambient_occlusion_factor(
            rgb_normalized, depth_normalized, height, width, radius
        )
    else:
        tile_size = tile_size if tile_size <= MAX_TILE_SIZE else MAX_TILE_SIZE
        num_tiles_x = (width - 1) // tile_size + 1
        num_tiles_y = (height - 1) // tile_size + 1

        occlusion_array = np.zeros((height, width), dtype=np.uint8)
        progress = comfy.utils.ProgressBar(num_tiles_y * num_tiles_x)

        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                tile_left = tile_x * tile_size
                tile_upper = tile_y * tile_size
                tile_right = min(tile_left + tile_size, width)
                tile_lower = min(tile_upper + tile_size, height)

                tile_rgb = rgb_normalized[tile_upper:tile_lower, tile_left:tile_right]
                tile_depth = depth_normalized[tile_upper:tile_lower, tile_left:tile_right]

                occlusion_array[tile_upper:tile_lower, tile_left:tile_right] = (
                    calculate_ambient_occlusion_factor(
                        tile_rgb, tile_depth, tile_rgb.shape[0], tile_rgb.shape[1], radius
                    )
                )
                progress.update(1)

    occlusion_array = (occlusion_array * strength).clip(0, 255).astype(np.uint8)

    occlusion_image = Image.fromarray(occlusion_array, mode='L')
    occlusion_image = occlusion_image.filter(ImageFilter.GaussianBlur(radius=ao_blur))
    occlusion_image = occlusion_image.filter(ImageFilter.SMOOTH)
    occlusion_image = ImageChops.multiply(
        occlusion_image, ImageChops.multiply(occlusion_image, occlusion_image)
    )

    mask = rgb_image.convert('L')
    mask = mask.point(lambda x: x > spec_threshold, mode='1')
    mask = mask.convert("RGB")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.5)).convert("L")

    if enable_specular_masking:
        occlusion_image = Image.composite(
            Image.new("L", rgb_image.size, 255), occlusion_image, mask
        )
    occlusion_result = ImageChops.multiply(rgb_image, occlusion_image.convert("RGB"))

    return occlusion_result, occlusion_image, mask


class ImageAmbientOcclusion(io.ComfyNode):
    """Darken the crevices of an image using the depth map that goes with it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image SSAO (Ambient Occlusion)",
            display_name="Image SSAO (Ambient Occlusion)",
            search_aliases=[
                "Image SSAO (Ambient Occlusion)",
                "ambient occlusion",
                "ssao",
                "contact shadow",
                "depth shading",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Add contact shadows to an image using a depth map: wherever the depth "
                "jumps, the shallower side is darkened, the way light fails to reach a "
                "crevice. Gives a flat render a sense of solidity."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The image to shade. A batch is handled one image at a time.",
                ),
                io.Image.Input(
                    "depth_images",
                    tooltip=(
                        "The matching depth map, where bright means near and dark means far. It "
                        "is resized to the image, so it need not match its size. MiDaS Depth "
                        "Approximation produces a suitable one."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    min=0.0,
                    max=5.0,
                    default=1.0,
                    step=0.01,
                    tooltip=(
                        "How dark the shading goes. 1.0 is the measured amount, 0.5 is half as "
                        "deep, 2.0 exaggerates it. 0.0 removes the shading, leaving the image "
                        "black because the occlusion field itself goes to black."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    min=0.01,
                    max=1024,
                    default=30,
                    step=0.01,
                    tooltip=(
                        "How far around each pixel the depth is compared, in pixels. Small "
                        "values such as 4 give tight outlines around objects; 30 gives broad "
                        "soft shading. Cost grows with the square of this, so large values are "
                        "very slow."
                    ),
                ),
                io.Float.Input(
                    "ao_blur",
                    min=0.01,
                    max=1024,
                    default=2.5,
                    step=0.01,
                    tooltip=(
                        "How much the shading is softened before it is applied, in pixels. 2.5 "
                        "smooths away the pixel-level noise; 20 turns the shading into a broad "
                        "gradient."
                    ),
                ),
                io.Int.Input(
                    "specular_threshold",
                    min=0,
                    max=255,
                    default=25,
                    step=1,
                    tooltip=(
                        "How bright a pixel has to be, on a 0-255 scale, to count as a highlight "
                        "that should not be shaded. 25 protects almost everything that is not "
                        "nearly black; 200 protects only the brightest highlights. Only read "
                        "when enable_specular_masking is on, but it always decides the third "
                        "output."
                    ),
                ),
                io.Boolean.Input(
                    "enable_specular_masking",
                    default=True,
                    tooltip=(
                        "Keep the bright areas picked out by specular_threshold free of shading. "
                        "On protects highlights and light sources from being darkened by their "
                        "own depth edge; off shades the whole image."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    min=1,
                    max=512,
                    default=1,
                    step=1,
                    tooltip=(
                        "Measure the image in square tiles of this many pixels instead of all at "
                        "once. 1 measures the whole image and is the only setting that gives "
                        "correct shading; anything larger cuts each pixel's comparison off at "
                        "the tile edge, which shows as a grid. Values above 8 are treated as 8."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="composited_images",
                    tooltip="The source image with the shading multiplied into it.",
                ),
                io.Image.Output(
                    display_name="ssao_images",
                    tooltip=(
                        "The shading on its own, as a greyscale image: white where light "
                        "reaches, dark in the crevices."
                    ),
                ),
                io.Image.Output(
                    display_name="specular_mask_images",
                    tooltip=(
                        "The area treated as highlight, white where it was protected from "
                        "shading. Produced whether or not the masking was enabled."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, depth_images, strength, radius, ao_blur, specular_threshold,
                enable_specular_masking, tile_size) -> io.NodeOutput:
        folded = dynamic.fold(images)
        maps = dynamic.fold(depth_images).images
        composited = []
        occlusions = []
        speculars = []
        for i, image in enumerate(folded.images):
            logger.info("Processing SSAO image %d/%d ...", i + 1, len(folded.images))
            composited_image, occlusion_image, specular_mask = create_ambient_occlusion(
                tensor2pil(image),
                tensor2pil(maps[i if i < len(maps) else -1]),
                strength=strength,
                radius=radius,
                ao_blur=ao_blur,
                spec_threshold=specular_threshold,
                enable_specular_masking=enable_specular_masking,
                tile_size=tile_size,
            )
            composited.append(pil2tensor(composited_image))
            occlusions.append(pil2tensor(occlusion_image))
            speculars.append(pil2tensor(specular_mask))

        return io.NodeOutput(
            dynamic.unfold(torch.cat(composited, dim=0), folded),
            torch.cat(occlusions, dim=0),
            torch.cat(speculars, dim=0),
        )
