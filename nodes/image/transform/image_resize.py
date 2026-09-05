"""Rescale or resize a batch of images."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.image import sizing
from ....modules.image.draw import parse_color
from ....modules.interface import size_report

#: What a resize target is rounded up to a multiple of, in resize mode.
SIZE_STEP = 8

#: How much larger than the target each side of the intermediate is taken where
#: supersampling is on. Eight on a side is 64 times the area.
SUPERSAMPLE_SCALE = 8

#: The pad colour used when pad_color holds something that is not a colour: opaque black,
#: which is what the widget's own default spells.
FALLBACK_PAD = (0, 0, 0, 255)


def resize_target(size, mode="rescale", factor=2, width=1024, height=1024):
    """The size a resize was asked for and the size it can deliver.

    Args:
        size: The source ``(width, height)``.
        mode: ``'rescale'`` to multiply the current size by ``factor``, anything else to
            take ``width`` and ``height``.
        factor: Scale multiplier, used only in ``'rescale'`` mode.
        width: Target width in pixels, used only outside ``'rescale'`` mode.
        height: Target height in pixels, used only outside ``'rescale'`` mode.

    Returns:
        ``(requested, delivered)``, each a ``(width, height)`` pair. In rescale mode the
        request is the factor applied to the longer side with the shorter side derived from
        the source's proportions. In resize mode it is the pair given, and the delivery
        rounds each side up to a multiple of :data:`SIZE_STEP`. Neither side is delivered
        below one pixel.
    """
    current_width, current_height = int(size[0]), int(size[1])

    if mode == "rescale":
        if current_width >= current_height:
            new_width = round(current_width * factor)
            new_height = (
                round(new_width * current_height / current_width) if current_width else 0
            )
        else:
            new_height = round(current_height * factor)
            new_width = (
                round(new_height * current_width / current_height) if current_height else 0
            )
        requested = (new_width, new_height)
    else:
        requested = (int(width), int(height))
        new_width = int(width) + (-int(width) % SIZE_STEP)
        new_height = int(height) + (-int(height) % SIZE_STEP)

    return requested, (max(1, new_width), max(1, new_height))


def apply_resize_image(image, mode="rescale", supersample=True, factor=2,
                       width=1024, height=1024, resample="bicubic"):
    """Resize one image, optionally through an oversampled intermediate.

    Args:
        image: Source PIL image.
        mode: ``'rescale'`` to multiply the current size by ``factor``, anything else to
            take ``width`` and ``height``, each rounded up to the next multiple of 8.
        supersample: Resize to eight times the target first and then down to
            it, which softens aliasing at the cost of an intermediate 64 times the area.
        factor: Scale multiplier, used only in ``'rescale'`` mode. The longer side takes it
            and the shorter side follows the source's proportions.
        width: Target width in pixels, used only outside ``'rescale'`` mode.
        height: Target height in pixels, used only outside ``'rescale'`` mode.
        resample: ``'nearest'``, ``'bilinear'``, ``'bicubic'`` or ``'lanczos'``.

    Returns:
        The resized image, never smaller than one pixel on either side.

    Raises:
        KeyError: ``resample`` is not one of the four filter names.
    """
    from PIL import Image

    _, (new_width, new_height) = resize_target(image.size, mode, factor, width, height)

    resample_filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }

    if supersample:
        image = image.resize(
            (new_width * SUPERSAMPLE_SCALE, new_height * SUPERSAMPLE_SCALE),
            resample=resample_filters[resample],
        )

    return image.resize((new_width, new_height), resample=resample_filters[resample])


def fitted(image, target, resize_mode, resampling, align, pad, supersample=False):
    """One image at exactly the target size, in the mode it arrived in.

    Args:
        image: Source PIL image.
        target: ``(width, height)`` the answer comes out at.
        resize_mode: One of :data:`modules.image.sizing.MODES`.
        resampling: A key of :data:`modules.image.sizing.FILTERS`.
        align: A key of :data:`modules.image.sizing.ALIGNMENTS`.
        pad: ``(red, green, blue, alpha)`` filling whatever the image does not cover.
        supersample: Fit to :data:`SUPERSAMPLE_SCALE` times the target first and resample
            down to it, which softens aliasing at the cost of a far larger intermediate. Not
            applied under :data:`modules.image.sizing.CROP_OR_PAD`, which resamples nothing.

    Returns:
        The image at the target size, back in the mode it was given in, so a batch of RGB
        images stays RGB and one carrying transparency keeps it.
    """
    width, height = target
    if supersample and resize_mode != sizing.CROP_OR_PAD:
        oversized = sizing.fit(
            image, width * SUPERSAMPLE_SCALE, height * SUPERSAMPLE_SCALE,
            resize_mode, resampling, align, pad,
        )
        chosen = sizing.FILTERS.get(resampling, sizing.FILTERS[sizing.DEFAULT_FILTER])
        sized = oversized.resize((width, height), chosen)
    else:
        sized = sizing.fit(image, width, height, resize_mode, resampling, align, pad)
    return sized if sized.mode == image.mode else sized.convert(image.mode)


def rounded(width: int, height: int, multiple_of: int) -> tuple[int, int]:
    """A size held to a multiple, never below one on either side.

    Args:
        width: Width in pixels.
        height: Height in pixels.
        multiple_of: Round both sides down to a multiple of this, 0 or 1 to leave them.

    Returns:
        ``(width, height)``.
    """
    if multiple_of > 1:
        width -= width % multiple_of
        height -= height % multiple_of
    return max(1, int(width)), max(1, int(height))


class ImageResize(io.ComfyNode):
    """Scale every image in a batch by a factor or to a fixed size."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Resize",
            display_name="Image Resize",
            search_aliases=["Image Resize", "Image Rescale", "scale", "upscale", "resize"],
            category="WAS Suite/Image/Transform",
            description=(
                "Scale every image in the batch, either by a multiplier or to an exact "
                "width and height. Rescale mode holds the source proportions. Resize mode "
                "goes to the two sides given, each rounded up to the next multiple of 8, so "
                "a requested 1001 is delivered as 1008, and resize_mode decides how the "
                "picture meets them: padded, cropped, stretched, or left unresampled. "
                "Neither mode goes below one pixel on a side."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The images to scale.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["rescale", "resize"],
                    tooltip=(
                        "`rescale` multiplies the current size by rescale_factor and "
                        "ignores the two size fields. `resize` goes to resize_width by "
                        "resize_height and ignores the factor."
                    ),
                ),
                io.Boolean.Input(
                    "supersample",
                    default=True,
                    tooltip=(
                        "On scales to eight times the target size first and then down "
                        "to it, which smooths jagged edges when enlarging. It builds an "
                        "intermediate image 64 times the target area, so a large target "
                        "needs a great deal of memory; off resizes in one step. Ignored "
                        "under `crop or pad`, which resamples nothing."
                    ),
                ),
                io.Combo.Input(
                    "resampling",
                    options=["lanczos", "nearest", "bilinear", "bicubic"],
                    tooltip=(
                        "How pixels are interpolated in `rescale` mode. `lanczos` is the "
                        "sharpest and the slowest, `bicubic` and `bilinear` are "
                        "progressively softer and quicker, `nearest` copies the closest "
                        "pixel and keeps hard edges and pixel art crisp. `resize` mode "
                        "follows resampling above."
                    ),
                ),
                io.Float.Input(
                    "rescale_factor",
                    default=2,
                    min=0.01,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Size multiplier used in rescale mode. 2.0 doubles both sides, 0.5 "
                        "halves them, 1.0 leaves the size alone."
                    ),
                ),
                io.Int.Input(
                    "resize_width",
                    default=1024,
                    min=1,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Target width in pixels for resize mode, rounded up to the next "
                        "multiple of 8: 1000 gives 1000, 1001 gives 1008."
                    ),
                ),
                io.Int.Input(
                    "resize_height",
                    default=1536,
                    min=1,
                    max=48000,
                    step=1,
                    tooltip=(
                        "Target height in pixels for resize mode, rounded up to the next "
                        "multiple of 8. Set it independently of the width; resize_mode "
                        "decides what becomes of the aspect ratio."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(sizing.MODES),
                    default=sizing.STRETCH,
                    optional=True,
                    tooltip=(
                        "How the picture meets the requested size in `resize` mode. "
                        "`stretch` takes both sides exactly and distorts, which is what this "
                        "node has always done. `fit and pad`: the whole picture inside "
                        "pad_color bars. `fill and crop`: fills the size, the overhang is "
                        "cut. `crop or pad`: no resampling at all. Ignored in `rescale`."
                    ),
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    optional=True,
                    tooltip=(
                        "Which part of the picture survives a crop, and which side takes the "
                        "wider pad bar. `top center` suits portraits, where a centred crop "
                        "takes the forehead off. Ignored in `stretch` and in `rescale` mode."
                    ),
                ),
                io.String.Input(
                    "pad_color",
                    default="#000000",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Fill for space the picture does not cover; STRING. Any Pillow "
                        "colour: `#RRGGBB`, a name, or `#RRGGBBAA`. Empty is transparent, "
                        "which only shows on a batch carrying alpha. Seen in `fit and pad` "
                        "and `crop or pad`. Eg: white"
                    ),
                ),
                io.Int.Input(
                    "multiple_of",
                    default=0,
                    min=0,
                    max=256,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Round both sides of a `resize` down to a multiple of this, which is "
                        "how a size is made safe for a latent. 8 suits most models, 16 and 64 "
                        "some others. 0 leaves the size alone. Ignored in `rescale`."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The scaled images, all at the new size.",
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Width of the delivered images in pixels: the rescaled width in "
                        "`rescale` mode, or the requested width after the multiple of 8 and "
                        "after multiple_of in `resize` mode."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "Height of the delivered images in pixels: the rescaled height in "
                        "`rescale` mode, or the requested height after the multiple of 8 and "
                        "after multiple_of in `resize` mode."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, mode, supersample, resampling, rescale_factor, resize_width,
                resize_height, resize_mode=sizing.STRETCH,
                align=sizing.DEFAULT_ALIGNMENT, pad_color="#000000",
                multiple_of=0) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        source = size_report.frame_size(image)
        requested, delivered = resize_target(
            # A source whose size could not be read falls back to 1 by 1.
            source or (1, 1), mode, rescale_factor, resize_width, resize_height,
        )

        if mode == "resize":
            target = rounded(delivered[0], delivered[1], multiple_of)
            pad = parse_color(pad_color, FALLBACK_PAD)
            planes = [
                pil2tensor(
                    fitted(
                        tensor2pil(img),
                        target,
                        resize_mode,
                        resampling,
                        align,
                        pad,
                        supersample,
                    )
                )
                for img in image
            ]
        else:
            planes = [
                pil2tensor(
                    apply_resize_image(
                        tensor2pil(img),
                        mode,
                        supersample,
                        rescale_factor,
                        resize_width,
                        resize_height,
                        resampling,
                    )
                )
                for img in image
            ]
        scaled = torch.cat(planes, dim=0)

        size_report.publish(
            image,
            scaled,
            action="rescaled" if mode == "rescale" else "resized",
            requested=requested if source else None,
        )
        height, width = int(scaled.shape[1]), int(scaled.shape[2])
        return io.NodeOutput(dynamic.unfold(scaled, folded), width, height)
