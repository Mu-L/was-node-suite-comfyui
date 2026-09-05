"""Replace one colour throughout an image with another."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic

#: Pillow's ITU-R 601-2 luminance weights, and the rounding and shift that turn the
#: weighted sum back into an 8-bit level. ``Image.convert("L")`` computes
#: ``(r * 19595 + g * 38470 + b * 7471 + 32768) >> 16``, and this transcription of it was
#: checked against Pillow over all 16777216 colours.
LUMA_WEIGHTS = (19595, 38470, 7471)
LUMA_ROUNDING = 0x8000
LUMA_SHIFT = 16


class ImageRemoveColor(io.ComfyNode):
    """Swap every pixel close to one colour for another colour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Remove Color",
            display_name="Image Remove Color",
            search_aliases=["Image Remove Color", "replace colour", "chroma key"],
            category="WAS Suite/Image/Process",
            description=(
                "Find every pixel close to one colour and repaint it in another, turning "
                "a white background black, for instance."
            ),
            inputs=[
                io.Image.Input("image", tooltip="The image to repaint."),
                io.Int.Input(
                    "target_red",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the colour being looked for, 0 to 255.",
                ),
                io.Int.Input(
                    "target_green",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the colour being looked for, 0 to 255.",
                ),
                io.Int.Input(
                    "target_blue",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the colour being looked for, 0 to 255. The three "
                        "together default to white."
                    ),
                ),
                io.Int.Input(
                    "replace_red",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the colour painted over every match, 0 to 255.",
                ),
                io.Int.Input(
                    "replace_green",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the colour painted over every match, 0 to 255.",
                ),
                io.Int.Input(
                    "replace_blue",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the colour painted over every match, 0 to 255. Leave all "
                        "three at 255 and the matched area is painted white; set them to 0 for "
                        "black."
                    ),
                ),
                io.Int.Input(
                    "clip_threshold",
                    default=10,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "How far a pixel may differ from the target colour and still be "
                        "repainted. 0 repaints only exact matches, 10 tolerates slight "
                        "gradients and compression noise, and a high value repaints most of "
                        "the image."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The image with every matched pixel repainted in the replacement colour.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        target_red=255,
        target_green=255,
        target_blue=255,
        replace_red=255,
        replace_green=255,
        replace_blue=255,
        clip_threshold=10,
    ) -> io.NodeOutput:
        folded = dynamic.fold(image)
        image = folded.images
        return io.NodeOutput(dynamic.unfold(
            cls.apply_remove_color(
                image,
                clip_threshold,
                (target_red, target_green, target_blue),
                (replace_red, replace_green, replace_blue),
            ),
            folded,
        ))

    @classmethod
    def apply_remove_color(cls, image, threshold=10, color=(255, 255, 255), rep_color=(0, 0, 0)):
        """Repaint the pixels of a batch that are near one colour.

        Args:
            image: Image tensor shaped ``(batch, height, width, channels)`` scaled to
                ``[0, 1]``. Only the first three channels are read, so an alpha channel is
                dropped; a single-channel image is read as grey in all three.
            threshold: Greatest greyscale difference from ``color`` that still counts as a
                match. The comparison is strict, so a difference exactly equal to the
                threshold is a match.
            color: The ``(r, g, b)`` colour being looked for.
            rep_color: The ``(r, g, b)`` colour painted over every match.

        Returns:
            A three-channel image tensor the size of the input, on the device it arrived
            on.
        """
        quantised = torch.clamp(image * 255.0, 0, 255).to(torch.uint8)
        if quantised.shape[-1] == 1:
            quantised = quantised.expand(*quantised.shape[:-1], 3)
        pixels = quantised[..., :3].to(torch.int32)

        target = torch.tensor(
            [int(value) for value in color], dtype=torch.int32, device=image.device
        )
        weights = torch.tensor(LUMA_WEIGHTS, dtype=torch.int32, device=image.device)
        grey = (((pixels - target).abs() * weights).sum(dim=-1) + LUMA_ROUNDING) >> LUMA_SHIFT

        replacement = torch.tensor(
            [int(value) for value in rep_color], dtype=torch.int32, device=image.device
        )
        repainted = torch.where((grey <= threshold).unsqueeze(-1), replacement, pixels)
        return repainted.to(torch.float32) / 255.0
