"""Matching a batch's colour to one reference frame or clip."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.image.color_match import COLOR_SPACES, METHODS, color_match


class ImageColorMatch(io.ComfyNode):
    """Match ``images`` to ``reference`` with one fixed transform applied to every frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Color Match",
            display_name="Image Color Match",
            search_aliases=[
                "Image Color Match", "color match", "colour match", "colour transfer",
                "color grade", "consistent color",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Match a batch's colour to one reference, in one fixed transform applied "
                "to every frame, so the result cannot flicker."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Batch to correct; IMAGE. Every frame gets the same transform.",
                ),
                io.Image.Input(
                    "reference",
                    tooltip=(
                        "Target colour; IMAGE. A batch is pooled into one distribution, "
                        "not just its first frame."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    options=METHODS,
                    default="mkl",
                    tooltip=(
                        "`mkl`: matches full covariance, keeps hue accurate. `reinhard`: "
                        "matches mean/std per channel, cheap, can shift hue. `histogram`: "
                        "matches the whole tonal curve, fixes contrast/gamma too, can "
                        "blotch without regrain_strength."
                    ),
                ),
                io.Combo.Input(
                    "color_space",
                    options=COLOR_SPACES,
                    default="Lab",
                    tooltip=(
                        "`Lab` separates brightness from colour, so `mkl` and `histogram` "
                        "hold hue steadier. `RGB` is cheaper and skips a colour-space "
                        "round trip."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How much of the match applies; FLOAT. 0 = unchanged, 1 = full match.",
                ),
                io.Boolean.Input(
                    "luminance_only",
                    default=False,
                    tooltip=(
                        "`reinhard` only: match brightness, leave the batch's own colour "
                        "cast alone. No effect on `mkl` or `histogram`."
                    ),
                ),
                io.Float.Input(
                    "regrain_strength",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "`histogram` only: blend back this much of each pixel's original "
                        "local value, to fix banding. No effect on `mkl` or `reinhard`."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="images with reference's colour applied; IMAGE.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        reference,
        method="mkl",
        color_space="Lab",
        strength=1.0,
        luminance_only=False,
        regrain_strength=0.0,
    ) -> io.NodeOutput:
        scale = dynamic.peak(images, reference)
        folded = dynamic.fold(images, scale)
        matched = color_match(
            folded.images[..., :3],
            dynamic.fold(reference, scale).images[..., :3],
            method,
            color_space,
            strength,
            luminance_only,
            regrain_strength,
        )
        matched = dynamic.unfold(matched, folded)
        if images.shape[-1] > 3:
            matched = torch.cat([matched, images[..., 3:]], dim=-1)
        return io.NodeOutput(matched)
