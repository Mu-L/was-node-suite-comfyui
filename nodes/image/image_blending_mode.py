"""Combine two images with one of the blend modes a layer stack names."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.image import blend_modes, dynamic
from ...modules.convert.tensors import broadcast_image_planes

#: Widget value -> the blend mode behind it. Every name spells its mode with underscores,
#: which is how this node has always spelled them, and ``add`` sums the two layers.
MODES = {
    "add": "linear-dodge",
    **{name.replace("-", "_"): name for name in blend_modes.MODES},
}


class ImageBlendingMode(io.ComfyNode):
    """Blend two images with one of the layer blend modes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Blending Mode",
            display_name="Image Blending Mode",
            search_aliases=[
                "Image Blending Mode",
                "blend mode",
                "multiply",
                "screen",
                "overlay",
            ],
            category="WAS Suite/Image",
            description=(
                "Combine two images with any of the 26 blending modes a layer stack names, "
                "then fade the result back towards image_a. The output is the size of "
                "image_a, and image_b is resampled to match it. `normal` lays image_b over "
                "image_a and `add` sums the two, `multiply` darkens, `screen` lightens, "
                "`overlay`, `soft_light`, `hard_light`, `vivid_light` and `linear_light` "
                "raise contrast, `darken` and `lighten` keep the darker or lighter pixel, "
                "`difference`, `exclusion`, `subtract` and `grain_extract` show where the "
                "two disagree, and `hue`, `saturation`, `color` and `luminosity` take one "
                "part of image_b and keep the rest of image_a."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "The base layer, and the one that sets the output size. Also what "
                        "the result fades back to as blend_percentage drops."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "The layer blended on top of image_a. A batch here is paired with "
                        "the image_a batch frame by frame, a single image is blended into "
                        "every frame, and the result is as long as the longer of the two."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=[
                        "add",
                        "color",
                        "color_burn",
                        "color_dodge",
                        "darken",
                        "difference",
                        "exclusion",
                        "hard_light",
                        "hue",
                        "lighten",
                        "multiply",
                        "overlay",
                        "screen",
                        "soft_light",
                        "normal",
                        "linear_burn",
                        "vivid_light",
                        "pin_light",
                        "linear_light",
                        "hard_mix",
                        "subtract",
                        "divide",
                        "grain_extract",
                        "grain_merge",
                        "saturation",
                        "luminosity",
                    ],
                    tooltip=(
                        "How the two layers are combined. `normal` lays image_b over "
                        "image_a; `add` sums them; `multiply` darkens; `screen` lightens; "
                        "`overlay` raises contrast; `hue` takes colour from image_b while "
                        "keeping the shading of image_a."
                    ),
                ),
                io.Float.Input(
                    "blend_percentage",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the blended result is applied. 1.0 uses it as it is, "
                        "0.0 returns image_a untouched, 0.5 is an even mix of the two."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The two layers combined with the chosen mode.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, mode, blend_percentage) -> io.NodeOutput:
        scale = dynamic.peak(image_a, image_b)
        first = dynamic.fold(image_a, scale)
        frames = broadcast_image_planes(first.images, dynamic.fold(image_b, scale).images)
        blended = [cls.composite(a, b, mode, blend_percentage) for a, b in frames]
        return io.NodeOutput(dynamic.unfold(torch.stack(blended, dim=0), first))

    @staticmethod
    def composite(image_a, image_b, mode: str, blend_percentage: float):
        """Combine one image with another under a blend mode.

        Args:
            image_a: Base layer, as ``(height, width, channels)`` in ``[0, 1]``. Sets the
                output size and is what the result fades back to.
            image_b: Layer blended on top, resampled to the base's size.
            mode: A key of :data:`MODES`. Any other name is read as ``normal``.
            blend_percentage: How strongly the blended result is applied, 0.0 to 1.0.

        Returns:
            A ``(height, width, 3)`` tensor.
        """
        base = _colours(image_a)
        top = _colours(image_b)
        if top.shape[:2] != base.shape[:2]:
            top = torch.nn.functional.interpolate(
                top.permute(2, 0, 1).unsqueeze(0),
                size=(int(base.shape[0]), int(base.shape[1])),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
        mixed = blend_modes.blend(base, top, MODES.get(mode, ""))
        return (base + (mixed - base) * float(blend_percentage)).clamp(0.0, 1.0)


def _colours(plane):
    """One image plane as three colour channels."""
    if plane.ndim == 2:
        plane = plane.unsqueeze(-1)
    plane = plane.to(dtype=torch.float32)
    if int(plane.shape[2]) >= 3:
        return plane[:, :, :3]
    return plane[:, :, :1].repeat(1, 1, 3)
