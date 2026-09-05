"""Rotate a batch of images in quarter turns."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.interface import size_report


class ImageRotate(io.ComfyNode):
    """Rotate every image in a batch by a multiple of 90 degrees."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Rotate",
            display_name="Image Rotate (Advanced)",
            search_aliases=["Image Rotate", "turn", "quarter turn", "orientation"],
            category="WAS Suite/Image/Transform",
            description=(
                "Turn every image in the batch counter-clockwise by a multiple of 90 "
                "degrees. Anything between multiples is rounded down, so 100 degrees "
                "rotates by 90."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to turn. Every image in a batch gets the same rotation and "
                        "comes back in the order it arrived, with its width and height swapped "
                        "at 90 or 270 degrees in `transpose` mode."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["transpose", "internal"],
                    tooltip=(
                        "`transpose` turns the image by moving whole pixels, so a portrait "
                        "image becomes landscape and nothing is lost. `internal` rotates "
                        "inside the original frame instead, which keeps the width and "
                        "height as they were and crops the corners off a quarter turn of a "
                        "non-square image."
                    ),
                ),
                io.Int.Input(
                    "rotation",
                    default=0,
                    min=0,
                    max=360,
                    step=90,
                    tooltip=(
                        "How far to turn, in degrees counter-clockwise. Only multiples of "
                        "90 are applied: 90, 180 and 270 turn, while 0 and 360 leave the "
                        "image alone."
                    ),
                ),
                io.Combo.Input(
                    "sampler",
                    options=["nearest", "bilinear", "bicubic"],
                    tooltip=(
                        "Filter used when pixels have to be interpolated. Quarter turns "
                        "land on whole pixels, so this changes nothing at any rotation the "
                        "node accepts; `nearest` is the cheapest of the three."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The rotated images, in the order they arrived.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode, rotation, sampler) -> io.NodeOutput:
        folded = dynamic.fold(images)
        images = folded.images
        from PIL import Image

        resample = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }.get(sampler, Image.Resampling.BILINEAR)

        # A saved workflow can hold any value in range, not only a multiple of the
        # widget's step, and both rotations here are whole quarter turns.
        asked = rotation
        if rotation % 90 != 0:
            rotation = int((rotation // 90) * 90)

        batch_tensor = []
        for image in images:
            image = tensor2pil(image)

            if mode == "internal":
                image = image.rotate(rotation, resample)
            else:
                for _ in range(int(rotation / 90)):
                    image = image.transpose(Image.Transpose.ROTATE_90)

            batch_tensor.append(pil2tensor(image))

        turned = torch.cat(batch_tensor, dim=0)
        size_report.publish(
            images,
            turned,
            action="rotated",
            facts=(
                {"turn": f"{rotation} degrees, {asked} asked"} if asked != rotation
                else None
            ),
        )
        return io.NodeOutput(dynamic.unfold(turned, folded))
