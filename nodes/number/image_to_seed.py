"""Derive a reproducible seed from an image's pixels."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import tensor2pil
from ...modules.util.hashing import image2seed


class ImageToSeed(io.ComfyNode):
    """Turn each image in a batch into a seed number."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image to Seed",
            display_name="Image to Seed",
            search_aliases=["Image to Seed", "image hash", "seed from image"],
            category="WAS Suite/Number/Operations",
            description=(
                "Turn images into seed numbers, so a picture can stand in for a seed. The "
                "same picture always gives the same number."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to digest. Every image in the batch produces its own seed."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    is_output_list=True,
                    tooltip=(
                        "One seed per image, between 0 and 4294967295. Because this is a list, "
                        "a node reading it runs once per seed."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        return io.NodeOutput([image2seed(tensor2pil(image)) for image in images])
