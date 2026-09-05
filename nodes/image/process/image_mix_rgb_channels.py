"""Build one colour image out of three greyscale images."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import broadcast_image_planes, stack_images, tensor2pil


class ImageMixRGBChannels(io.ComfyNode):
    """Assemble a colour image from three separate channel images."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Mix RGB Channels",
            display_name="Image Mix RGB Channels",
            search_aliases=["Image Mix RGB Channels", "merge channels", "combine rgb"],
            category="WAS Suite/Image/Process",
            description=(
                "Combine three greyscale images into one colour image, using their "
                "brightness as the red, green and blue channels."
            ),
            inputs=[
                io.Image.Input(
                    "red_channel",
                    tooltip=(
                        "Image whose brightness becomes the red channel. All three inputs must "
                        "be the same size."
                    ),
                ),
                io.Image.Input(
                    "green_channel",
                    tooltip="Image whose brightness becomes the green channel.",
                ),
                io.Image.Input(
                    "blue_channel",
                    tooltip="Image whose brightness becomes the blue channel.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The recombined colour image. Feeding the same picture to all three "
                        "inputs gives a grey copy of it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, red_channel, green_channel, blue_channel) -> io.NodeOutput:
        folded = dynamic.fold(red_channel)
        red_channel = folded.images
        mixed = [
            cls.mix_rgb_channels(
                tensor2pil(red).convert("L"),
                tensor2pil(green).convert("L"),
                tensor2pil(blue).convert("L"),
            )
            for red, green, blue in broadcast_image_planes(
                red_channel, green_channel, blue_channel
            )
        ]
        return io.NodeOutput(dynamic.unfold(stack_images(mixed), folded))

    @classmethod
    def mix_rgb_channels(cls, red, green, blue):
        """Merge three single-channel images into one ``RGB`` image.

        Args:
            red: Channel image in mode ``L``.
            green: Channel image in mode ``L``.
            blue: Channel image in mode ``L``.

        Returns:
            An ``RGB`` image.

        Raises:
            ValueError: The three images are not all the same size, which is what
                ``Image.merge`` requires.
        """
        from PIL import Image

        return Image.merge("RGB", (red, green, blue))
