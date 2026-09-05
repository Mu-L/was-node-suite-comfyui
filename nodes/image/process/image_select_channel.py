"""Isolate one colour channel as a greyscale image."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import dynamic

#: Position of each named channel in an ``IMAGE`` tensor's last axis.
CHANNELS = {"red": 0, "green": 1, "blue": 2}


class ImageSelectChannel(io.ComfyNode):
    """Keep one of the three colour channels and show it as grey."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Select Channel",
            display_name="Image Select Channel",
            search_aliases=["Image Select Channel", "extract channel", "split rgb"],
            category="WAS Suite/Image/Process",
            description=(
                "Extract one colour channel and return it as a greyscale image, where "
                "white means that channel was at full strength."
            ),
            inputs=[
                io.Image.Input("image", tooltip="The image to read the channel from."),
                io.Combo.Input(
                    "channel",
                    options=["red", "green", "blue"],
                    tooltip=(
                        "Which channel to keep. Skin and warm light are brightest in `red`, "
                        "foliage and most detail in `green`, sky and shadows in `blue`. The "
                        "other two channels are discarded."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The chosen channel as a grey image the same size as the input, "
                        "suitable for use as a mask or a depth-like map."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, channel="red") -> io.NodeOutput:
        folded = dynamic.fold(image)
        return io.NodeOutput(dynamic.unfold(
            cls.convert_to_single_channel(folded.images, channel), folded
        ))

    @classmethod
    def convert_to_single_channel(cls, image, channel="red"):
        """Copy one channel of every image in a batch into all three.

        Args:
            image: Image tensor shaped ``(batch, height, width, channels)`` scaled to
                ``[0, 1]``. A single-channel image gives the same grey whichever channel
                is named, since all three of its colour channels hold that one value.
            channel: ``"red"``, ``"green"`` or ``"blue"``.

        Returns:
            An image tensor the same size and batch length, with three channels all
            holding the chosen one. It stays on the device the input arrived on.

        Raises:
            ValueError: ``channel`` names no channel.
        """
        if channel not in CHANNELS:
            raise ValueError("Invalid channel option. Please choose 'red', 'green', or 'blue'.")

        quantised = torch.clamp(image * 255.0, 0, 255).to(torch.uint8)
        index = min(CHANNELS[channel], quantised.shape[-1] - 1)
        plane = quantised[..., index]
        return torch.stack((plane, plane, plane), dim=-1).to(torch.float32) / 255.0
