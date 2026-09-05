"""Keep only the pixels near one colour and black out the rest."""

from __future__ import annotations

import torch
from comfy_api.latest import io


class ImageSelectColor(io.ComfyNode):
    """Isolate one colour, blacking out everything else."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Select Color",
            display_name="Image Select Color",
            search_aliases=["Image Select Color", "color key", "isolate colour"],
            category="WAS Suite/Image/Process",
            description=(
                "Keep only the pixels close to one colour and turn the rest black, for "
                "isolating a green screen, a sky or a single painted object."
            ),
            inputs=[
                io.Image.Input("image", tooltip="The image to search for the colour in."),
                io.Int.Input(
                    "red",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the colour to look for, 0 to 255.",
                ),
                io.Int.Input(
                    "green",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the colour to look for, 0 to 255.",
                ),
                io.Int.Input(
                    "blue",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the colour to look for, 0 to 255. Together the three "
                        "make one colour: 255/255/255 is white, 0/255/0 is pure green."
                    ),
                ),
                io.Int.Input(
                    "variance",
                    default=10,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "How far each channel may differ from the target and still count as a "
                        "match. 0 keeps only the exact colour, 10 tolerates slight shading, and "
                        "255 matches every pixel in the image."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The matching pixels at their original colour, with every other pixel "
                        "black."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, red=255, green=255, blue=255, variance=10) -> io.NodeOutput:
        return io.NodeOutput(cls.color_pick(image, red, green, blue, variance))

    @classmethod
    def color_pick(cls, image, red=255, green=255, blue=255, variance=10):
        """Black out every pixel of a batch that falls outside a cube around one colour.

        Args:
            image: Image tensor shaped ``(batch, height, width, channels)`` scaled to
                ``[0, 1]``. Only the first three channels are read; an alpha channel is
                dropped, as converting to ``RGB`` did.
            red: Red level of the target, 0-255.
            green: Green level of the target, 0-255.
            blue: Blue level of the target, 0-255.
            variance: Half-width of the accepted band on each channel. A band reaching
                past 0 or 255 accepts everything on that side.

        Returns:
            A three-channel image tensor holding the matched pixels and black elsewhere,
            on the device the input arrived on.
        """
        quantised = torch.clamp(image * 255.0, 0, 255).to(torch.uint8)
        if quantised.shape[-1] == 1:
            quantised = quantised.expand(*quantised.shape[:-1], 3)
        pixels = quantised[..., :3]

        target = torch.tensor(
            [float(red), float(green), float(blue)], dtype=torch.float32, device=image.device
        )
        values = pixels.to(torch.float32)
        within = (values >= target - variance) & (values <= target + variance)
        selected = torch.where(within.all(dim=-1, keepdim=True), pixels, pixels.new_zeros(()))
        return selected.to(torch.float32) / 255.0
