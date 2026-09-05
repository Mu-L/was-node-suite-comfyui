"""Rotate every colour around the hue wheel."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic

logger = log.get_logger("nodes.image.adjust")

#: :mod:`colorsys`' own ramp boundaries, which :func:`_channel` is transcribed against.
ONE_THIRD = 1.0 / 3.0
ONE_SIXTH = 1.0 / 6.0
TWO_THIRD = 2.0 / 3.0


def _channel(low: torch.Tensor, high: torch.Tensor, hue: torch.Tensor) -> torch.Tensor:
    """One RGB channel of an HLS triple, as ``colorsys._v`` computes it.

    Args:
        low: Bottom of the channel ramp, the minimum of the original triple.
        high: Top of the channel ramp, the maximum of the original triple.
        hue: Position on the wheel to read the channel at. Wrapped into ``[0, 1)`` first.

    Returns:
        Channel values in ``[0, 1]``, shaped like the broadcast arguments.
    """
    hue = torch.remainder(hue, 1.0)
    rising = low + (high - low) * hue * 6.0
    falling = low + (high - low) * (TWO_THIRD - hue) * 6.0
    return torch.where(
        hue < ONE_SIXTH,
        rising,
        torch.where(hue < 0.5, high, torch.where(hue < TWO_THIRD, falling, low)),
    )


def hue_rotation(image: torch.Tensor, hue_shift: float = 0.0) -> torch.Tensor:
    """Shift the hue of every pixel of an image batch by a fraction of the colour wheel.

    Args:
        image: Image tensor whose last axis is three colour channels, holding values in
            ``[0, 1]``. Values outside that range are clipped rather than scaled.
        hue_shift: Fraction of the wheel to advance by, 0.0 to 1.0.

    Returns:
        A float32 tensor shaped like ``image``, quantised to 8 bits, on ``image``'s own
        device. The arithmetic runs on the CPU in float64 and the result is moved back.

    Raises:
        ValueError: The last axis is not three channels, so there is no RGB triple to
            rotate.
    """
    if image.shape[-1] != 3:
        raise ValueError(
            f"Image Rotate Hue needs three colour channels and this image has "
            f"{image.shape[-1]}."
        )

    # Transcribed operation for operation, so the whole batch is one set of tensor
    # operations rather than a Python loop over pixels. float64 is absent on MPS and runs
    # at a fraction of the float32 rate on consumer CUDA cards, so the work is done on the
    # CPU and the result moved back to the device the batch arrived on.
    quantised = torch.clamp(image.cpu() * 255.0, 0, 255).to(torch.uint8)
    scaled = quantised.to(torch.float64) / 255
    red, green, blue = scaled[..., 0], scaled[..., 1], scaled[..., 2]

    high = torch.maximum(torch.maximum(red, green), blue)
    low = torch.minimum(torch.minimum(red, green), blue)
    total = high + low
    spread = high - low
    lightness = total / 2.0

    saturation = torch.where(lightness <= 0.5, spread / total, spread / ((2.0 - high) - low))
    red_gap = (high - red) / spread
    green_gap = (high - green) / spread
    blue_gap = (high - blue) / spread
    hue = torch.where(
        red == high,
        blue_gap - green_gap,
        torch.where(green == high, (2.0 + red_gap) - blue_gap, (4.0 + green_gap) - red_gap),
    )
    hue = torch.remainder(hue / 6.0, 1.0)

    # A pixel whose three channels are equal has no hue and no saturation, and divides by
    # a zero spread on the way to both.
    flat = low == high
    hue = torch.where(flat, torch.zeros_like(hue), hue)
    saturation = torch.where(flat, torch.zeros_like(saturation), saturation)

    hue = torch.remainder(hue + hue_shift, 1.0)

    ceiling = torch.where(
        lightness <= 0.5,
        lightness * (1.0 + saturation),
        (lightness + saturation) - (lightness * saturation),
    )
    floor = 2.0 * lightness - ceiling
    rotated = torch.stack(
        (
            _channel(floor, ceiling, hue + ONE_THIRD),
            _channel(floor, ceiling, hue),
            _channel(floor, ceiling, hue - ONE_THIRD),
        ),
        dim=-1,
    )
    grey = (saturation == 0.0).unsqueeze(-1)
    rotated = torch.where(grey, lightness.unsqueeze(-1).expand_as(rotated), rotated)

    return ((rotated * 255).trunc().to(torch.uint8).to(torch.float32) / 255.0).to(image.device)


class ImageRotateHue(io.ComfyNode):
    """Advance every colour in an image around the hue wheel by the same amount."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Rotate Hue",
            display_name="Image Rotate Hue",
            search_aliases=["Image Rotate Hue", "hue shift", "recolor", "colour wheel"],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Shift every colour around the hue wheel by the same amount, keeping "
                "brightness and saturation as they were. Red becomes green, green becomes "
                "blue, and so on."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to recolour. A whole batch is rotated at once, and each "
                        "image comes back the same as if it had been sent on its own."
                    ),
                ),
                io.Float.Input(
                    "hue_shift",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How far around the colour wheel to turn, as a fraction of a full "
                        "turn. 0.0 and 1.0 both leave the colours where they are, 0.333 moves "
                        "red to green, 0.5 sends every colour to its opposite, 0.667 moves red "
                        "to blue."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The recoloured image, the same size as the source."),
            ],
        )

    @classmethod
    def execute(cls, image, hue_shift) -> io.NodeOutput:
        if hue_shift > 1.0 or hue_shift < 0.0:
            logger.error(
                "The hue_shift `%s` is out of range. Valid range is 0.0 - 1.0, so no hue "
                "rotation was applied.",
                hue_shift,
            )
            hue_shift = 0.0
        folded = dynamic.fold(image)
        return io.NodeOutput(
            dynamic.unfold(hue_rotation(folded.images, hue_shift), folded)
        )
