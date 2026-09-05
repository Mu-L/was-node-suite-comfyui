"""Colour grading that works on the whole frame: three-way balance and automatic levels.

Images are ``(batch, height, width, channels)`` floats on a 0 to 1 scale. Only the first
three channels are touched.
"""

from __future__ import annotations

__all__ = ["CHANNELS", "METHODS", "TONES", "auto_levels", "balanced", "measured"]

import torch

#: The three tonal ranges a balance acts on, in the order a node lists them.
TONES = ("shadows", "midtones", "highlights")

#: How the black and white points are found, in menu order.
METHODS = ("per channel", "on brightness")

#: The channels a measurement is taken over.
CHANNELS = ("red", "green", "blue")

#: Weights brightness is measured with, matching the rest of the pack.
LUMINANCE = (0.2224884, 0.71690369, 0.06060791)

#: Below this a division is read as a division by zero.
EPSILON = 1e-6

#: Width of each tonal range's weighting curve, as a share of the 0 to 1 scale.
SPREAD = 0.32


def _weights(light: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """How much each of the three ranges owns every pixel, summing to 1."""
    shadows = torch.exp(-(light ** 2) / (2.0 * SPREAD * SPREAD))
    highlights = torch.exp(-((1.0 - light) ** 2) / (2.0 * SPREAD * SPREAD))
    midtones = torch.exp(-((light - 0.5) ** 2) / (2.0 * SPREAD * SPREAD))
    total = (shadows + midtones + highlights).clamp(min=EPSILON)
    return shadows / total, midtones / total, highlights / total


def balanced(
    images: torch.Tensor,
    shadows=(0.0, 0.0, 0.0),
    midtones=(0.0, 0.0, 0.0),
    highlights=(0.0, 0.0, 0.0),
    preserve_luminosity: bool = True,
) -> torch.Tensor:
    """Push colour into the shadows, the midtones and the highlights separately.

    Args:
        images: ``(batch, height, width, channels)`` in 0 to 1.
        shadows: ``(red, green, blue)`` each -1.0 to 1.0, added where the frame is dark.
        midtones: ``(red, green, blue)`` each -1.0 to 1.0, added through the middle.
        highlights: ``(red, green, blue)`` each -1.0 to 1.0, added where the frame is light.
        preserve_luminosity: Whether every pixel is put back to the brightness it had, so
            the grade moves colour without moving exposure.

    Returns:
        A tensor of the shape and dtype it was given.
    """
    if not any(any(row) for row in (shadows, midtones, highlights)):
        return images
    frames = images.to(dtype=torch.float32)
    colours = frames[..., :3].clamp(0.0, 1.0)
    weights = torch.tensor(LUMINANCE, dtype=colours.dtype, device=colours.device)
    before = (colours * weights).sum(dim=-1, keepdim=True)

    low, mid, high = _weights(before)
    shifted = colours.clone()
    for index in range(3):
        shifted[..., index] = colours[..., index] + (
            low[..., 0] * float(shadows[index])
            + mid[..., 0] * float(midtones[index])
            + high[..., 0] * float(highlights[index])
        )
    shifted = shifted.clamp(0.0, 1.0)

    if preserve_luminosity:
        after = (shifted * weights).sum(dim=-1, keepdim=True)
        ratio = torch.where(
            after > EPSILON, before / after.clamp(min=EPSILON), torch.ones_like(after)
        )
        shifted = (shifted * ratio).clamp(0.0, 1.0)

    answer = frames.clone()
    answer[..., :3] = shifted
    return answer.to(dtype=images.dtype)


def measured(plane: torch.Tensor, clip_low: float, clip_high: float) -> tuple[float, float]:
    """The levels a share of the darkest and lightest pixels sits inside.

    Args:
        plane: Any shape of values on a 0 to 1 scale.
        clip_low: Share of the darkest pixels allowed to go to black, 0.0 to 0.2.
        clip_high: Share of the lightest pixels allowed to go to white, 0.0 to 0.2.

    Returns:
        ``(black, white)`` on a 0 to 1 scale, with ``white`` above ``black``.
    """
    flat = plane.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == 0:
        return 0.0, 1.0
    quantiles = torch.tensor(
        [min(max(clip_low, 0.0), 0.2), 1.0 - min(max(clip_high, 0.0), 0.2)],
        dtype=torch.float32,
        device=flat.device,
    )
    # A quantile over more than 16 million values raises, so a long frame is sampled.
    if flat.numel() > 1 << 24:
        flat = flat[:: (flat.numel() // (1 << 24)) + 1]
    low, high = torch.quantile(flat, quantiles).tolist()
    if high - low < EPSILON:
        return float(low), float(low) + 1.0
    return float(low), float(high)


def auto_levels(
    images: torch.Tensor,
    method: str = METHODS[0],
    clip_low: float = 0.001,
    clip_high: float = 0.001,
    strength: float = 1.0,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Stretch a frame so its darkest pixels reach black and its lightest reach white.

    Args:
        images: ``(batch, height, width, channels)`` in 0 to 1.
        method: One of :data:`METHODS`. ``per channel`` also neutralises a colour cast.
        clip_low: Share of the darkest pixels allowed to go to black, 0.0 to 0.2.
        clip_high: Share of the lightest pixels allowed to go to white, 0.0 to 0.2.
        strength: How far towards the stretched result the frame moves, 0.0 to 1.0.

    Returns:
        ``(images, (black, white))``. The two levels are the ones found on brightness,
        whichever method ran.
    """
    frames = images.to(dtype=torch.float32)
    colours = frames[..., :3].clamp(0.0, 1.0)
    weights = torch.tensor(LUMINANCE, dtype=colours.dtype, device=colours.device)
    light = (colours * weights).sum(dim=-1)
    found = measured(light, clip_low, clip_high)

    stretched = colours.clone()
    if method == METHODS[0]:
        for index in range(3):
            black, white = measured(colours[..., index], clip_low, clip_high)
            stretched[..., index] = (colours[..., index] - black) / max(white - black, EPSILON)
    else:
        black, white = found
        stretched = (colours - black) / max(white - black, EPSILON)
    stretched = stretched.clamp(0.0, 1.0)

    mixed = torch.lerp(colours, stretched, min(max(float(strength), 0.0), 1.0))
    answer = frames.clone()
    answer[..., :3] = mixed
    return answer.to(dtype=images.dtype), found
