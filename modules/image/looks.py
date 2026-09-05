"""The grade primitives a photo look is built from, as torch operations.

Planes are ``(..., 3)`` floats. A result is held inside the range its source occupied, so a
frame carrying light above white keeps it.
"""

from __future__ import annotations

__all__ = [
    "bloom",
    "brightness",
    "contrast",
    "exposure",
    "fade",
    "grain",
    "greyscale",
    "hue_rotate",
    "saturation",
    "sepia",
    "split_tone",
    "temperature",
    "tint_shadows",
]

import torch

from .blend_modes import LUMINANCE, blend, ceiling_of

#: Widest gaussian a bloom is blurred by, in fractions of the frame's shorter side.
BLOOM_REACH = 0.25

#: Weights the filter specification measures brightness with, for a saturation change.
FILTER_LUMA = (0.213, 0.715, 0.072)

#: Weights it measures brightness with when colour is drained, which are Rec. 709's.
GREY_LUMA = (0.2126, 0.7152, 0.0722)

#: Channel gains a warmth of 1.0 applies, in red, green, blue order.
WARM = (1.06, 1.0, 0.94)

#: Furthest a split tone leans a channel, as a fraction of the range.
SPLIT_REACH = 0.14

#: Furthest a shadow tint leans a channel, as a fraction of the range.
TINT_REACH = 0.30


def _luma(
    plane: torch.Tensor, weights: tuple[float, float, float] = LUMINANCE
) -> torch.Tensor:
    """One brightness plane out of an RGB picture, keeping its trailing axis.

    Args:
        plane: ``(..., 3)`` colours.
        weights: What each of the three channels counts for.

    Returns:
        A ``(..., 1)`` tensor.
    """
    measure = torch.tensor(weights, dtype=plane.dtype, device=plane.device)
    return (plane * measure).sum(dim=-1, keepdim=True)


def _held(result: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Hold a result inside the range its source occupied.

    Args:
        result: The plane a step produced.
        source: The plane that went into it.

    Returns:
        The result, clamped to 0 and the source's ceiling.
    """
    return result.clamp(0.0, ceiling_of(source))


def exposure(plane: torch.Tensor, stops: float) -> torch.Tensor:
    """Multiply a plane by a number of photographic stops.

    Args:
        plane: ``(..., 3)`` colours.
        stops: Doublings of light. 1.0 is twice as bright, -1.0 is half.

    Returns:
        The brightened plane.
    """
    return plane * (2.0 ** stops)


def brightness(plane: torch.Tensor, amount: float) -> torch.Tensor:
    """Scale every channel of a plane by one number.

    Args:
        plane: ``(..., 3)`` colours.
        amount: 1.0 leaves it alone, 1.2 brightens, 0.9 darkens.

    Returns:
        The plane, at the new brightness.
    """
    return _held(plane * amount, plane)


def contrast(plane: torch.Tensor, amount: float, pivot: float = 0.5) -> torch.Tensor:
    """Push a plane away from a pivot.

    Args:
        plane: ``(..., 3)`` colours.
        amount: 1.0 leaves it alone, 1.3 firms it up, 0.7 flattens it.
        pivot: The value that does not move.

    Returns:
        The plane, at the new contrast.
    """
    return _held((plane - pivot) * amount + pivot, plane)


def saturation(plane: torch.Tensor, amount: float) -> torch.Tensor:
    """Move a plane towards or away from its own greyscale.

    Args:
        plane: ``(..., 3)`` colours.
        amount: 1.0 leaves it alone, 0.0 is grey, 1.4 is vivid.

    Returns:
        The plane, at the new saturation.
    """
    grey = _luma(plane, FILTER_LUMA)
    return _held(grey + (plane - grey) * amount, plane)


def temperature(plane: torch.Tensor, warmth: float) -> torch.Tensor:
    """Warm or cool a plane by gaining its red and blue apart.

    Args:
        plane: ``(..., 3)`` colours.
        warmth: 1.0 is a full warm step, -1.0 the same towards blue, 0.0 nothing.

    Returns:
        The plane, warmed or cooled.
    """
    gains = torch.tensor(
        [1.0 + (WARM[0] - 1.0) * warmth, 1.0, 1.0 + (WARM[2] - 1.0) * warmth],
        dtype=plane.dtype,
        device=plane.device,
    )
    return _held(plane * gains, plane)


def fade(plane: torch.Tensor, amount: float, ceiling: float | None = None) -> torch.Tensor:
    """Lift the black point, which is what makes a look matte.

    Args:
        plane: ``(..., 3)`` colours.
        amount: How far black rises. 0.08 is a gentle matte, 0.2 is heavy.
        ceiling: The top of the range, or None to read it off the plane.

    Returns:
        The plane, with its blacks lifted and its range squeezed to suit.
    """
    top = ceiling_of(plane) if ceiling is None else ceiling
    return plane * (top - amount) / top + amount


def greyscale(plane: torch.Tensor, amount: float = 1.0) -> torch.Tensor:
    """Drain a plane towards its brightness.

    Args:
        plane: ``(..., 3)`` colours.
        amount: 1.0 is fully grey, 0.5 half way. More than 1.0 reads as 1.0.

    Returns:
        The drained plane.
    """
    grey = _luma(plane, GREY_LUMA)
    if amount >= 1.0:
        return _held(grey.expand_as(plane), plane)
    return _held(plane + (grey - plane) * amount, plane)


def sepia(plane: torch.Tensor, amount: float = 1.0) -> torch.Tensor:
    """Tone a plane through the sepia matrix.

    Args:
        plane: ``(..., 3)`` colours.
        amount: 1.0 is full sepia, 0.0 leaves the plane alone. More than 1.0 reads as 1.0.

    Returns:
        The toned plane.
    """
    matrix = torch.tensor(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=plane.dtype,
        device=plane.device,
    )
    toned = plane @ matrix.T
    return _held(plane + (toned - plane) * min(amount, 1.0), plane)


def hue_rotate(plane: torch.Tensor, degrees: float) -> torch.Tensor:
    """Turn every hue around the colour wheel.

    Args:
        plane: ``(..., 3)`` colours.
        degrees: How far to turn. 360.0 is a full turn back to the start.

    Returns:
        The rotated plane.
    """
    angle = torch.tensor(degrees * torch.pi / 180.0, dtype=plane.dtype, device=plane.device)
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, two = 0.213, 0.715
    three = 0.072
    matrix = torch.tensor(
        [
            [one + cos * (1 - one) - sin * one,
             two - cos * two - sin * two,
             three - cos * three + sin * (1 - three)],
            [one - cos * one + sin * 0.143,
             two + cos * (1 - two) + sin * 0.140,
             three - cos * three - sin * 0.283],
            [one - cos * one - sin * (1 - one),
             two - cos * two + sin * two,
             three + cos * (1 - three) + sin * three],
        ],
        dtype=plane.dtype,
        device=plane.device,
    )
    return _held(plane @ matrix.T, plane)


def split_tone(
    plane: torch.Tensor,
    shadows: tuple[float, float, float],
    highlights: tuple[float, float, float],
    strength: float = 1.0,
) -> torch.Tensor:
    """Tint the dark end and the bright end towards two colours.

    Args:
        plane: ``(..., 3)`` colours.
        shadows: The colour the dark end leans towards, in 0 to 1.
        highlights: The colour the bright end leans towards.
        strength: How far it leans. 0.0 leaves the plane alone.

    Returns:
        The split-toned plane.
    """
    top = ceiling_of(plane)
    weight = (_luma(plane) / top).clamp(0.0, 1.0)
    dark = torch.tensor(shadows, dtype=plane.dtype, device=plane.device) - 0.5
    light = torch.tensor(highlights, dtype=plane.dtype, device=plane.device) - 0.5
    # A colour is a lean away from neutral, added to what is there, so the picture keeps its
    # own colour and takes a cast rather than being replaced by one.
    offset = dark * (1.0 - weight) + light * weight
    return _held(plane + offset * strength * SPLIT_REACH * top, plane)


def tint_shadows(plane: torch.Tensor, colour: tuple[float, float, float], amount: float) -> torch.Tensor:
    """Wash a colour into the dark end alone.

    Args:
        plane: ``(..., 3)`` colours.
        colour: The colour to wash in, in 0 to 1.
        amount: How much of it. 0.1 is a hint.

    Returns:
        The plane, with its shadows tinted.
    """
    top = ceiling_of(plane)
    # Squared, so the wash gathers in the darkest end rather than across the midtones.
    weight = (1.0 - (_luma(plane) / top).clamp(0.0, 1.0)) ** 2
    lean = (torch.tensor(colour, dtype=plane.dtype, device=plane.device) - 0.5) * 2.0
    return _held(plane + lean * weight * amount * TINT_REACH * top, plane)


def bloom(
    plane: torch.Tensor,
    threshold: float = 0.7,
    radius: float = 0.04,
    intensity: float = 0.5,
    colour: tuple[float, float, float] | None = None,
    mode: str = "screen",
) -> torch.Tensor:
    """Spread the highlights back over the picture as a glow.

    Args:
        plane: ``(..., 3)`` colours.
        threshold: Where the highlights start, as a fraction of the range. 0.7 catches a
            bright sky, 0.9 only a specular.
        radius: How far the glow reaches, as a fraction of the shorter side. 0.04 is a
            halo, 0.15 a wash.
        intensity: How strongly it comes back. 0.0 is off, 1.0 is heavy.
        colour: The colour the glow is tinted, in 0 to 1, or None to keep the highlights'
            own colour.
        mode: How the glow returns, one of the blend mode names. ``screen`` and ``lighten``
            are the two that read as light.

    Returns:
        The plane with its highlights spread.
    """
    if intensity <= 0.0:
        return plane

    from .sharpen import _blur

    top = ceiling_of(plane)
    # A soft knee, so a highlight enters the glow gradually rather than at one value.
    weight = ((_luma(plane) / top - threshold) / max(1.0 - threshold, 1e-4)).clamp(0.0, 1.0)
    highlights = plane * (weight * weight)

    tint = None
    if colour is not None:
        tint = torch.tensor(colour, dtype=plane.dtype, device=plane.device)
        # One plane to blur rather than three, since the tint goes on afterwards.
        highlights = _luma(highlights)

    height, width = plane.shape[-3], plane.shape[-2]
    reach = max(1.0, min(BLOOM_REACH, radius) * min(height, width))

    # _blur works on (batch, channels, height, width).
    channels = highlights.shape[-1]
    folded = highlights.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
    # Two passes of half the reach, which together fall away to nothing at the full reach.
    folded = _blur(_blur(folded, reach / 2.0, "gaussian"), reach / 2.0, "gaussian")
    glow = folded.permute(0, 2, 3, 1).reshape(plane.shape[:-1] + (channels,))
    if tint is not None:
        glow = glow * tint

    return _held(blend(plane, glow * intensity, mode), plane)


def grain(plane: torch.Tensor, amount: float, seed: int = 0) -> torch.Tensor:
    """Lay film grain over a plane.

    Args:
        plane: ``(..., 3)`` colours.
        amount: How coarse it is. 0.02 is fine, 0.08 heavy.
        seed: Which grain. The same seed lays the same grain twice.

    Returns:
        The grained plane.
    """
    if amount <= 0.0:
        return plane
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    speckle = torch.randn(
        plane.shape[:-1] + (1,), generator=generator, dtype=torch.float32
    ).to(plane.device, plane.dtype)
    top = ceiling_of(plane)
    # Grain is strongest in the midtones and fades out at both ends, as film does.
    weight = 1.0 - (2.0 * (_luma(plane) / top).clamp(0.0, 1.0) - 1.0).abs()
    return _held(plane + speckle * amount * weight * top, plane)
