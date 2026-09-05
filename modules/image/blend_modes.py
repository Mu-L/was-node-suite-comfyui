"""The blend modes a layer stack names, as torch operations on colour planes.

Planes are ``(..., 3)`` floats on a 0 to 1 scale. :func:`blend` mixes two of them and
composites nothing.
"""

from __future__ import annotations

__all__ = ["EPSILON", "LUMINANCE", "MODES", "blend", "ceiling_of", "known"]

import torch

#: Every blend mode, in the order the compositor lists them.
MODES = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "linear-dodge",
    "linear-burn",
    "vivid-light",
    "pin-light",
    "linear-light",
    "hard-mix",
    "subtract",
    "divide",
    "grain-extract",
    "grain-merge",
    "hue",
    "saturation",
    "color",
    "luminosity",
)

#: Below this a division is read as a division by zero.
EPSILON = 1e-6

#: Weights a colour's brightness is measured with, as the compositing specification sets
#: them. Tone mapping and grading weigh linear light differently and carry their own.
LUMINANCE = (0.3, 0.59, 0.11)



#: How far above white a value may sit and still count as a rounding artefact.
HEADROOM = 1e-4


def ceiling_of(*planes) -> float:
    """The largest value the planes going in hold, or 1.0 where none passes white.

    Args:
        *planes: ``(..., 3)`` tensors, any of which may be None.

    Returns:
        1.0 for a picture already inside 0 to 1, otherwise the highest value present.
    """
    high = 1.0
    for plane in planes:
        if plane is None:
            continue
        seen = float(plane.detach().amax())
        if seen > high:
            high = seen
    return high if high > 1.0 + HEADROOM else 1.0


def _ceiling(i, l) -> float:
    """The range two planes are mixed inside."""
    return ceiling_of(i, l)


def _divided(numerator, denominator):
    """A division answering zero where the divisor is too near zero to use."""
    usable = denominator.abs() >= EPSILON
    safe = torch.where(usable, denominator, torch.ones_like(denominator))
    return torch.where(usable, numerator / safe, torch.zeros_like(denominator))


def _luminance(rgb):
    """One brightness plane out of an RGB picture."""
    return rgb[..., 0] * LUMINANCE[0] + rgb[..., 1] * LUMINANCE[1] + rgb[..., 2] * LUMINANCE[2]


#: Blend mode -> how one channel of the layer mixes with the channel under it.
_CHANNEL = {
    "normal": lambda i, l: l,
    "multiply": lambda i, l: i * l,
    "screen": lambda i, l: 1.0 - (1.0 - i) * (1.0 - l),
    "overlay": lambda i, l: torch.where(
        i < 0.5, 2.0 * i * l, 1.0 - 2.0 * (1.0 - l) * (1.0 - i)
    ),
    "darken": lambda i, l: torch.minimum(i, l),
    "lighten": lambda i, l: torch.maximum(i, l),
    "soft-light": lambda i, l: _soft_light(i, l),
    "difference": lambda i, l: (i - l).abs(),
    "exclusion": lambda i, l: 0.5 - 2.0 * (i - 0.5) * (l - 0.5),
    "linear-dodge": lambda i, l: i + l,
    "linear-burn": lambda i, l: i + l - 1.0,
    "pin-light": lambda i, l: torch.where(
        l > 0.5, torch.maximum(i, 2.0 * (l - 0.5)), torch.minimum(i, 2.0 * l)
    ),
    "linear-light": lambda i, l: i + 2.0 * l - 1.0,
    "hard-mix": lambda i, l: torch.where(i + l < 1.0, torch.zeros_like(i), torch.ones_like(i)),
    "subtract": lambda i, l: (i - l).clamp(min=0.0),
    "grain-extract": lambda i, l: i - l + 0.5,
    "grain-merge": lambda i, l: i + l - 0.5,
}


#: Blend mode -> a per-channel mix that has to know where the range ends. A picture already
#: inside 0 to 1 gets a ceiling of 1.0 and the familiar result; one carrying light above white
#: keeps it.
_CEILED = {
    # The order of the two tests is the specification's: a black backdrop stays black even
    # under a white layer.
    "color-dodge": lambda i, l, c: torch.where(
        i <= EPSILON,
        torch.zeros_like(i),
        torch.where(
            c - l <= EPSILON, torch.full_like(i, c), _divided(i, 1.0 - l / c).clamp(max=c)
        ),
    ),
    # A backdrop already at the top of the range stays there, even under a black layer.
    "color-burn": lambda i, l, c: torch.where(
        c - i <= EPSILON,
        torch.full_like(i, c),
        torch.where(
            l <= EPSILON, torch.zeros_like(i), c - _divided(c - i, l / c).clamp(max=c)
        ),
    ),
    "hard-light": lambda i, l, c: torch.where(
        l > 0.5 * c,
        (c - (c - i) * (c - (l - 0.5 * c) * 2.0) / c).clamp(max=c),
        (i * (l * 2.0) / c).clamp(max=c),
    ),
    "vivid-light": lambda i, l, c: torch.where(
        l <= 0.5 * c,
        torch.where(
            2.0 * l <= EPSILON,
            torch.zeros_like(i),
            (c - _divided(c - i, 2.0 * l / c)).clamp(min=0.0),
        ),
        torch.where(
            2.0 * (c - l) <= EPSILON,
            torch.full_like(i, c),
            _divided(i, 2.0 * (c - l) / c).clamp(max=c),
        ),
    ),
    "divide": lambda i, l, c: (i / l.clamp(min=EPSILON)).clamp(0.0, c),
}


def _soft_light(i, l):
    """The soft-light curve, as the compositing specification writes it."""
    dark = i - (1.0 - 2.0 * l) * i * (1.0 - i)
    gentle = ((16.0 * i - 12.0) * i + 4.0) * i
    steep = i.clamp(min=0.0).sqrt()
    reach = torch.where(i <= 0.25, gentle, steep)
    return torch.where(l <= 0.5, dark, i + (2.0 * l - 1.0) * (reach - i))


def _clip_colour(rgb, ceiling):
    """Pull a colour back inside the range while holding its brightness.

    Args:
        rgb: ``(..., 3)`` colours.
        ceiling: The largest value the range allows.

    Returns:
        The colours, each inside ``0`` to ``ceiling``.
    """
    lum = _luminance(rgb).unsqueeze(-1)
    low = rgb.amin(dim=-1, keepdim=True)
    high = rgb.amax(dim=-1, keepdim=True)
    below = lum - low
    above = high - lum
    rgb = torch.where(
        low < 0.0, lum + _divided((rgb - lum) * lum, below), rgb
    )
    return torch.where(
        high > ceiling, lum + _divided((rgb - lum) * (ceiling - lum), above), rgb
    )


def _set_luminance(rgb, target, ceiling):
    """Move a colour to a brightness, keeping its hue and saturation."""
    return _clip_colour(rgb + (target - _luminance(rgb)).unsqueeze(-1), ceiling)


def _saturation_of(rgb):
    """How far a colour's channels spread apart."""
    return rgb.amax(dim=-1) - rgb.amin(dim=-1)


def _set_saturation(rgb, target):
    """Stretch a colour's channels to a spread, keeping their order."""
    low = rgb.amin(dim=-1, keepdim=True)
    high = rgb.amax(dim=-1, keepdim=True)
    spread = high - low
    stretched = _divided((rgb - low) * target.unsqueeze(-1), spread)
    return torch.where(spread > EPSILON, stretched, torch.zeros_like(rgb))


def _hue(i, l):
    """The backdrop's brightness and saturation carrying the layer's hue."""
    ceiling = _ceiling(i, l)
    return _set_luminance(
        _set_saturation(l, _saturation_of(i)), _luminance(i), ceiling
    )


def _saturation(i, l):
    """The backdrop's brightness and hue carrying the layer's saturation."""
    ceiling = _ceiling(i, l)
    return _set_luminance(
        _set_saturation(i, _saturation_of(l)), _luminance(i), ceiling
    )


def _color(i, l):
    """The backdrop's brightness carrying the layer's hue and saturation."""
    return _set_luminance(l, _luminance(i), _ceiling(i, l))


def _luminosity(i, l):
    """The backdrop's hue and saturation carrying the layer's brightness."""
    return _set_luminance(i, _luminance(l), _ceiling(i, l))


#: Blend mode -> the whole-pixel mix it uses instead of a per-channel one.
_PIXEL = {
    "hue": _hue,
    "saturation": _saturation,
    "color": _color,
    "luminosity": _luminosity,
}


def known(mode: str) -> bool:
    """Whether a name is one of :data:`MODES`.

    Args:
        mode: The name to test.

    Returns:
        Whether the name has a mix behind it.
    """
    return mode in _PIXEL or mode in _CEILED or mode in _CHANNEL


def blend(backdrop: torch.Tensor, source: torch.Tensor, mode: str = "normal") -> torch.Tensor:
    """Mix two colour planes through one blend mode.

    Args:
        backdrop: ``(..., 3)`` colours already in place.
        source: ``(..., 3)`` colours going over them.
        mode: One of :data:`MODES`. An unknown name answers ``source``.

    Returns:
        A tensor shaped as the two that went in.
    """
    whole = _PIXEL.get(mode)
    if whole is not None:
        return whole(backdrop, source)
    ceiled = _CEILED.get(mode)
    if ceiled is not None:
        return ceiled(backdrop, source, ceiling_of(backdrop, source))
    channel = _CHANNEL.get(mode)
    return channel(backdrop, source) if channel is not None else source
