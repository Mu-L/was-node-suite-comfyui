"""Bringing linear light above 1.0 down to a range a display can show.

Images are ``(..., 3)`` linear light, 1.0 being diffuse white and anything above it a
highlight. Every operator answers 0 to 1.
"""

from __future__ import annotations

__all__ = ["APPLIED_TO", "OPERATORS", "STOP", "mapped", "shoulder"]

import torch

#: The operators offered, in the order a combo lists them.
OPERATORS = (
    "reinhard",
    "reinhard extended",
    "hable",
    "aces",
    "drago",
    "clip",
)

#: Whether an operator runs on each channel or on brightness alone, in menu order.
APPLIED_TO = ("each channel", "brightness")

#: What one stop of exposure multiplies the light by.
STOP = 2.0

#: Weights brightness is measured with, matching the rest of the pack.
LUMINANCE = (0.2224884, 0.71690369, 0.06060791)

#: Below this a division is read as a division by zero.
EPSILON = 1e-6

#: Curve constants of the Uncharted 2 filmic operator, in the order its author names them:
#: shoulder strength, linear strength, linear angle, toe strength, toe numerator, toe
#: denominator.
HABLE = (0.15, 0.50, 0.10, 0.20, 0.02, 0.30)

#: Input level the Uncharted 2 curve is normalised against.
HABLE_WHITE = 11.2

#: Rows of the ACES input and output fits, as numerator and denominator coefficients.
ACES = (2.51, 0.03, 2.43, 0.59, 0.14)


def _reinhard(light, white: float):
    """``x / (1 + x)``, which never reaches 1 and never clips."""
    return light / (1.0 + light)


def _reinhard_extended(light, white: float):
    """Reinhard rolled off so ``white`` and everything above it reaches 1."""
    ceiling = max(float(white), EPSILON)
    return (light * (1.0 + light / (ceiling * ceiling))) / (1.0 + light)


def _hable_curve(light):
    """The Uncharted 2 filmic curve, before its white normalisation."""
    a, b, c, d, e, f = HABLE
    return ((light * (a * light + c * b) + d * e) / (light * (a * light + b) + d * f)) - e / f


def _hable(light, white: float):
    """The Uncharted 2 filmic operator, normalised so ``white`` reaches 1."""
    scale = _hable_curve(torch.tensor(HABLE_WHITE, dtype=light.dtype, device=light.device))
    return _hable_curve(light) / scale.clamp(min=EPSILON)


def _aces(light, white: float):
    """The Narkowicz fit of the ACES filmic curve."""
    a, b, c, d, e = ACES
    return (light * (a * light + b)) / (light * (c * light + d) + e)


def _drago(light, white: float):
    """A logarithmic operator, which holds detail far into the highlights."""
    ceiling = max(float(white), 1.0 + EPSILON)
    top = torch.log1p(light)
    base = torch.log(
        torch.tensor(1.0 + ceiling, dtype=light.dtype, device=light.device)
    ).clamp(min=EPSILON)
    return top / base


def _clip(light, white: float):
    """Everything above 1.0 held at 1.0."""
    return light


#: Operator name -> the curve behind it.
_CURVES = {
    OPERATORS[0]: _reinhard,
    OPERATORS[1]: _reinhard_extended,
    OPERATORS[2]: _hable,
    OPERATORS[3]: _aces,
    OPERATORS[4]: _drago,
    OPERATORS[5]: _clip,
}


def shoulder(operator: str) -> bool:
    """Whether an operator reads a white point.

    Args:
        operator: One of :data:`OPERATORS`.

    Returns:
        Whether ``white`` changes what the operator answers.
    """
    return operator in (OPERATORS[1], OPERATORS[4])


def mapped(image: torch.Tensor, operator: str, white: float = 4.0,
           applied_to: str = APPLIED_TO[0]) -> torch.Tensor:
    """Bring linear light into 0 to 1 through one operator.

    Args:
        image: ``(..., 3)`` linear light, negatives read as zero.
        operator: One of :data:`OPERATORS`. An unknown name clips.
        white: Level that reaches 1, read by ``reinhard extended`` and ``drago``.
        applied_to: One of :data:`APPLIED_TO`. ``brightness`` maps the brightness plane
            and scales the three channels by the same ratio, which keeps the hue a
            per-channel map would wash out.

    Returns:
        A tensor the shape, dtype and device of ``image``, in 0 to 1.
    """
    curve = _CURVES.get(operator, _clip)
    light = image.to(dtype=torch.float32).clamp(min=0.0)

    if applied_to == APPLIED_TO[1] and int(light.shape[-1]) >= 3:
        weights = torch.tensor(LUMINANCE, dtype=light.dtype, device=light.device)
        before = (light[..., :3] * weights).sum(dim=-1, keepdim=True)
        after = curve(before, white)
        ratio = torch.where(
            before > EPSILON, after / before.clamp(min=EPSILON), torch.ones_like(before)
        )
        toned = light.clone()
        toned[..., :3] = light[..., :3] * ratio
    else:
        toned = curve(light, white)

    return toned.clamp(0.0, 1.0).to(dtype=image.dtype)
