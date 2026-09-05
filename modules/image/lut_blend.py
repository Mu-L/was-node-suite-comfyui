"""Mixing two colour lookup tables into one.

Every mix takes two ``(size, size, size, 3)`` sample cubes and a weight in ``[0, 1]``, and
answers samples of the same shape. A cube arriving as an array leaves as one.
"""

from __future__ import annotations

import math

import torch

from . import blend_modes
from .accelerate import run_on

__all__ = ["BLEND_MODES", "blend"]

#: Selectable mixing methods, in menu order.
BLEND_MODES = (
    "linear",
    "cosine",
    "smoothstep",
    "slerp",
    "hsv",
    "lab",
    "oklab",
    "auto",
    "multiply",
    "screen",
    "overlay",
)

#: Below this sine of the angle between two colour directions, slerp is a division by
#: nearly zero and linear interpolation is used instead.
PARALLEL = 1e-4

#: Angle past which :func:`_blend_auto` prefers slerp, in radians.
AUTO_ANGLE = 15.0 * math.pi / 180.0

#: D50 white point, the reference Lab is computed against.
LAB_WHITE = (0.96422, 1.0, 0.82521)

#: Smallest divisor a normalisation will use.
EPSILON = 1e-8


def _mix(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """One cube faded towards another."""
    return a * (1.0 - t) + b * t


def _blend_linear(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    return _mix(a, b, t)


def _layered(mode: str):
    """A mix that runs one of the shared blend modes and fades the result in.

    Args:
        mode: A name :func:`modules.image.blend_modes.blend` knows.

    Returns:
        A callable taking ``(a, b, t, ceiling)``.
    """

    def mixer(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
        return _mix(a, blend_modes.blend(a, b, mode), t)

    return mixer


def _blend_cosine(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    shaped = (1.0 - math.cos(math.pi * float(t))) * 0.5
    return _mix(a, b, shaped)


def _blend_smoothstep(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    shaped = float(t)
    shaped = shaped * shaped * (3.0 - 2.0 * shaped)
    return _mix(a, b, shaped)


def _blend_slerp(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    """Rotate each colour towards the other and interpolate brightness separately.

    Args:
        a: Samples of the first table.
        b: Samples of the second table, same shape as ``a``.
        t: Weight in ``[0, 1]``. 0.0 keeps ``a``, 1.0 reaches ``b``.
        ceiling: Largest value the result may hold.

    Returns:
        Samples of the same shape, inside ``0`` to ``ceiling``.
    """
    left_length = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    right_length = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    left_unit = a / left_length.clamp(min=EPSILON)
    right_unit = b / right_length.clamp(min=EPSILON)

    dot = (left_unit * right_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.arccos(dot)
    sine = torch.sin(angle)
    weight = float(t)
    # Below PARALLEL the rotation is a division by nearly zero, so those samples take the
    # straight interpolation instead.
    parallel = sine < PARALLEL
    safe = sine.clamp(min=EPSILON)

    left_coeff = torch.where(
        parallel, torch.full_like(sine, 1.0 - weight), torch.sin((1.0 - weight) * angle) / safe
    )
    right_coeff = torch.where(
        parallel, torch.full_like(sine, weight), torch.sin(weight * angle) / safe
    )
    unit = left_coeff * left_unit + right_coeff * right_unit
    unit = unit / torch.linalg.vector_norm(unit, dim=-1, keepdim=True).clamp(min=EPSILON)

    length = (1.0 - weight) * left_length + weight * right_length
    return (unit * length).clamp(0.0, ceiling)


def _rgb_to_hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split RGB into hue in ``[0, 1)``, saturation and value."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high = torch.maximum(torch.maximum(red, green), blue)
    low = torch.minimum(torch.minimum(red, green), blue)
    saturation = torch.where(
        high > 0, (high - low) / high.clamp(min=1e-8), torch.zeros_like(high)
    )
    spread = (high - low).clamp(min=1e-8)
    red_share = (high - red) / spread
    green_share = (high - green) / spread
    blue_share = (high - blue) / spread
    lit = high != low
    hue = torch.zeros_like(high)
    hue = torch.where((high == red) & lit, (blue_share - green_share) / 6.0, hue)
    hue = torch.where((high == green) & lit, (2.0 + red_share - blue_share) / 6.0, hue)
    hue = torch.where((high == blue) & lit, (4.0 + green_share - red_share) / 6.0, hue)
    return hue % 1.0, saturation, high


def _hsv_to_rgb(
    hue: torch.Tensor, saturation: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Rebuild RGB from hue, saturation and value."""
    sixths = (hue % 1.0) * 6.0
    sector = torch.floor(sixths)
    offset = sixths - sector
    down = value * (1.0 - saturation)
    falling = value * (1.0 - saturation * offset)
    rising = value * (1.0 - saturation * (1.0 - offset))
    wrapped = sector.to(torch.long) % 6
    red = torch.zeros_like(hue)
    green = torch.zeros_like(hue)
    blue = torch.zeros_like(hue)
    for index, (channel_r, channel_g, channel_b) in enumerate(
        (
            (value, rising, down),
            (falling, value, down),
            (down, value, rising),
            (down, falling, value),
            (rising, down, value),
            (value, down, falling),
        )
    ):
        sixth = wrapped == index
        red = torch.where(sixth, channel_r, red)
        green = torch.where(sixth, channel_g, green)
        blue = torch.where(sixth, channel_b, blue)
    return torch.stack((red, green, blue), dim=-1)


def _blend_hsv(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    """Interpolate hue the short way round the wheel, saturation and value straight."""
    hue_a, sat_a, val_a = _rgb_to_hsv(a)
    hue_b, sat_b, val_b = _rgb_to_hsv(b)
    weight = float(t)
    shortest = ((hue_b - hue_a + 0.5) % 1.0) - 0.5
    hue = (hue_a + weight * shortest) % 1.0
    saturation = _mix(sat_a, sat_b, weight)
    value = _mix(val_a, val_b, weight)
    return _hsv_to_rgb(hue, saturation, value).clamp(0.0, ceiling)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Undo the sRGB transfer curve."""
    steep = ((x + 0.055) / 1.055).clamp(min=0.0) ** 2.4
    return torch.where(x <= 0.04045, x / 12.92, steep)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Apply the sRGB transfer curve."""
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    """The real cube root, defined either side of zero."""
    return torch.sign(x) * x.abs() ** (1 / 3)


def _transform(values: torch.Tensor, matrix) -> torch.Tensor:
    """Multiply every triple in ``values`` by a 3x3 matrix, guarding against overflow.

    Args:
        values: ``(..., 3)`` samples.
        matrix: Three rows of three floats.

    Returns:
        ``(..., 3)`` samples, with any overflow replaced by a finite value.
    """
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    out = torch.stack(
        [red * row[0] + green * row[1] + blue * row[2] for row in matrix], dim=-1
    )
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

XYZ_TO_RGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)

D65_TO_D50 = (
    (1.0478112, 0.0228866, -0.0501270),
    (0.0295424, 0.9904844, -0.0170491),
    (-0.0092345, 0.0150436, 0.7521316),
)

D50_TO_D65 = (
    (0.9555766, -0.0230393, 0.0631636),
    (-0.0282895, 1.0099416, 0.0210077),
    (0.0122982, -0.0204830, 1.3299098),
)

LINEAR_TO_LMS = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)

LMS_TO_LINEAR = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


def _rgb_to_lab(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sRGB to CIE Lab against a D50 white point."""
    xyz = _transform(_transform(_srgb_to_linear(rgb), RGB_TO_XYZ), D65_TO_D50).clamp(min=0.0)
    white_x, white_y, white_z = LAB_WHITE
    knee = (6 / 29) ** 3
    slope = (29 / 6) ** 2 / 3

    def shape(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value > knee, value.clamp(min=0.0) ** (1 / 3), slope * value + 4 / 29)

    fx = shape(xyz[..., 0] / max(white_x, 1e-8))
    fy = shape(xyz[..., 1] / max(white_y, 1e-8))
    fz = shape(xyz[..., 2] / max(white_z, 1e-8))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _lab_to_rgb(
    lightness: torch.Tensor,
    green_red: torch.Tensor,
    blue_yellow: torch.Tensor,
    ceiling: float,
) -> torch.Tensor:
    """Convert CIE Lab back to sRGB."""
    fy = (lightness + 16.0) / 116.0
    fx = fy + (green_red / 500.0)
    fz = fy - (blue_yellow / 200.0)
    knee = 6 / 29
    slope = 3 * (knee ** 2)

    def unshape(value: torch.Tensor) -> torch.Tensor:
        return torch.where(value > knee, value ** 3, (value - 4 / 29) * slope)

    white_x, white_y, white_z = LAB_WHITE
    xyz = torch.stack(
        (unshape(fx) * white_x, unshape(fy) * white_y, unshape(fz) * white_z), dim=-1
    )
    linear = _transform(_transform(xyz, D50_TO_D65), XYZ_TO_RGB)
    return _linear_to_srgb(linear).clamp(0.0, ceiling)


def _rgb_to_oklab(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sRGB to Oklab."""
    lms = _transform(_srgb_to_linear(rgb), LINEAR_TO_LMS)
    long_, medium, short = _cbrt(lms[..., 0]), _cbrt(lms[..., 1]), _cbrt(lms[..., 2])
    return (
        0.2104542553 * long_ + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long_ - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long_ + 0.7827717662 * medium - 0.8086757660 * short,
    )


def _oklab_to_rgb(
    lightness: torch.Tensor,
    green_red: torch.Tensor,
    blue_yellow: torch.Tensor,
    ceiling: float,
) -> torch.Tensor:
    """Convert Oklab back to sRGB."""
    long_ = lightness + 0.3963377774 * green_red + 0.2158037573 * blue_yellow
    medium = lightness - 0.1055613458 * green_red - 0.0638541728 * blue_yellow
    short = lightness - 0.0894841775 * green_red - 1.2914855480 * blue_yellow
    lms = torch.stack((long_ ** 3, medium ** 3, short ** 3), dim=-1)
    return _linear_to_srgb(_transform(lms, LMS_TO_LINEAR)).clamp(0.0, ceiling)


def _blend_lab(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    """Interpolate in CIE Lab, clamped to the range real colours occupy."""
    la, aa, ba = _rgb_to_lab(a)
    lb, ab, bb = _rgb_to_lab(b)
    weight = float(t)
    lightness = _mix(la, lb, weight).clamp(0.0, 100.0)
    green_red = _mix(aa, ab, weight).clamp(-128.0, 128.0)
    blue_yellow = _mix(ba, bb, weight).clamp(-128.0, 128.0)
    return _lab_to_rgb(lightness, green_red, blue_yellow, ceiling)


def _blend_oklab(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    """Interpolate in Oklab."""
    la, aa, ba = _rgb_to_oklab(a)
    lb, ab, bb = _rgb_to_oklab(b)
    weight = float(t)
    return _oklab_to_rgb(
        _mix(la, lb, weight), _mix(aa, ab, weight), _mix(ba, bb, weight), ceiling
    )


def _blend_auto(a: torch.Tensor, b: torch.Tensor, t: float, ceiling: float) -> torch.Tensor:
    """Use slerp where the two colours point far apart, linear where they nearly agree."""
    left_unit = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp(min=EPSILON)
    right_unit = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp(min=EPSILON)
    dot = (left_unit * right_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    rotated = torch.arccos(dot) > AUTO_ANGLE
    return torch.where(
        rotated, _blend_slerp(a, b, t, ceiling), _blend_linear(a, b, t, ceiling)
    )


#: Mixes that run on the compute device. The rest run on the CPU.
_ACCELERATED = frozenset({"slerp", "hsv", "lab", "oklab", "auto"})

#: Mode name -> the function that mixes two tables that way.
_BLENDS = {
    "linear": _blend_linear,
    "cosine": _blend_cosine,
    "smoothstep": _blend_smoothstep,
    "slerp": _blend_slerp,
    "hsv": _blend_hsv,
    "lab": _blend_lab,
    "oklab": _blend_oklab,
    "auto": _blend_auto,
    "multiply": _layered("multiply"),
    "screen": _layered("screen"),
    "overlay": _layered("overlay"),
}


def blend(a, b, mode: str, t: float):
    """Mix two tables of the same shape.

    Args:
        a: Samples of the first table, an array or a tensor.
        b: Samples of the second table, same shape as ``a``.
        mode: One of :data:`BLEND_MODES`. Any other name is treated as ``overlay``, which
            is what a workflow saved against a mode that no longer exists lands on.
        t: Weight in ``[0, 1]``. 0.0 keeps ``a``, 1.0 reaches ``b``.

    Returns:
        Float32 samples of the same shape, inside ``0`` to the highest value either table
        held. An array in answers an array out; a tensor answers a tensor on its own device.
    """
    left = torch.as_tensor(a, dtype=torch.float32)
    right = torch.as_tensor(b, dtype=torch.float32)
    ceiling = blend_modes.ceiling_of(left, right)
    weight = float(t)
    mixer = _BLENDS.get(mode, _BLENDS["overlay"])

    def work(moved: torch.Tensor) -> torch.Tensor:
        return mixer(moved, right.to(moved.device), weight, ceiling).clamp(0.0, ceiling)

    mixed = run_on(left, work, prefer_gpu=mode in _ACCELERATED)
    if isinstance(a, torch.Tensor):
        return mixed
    return mixed.contiguous().numpy()
