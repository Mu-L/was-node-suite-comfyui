"""What a picture becomes on the way to a sensor reading.

Images are ``(batch, height, width, 3)``, and a colour filter array plane is
``(batch, height, width)`` in the order :class:`Profile` names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "CFA_PATTERNS",
    "PROFILES",
    "Profile",
    "apply_matrix",
    "camera_from_srgb",
    "encode",
    "invert_gains",
    "linearise",
    "mosaic",
]

#: Rows of the sRGB primaries in CIE XYZ, the D65 matrix every camera profile is written
#: against.
SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

#: CIE XYZ to linear sRGB, the matrix a file written against the sRGB primaries carries.
XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)

#: Where the sRGB transfer function stops being a straight line, and the slope and shape of
#: the two pieces either side of it.
SRGB_KNEE = 0.04045
SRGB_KNEE_LINEAR = 0.0031308
SRGB_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_EXPONENT = 2.4

#: Level at and above which a pixel counts as saturated, and is held up rather than pulled
#: down when white balance is taken back out.
HIGHLIGHT = 0.9

#: Smallest value carried into a linear image.
FLOOR = 1e-8

#: Colour of each pixel of a two by two tile, top left first, reading across then down.
CFA_PATTERNS = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}


@dataclass(frozen=True)
class Profile:
    """What makes a sensor's output read as that sensor.

    Attributes:
        name: What the profile is called.
        xyz_to_camera: Rows of the CIE XYZ to camera matrix, the DNG ``ColorMatrix1``.
        red_gain: White balance gain the camera applied to red.
        blue_gain: White balance gain the camera applied to blue.
        exposure: Overall gain applied on top of the two channel gains.
        cfa: A key of :data:`CFA_PATTERNS`.
        black_level: Level the sensor reads with no light, on a 0 to 1 scale.
        white_level: Level the sensor saturates at, on a 0 to 1 scale.
        shot_noise: Variance per unit of signal.
        read_noise: Variance present with no signal.
    """

    name: str = "Generic"
    xyz_to_camera: tuple = field(
        default=(
            (1.0234, -0.2969, -0.2266),
            (-0.5625, 1.6328, -0.0469),
            (-0.0703, 0.2188, 0.6406),
        )
    )
    red_gain: float = 2.0
    blue_gain: float = 1.7
    exposure: float = 1.0
    cfa: str = "RGGB"
    black_level: float = 0.0
    white_level: float = 1.0
    shot_noise: float = 0.0
    read_noise: float = 0.0


#: Widget option -> the sensor a file is written against.
PROFILES = {
    "sRGB primaries": Profile(
        name="sRGB primaries", xyz_to_camera=XYZ_TO_SRGB, red_gain=1.0, blue_gain=1.0
    ),
    "generic camera": Profile(name="generic camera"),
}


def linearise(image: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Undo the transfer function an sRGB picture is stored with.

    Args:
        image: ``(batch, height, width, 3)``, 0 to 1 for a picture and above it for a
            highlight that carries more.
        gamma: Treat the transfer as this power instead of the sRGB curve.

    Returns:
        A tensor of the same shape holding light-linear values. A code above 1.0 answers
        light above 1.0, since the curve is a power law and extends.
    """
    held = image.clamp(min=0.0)
    if gamma is not None:
        return held.clamp(min=FLOOR) ** gamma
    straight = held / SRGB_SLOPE
    curved = ((held + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)).clamp(min=FLOOR) ** SRGB_EXPONENT
    return torch.where(held <= SRGB_KNEE, straight, curved)


def encode(image: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Apply the transfer function an sRGB picture is stored with.

    Args:
        image: ``(batch, height, width, 3)`` holding light-linear values, above 1.0 for a
            highlight brighter than white.
        gamma: Treat the transfer as this power instead of the sRGB curve.

    Returns:
        A tensor of the same shape, 0 to 1 for light that fitted and above it for light
        that did not.
    """
    held = image.clamp(min=0.0)
    if gamma is not None:
        return held.clamp(min=FLOOR) ** (1.0 / gamma)
    straight = held * SRGB_SLOPE
    curved = (1.0 + SRGB_OFFSET) * held.clamp(min=FLOOR) ** (1.0 / SRGB_EXPONENT) - SRGB_OFFSET
    return torch.where(held <= SRGB_KNEE_LINEAR, straight, curved)


def camera_from_srgb(profile: Profile, device=None, dtype=torch.float32) -> torch.Tensor:
    """The matrix taking linear sRGB to this camera's own colour space.

    Args:
        profile: Profile holding the XYZ to camera matrix.
        device: Device the matrix is built on.
        dtype: Dtype the matrix is built in.

    Returns:
        A ``(3, 3)`` tensor whose rows each sum to 1.
    """
    to_xyz = torch.tensor(SRGB_TO_XYZ, device=device, dtype=dtype)
    to_camera = torch.tensor(profile.xyz_to_camera, device=device, dtype=dtype)
    combined = to_camera @ to_xyz
    return combined / combined.sum(dim=-1, keepdim=True).clamp(min=FLOOR)


def apply_matrix(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Send every pixel through a 3 by 3 colour matrix.

    Args:
        image: ``(batch, height, width, 3)``.
        matrix: ``(3, 3)`` tensor.

    Returns:
        A tensor of the same shape as ``image``.
    """
    flat = image.reshape(-1, 3)
    return (flat @ matrix.to(device=image.device, dtype=image.dtype).T).reshape(image.shape)


def _gains(profile: Profile, device, dtype) -> torch.Tensor:
    """The three per-channel gains a camera applied, as a ``(3,)`` tensor."""
    return torch.tensor(
        [profile.red_gain, 1.0, profile.blue_gain], device=device, dtype=dtype
    ) * profile.exposure


def invert_gains(image: torch.Tensor, profile: Profile) -> torch.Tensor:
    """Take white balance back out, holding saturated pixels up.

    Args:
        image: ``(batch, height, width, 3)`` holding light-linear values.
        profile: Profile holding the gains to remove.

    Returns:
        A tensor of the same shape.
    """
    inverse = 1.0 / _gains(profile, image.device, image.dtype).clamp(min=FLOOR)
    inverse = inverse.view(1, 1, 1, 3)
    grey = image.mean(dim=-1, keepdim=True)
    # A pixel at or above the highlight keeps its level rather than being pulled down.
    weight = ((grey - HIGHLIGHT).clamp(min=0.0) / (1.0 - HIGHLIGHT)) ** 2.0
    held = torch.maximum(weight + (1.0 - weight) * inverse, inverse)
    return image * held


def _tile(cfa: str) -> tuple:
    """The four colour indices of one tile, raising when the pattern is unknown."""
    order = CFA_PATTERNS.get(cfa.upper())
    if order is None:
        raise ValueError(
            f"CFA pattern must be one of {', '.join(CFA_PATTERNS)}, not {cfa!r}"
        )
    return order


def mosaic(image: torch.Tensor, cfa: str = "RGGB") -> torch.Tensor:
    """Keep one colour per pixel, in the sensor's tile order.

    Args:
        image: ``(batch, height, width, 3)``. Both sides are used to the even pixel below.
        cfa: A key of :data:`CFA_PATTERNS`.

    Returns:
        A ``(batch, height, width)`` tensor, height and width each rounded down to even.

    Raises:
        ValueError: ``cfa`` names no known pattern.
    """
    order = _tile(cfa)
    height = image.shape[1] - image.shape[1] % 2
    width = image.shape[2] - image.shape[2] % 2
    cut = image[:, :height, :width, :]
    plane = torch.empty(cut.shape[:3], device=image.device, dtype=image.dtype)
    plane[:, 0::2, 0::2] = cut[:, 0::2, 0::2, order[0]]
    plane[:, 0::2, 1::2] = cut[:, 0::2, 1::2, order[1]]
    plane[:, 1::2, 0::2] = cut[:, 1::2, 0::2, order[2]]
    plane[:, 1::2, 1::2] = cut[:, 1::2, 1::2, order[3]]
    return plane


