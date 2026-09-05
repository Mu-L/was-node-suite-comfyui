"""Matching a batch's colour to one reference, by a single fixed transform.

Three methods: ``reinhard`` (mean and standard deviation per channel), ``mkl`` (full
covariance), ``histogram`` (whole tonal curve).
"""

# Nothing here is derived from or resembles the PyPI package `color-matcher`, which is
# GPL-3.0 and unusable in this MIT-licensed pack. Every function below is implemented
# straight from its published source: Reinhard, Adhikhmin, Gooch and Shirley 2001 for the
# mean/std transfer; Pitie, Kokaram and Dahyot, "Automated Colour Grading Using Colour
# Distribution Transfer", CVIU 2007, for the linear map and the regrain step.

from __future__ import annotations

from dataclasses import dataclass

import torch

from .blend_modes import ceiling_of

__all__ = [
    "COLOR_SPACES",
    "METHODS",
    "ReferenceStats",
    "color_match",
    "from_space",
    "map_pixels",
    "measure",
    "to_space",
]

METHODS = ["mkl", "reinhard", "histogram"]
COLOR_SPACES = ["RGB", "Lab"]

#: Added to a covariance's diagonal before the matrix square roots MKL takes, so a frame of
#: one flat colour, whose true covariance is singular, still has one to take a root of.
COVARIANCE_EPSILON = 1e-5

#: sRGB primaries against D65, taking linear RGB to CIE XYZ.
_XYZ_FROM_RGB = torch.tensor(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=torch.float64,
)

#: The same transform read backwards, taking CIE XYZ to linear RGB.
_RGB_FROM_XYZ = torch.linalg.inv(_XYZ_FROM_RGB)

#: CIE XYZ tristimulus values of D65 for a two degree observer.
_D65_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)


@dataclass
class ReferenceStats:
    """One colour distribution's mean and covariance, measured once.

    Attributes:
        mean: ``(channels,)``.
        covariance: ``(channels, channels)``, only populated for ``mkl``.
        std: ``(channels,)``, only populated for ``reinhard``.
        sorted_pixels: Every pixel of every channel, sorted ascending, only populated for
            ``histogram``. Rank-interpolating into this is the distribution's inverse CDF.
    """

    mean: torch.Tensor
    covariance: torch.Tensor | None = None
    std: torch.Tensor | None = None
    sorted_pixels: torch.Tensor | None = None


def _wide_dtype(device: torch.device) -> torch.dtype:
    """The widest float ``device`` computes in.

    Args:
        device: Where the maths runs.

    Returns:
        ``torch.float32`` on Metal, which carries no double, ``torch.float64`` elsewhere.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB to CIE Lab, against D65 and a two degree observer.

    Args:
        rgb: ``(..., 3)`` sRGB, 1.0 being white.

    Returns:
        ``(..., 3)`` holding L, a and b, L running 0 to 100 across a 0 to 1 input.
    """
    values = rgb.to(_wide_dtype(rgb.device))
    linear = torch.where(
        values > 0.04045,
        ((values + 0.055).clamp(min=0.0) / 1.055) ** 2.4,
        values / 12.92,
    )
    scaled = (linear @ _XYZ_FROM_RGB.to(values).T) / _D65_WHITE.to(values)
    curved = torch.where(
        scaled > 0.008856,
        scaled.clamp(min=0.0) ** (1.0 / 3.0),
        7.787 * scaled + 16.0 / 116.0,
    )
    x, y, z = curved.unbind(-1)
    return torch.stack([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], dim=-1)


def _lab_to_rgb(lab: torch.Tensor, ceiling: float) -> torch.Tensor:
    """CIE Lab back to sRGB, held inside 0 to ``ceiling``.

    Args:
        lab: ``(..., 3)`` holding L, a and b.
        ceiling: Highest value the result may carry.

    Returns:
        ``(..., 3)`` sRGB. A negative Z, which no colour has, is read as zero.
    """
    values = lab.to(_wide_dtype(lab.device))
    lightness, green_red, blue_yellow = values.unbind(-1)
    y = (lightness + 16.0) / 116.0
    x = green_red / 500.0 + y
    z = (y - blue_yellow / 200.0).clamp(min=0.0)
    curved = torch.stack([x, y, z], dim=-1)
    xyz = torch.where(curved > 0.2068966, curved ** 3.0, (curved - 16.0 / 116.0) / 7.787)
    linear = (xyz * _D65_WHITE.to(values)) @ _RGB_FROM_XYZ.to(values).T
    srgb = torch.where(
        linear > 0.0031308,
        1.055 * linear.clamp(min=0.0) ** (1.0 / 2.4) - 0.055,
        linear * 12.92,
    )
    return srgb.clamp(0.0, ceiling)


def to_space(images_bhwc: torch.Tensor, color_space: str) -> torch.Tensor:
    """RGB in ``[0, 1]`` to the space a method measures in.

    Args:
        images_bhwc: ``(batch, height, width, 3)``, RGB in ``[0, 1]``.
        color_space: One of :data:`COLOR_SPACES`.

    Returns:
        The same shape, dtype and device, converted.
    """
    if color_space == "RGB":
        return images_bhwc
    return _rgb_to_lab(images_bhwc).to(images_bhwc.dtype)


def from_space(values_bhwc: torch.Tensor, color_space: str, ceiling: float = 1.0) -> torch.Tensor:
    """The inverse of :func:`to_space`, held inside the range the caller allows.

    Args:
        values_bhwc: A batch in the space named by ``color_space``.
        color_space: One of :data:`COLOR_SPACES`.
        ceiling: Highest value the result may carry. 1.0 for a picture that arrived
            inside 0 to 1, the picture's own peak for one carrying light above white.

    Returns:
        RGB in ``[0, ceiling]``, the same shape, dtype and device as ``values_bhwc``.
    """
    if color_space == "RGB":
        return values_bhwc.clamp(0.0, ceiling)
    return _lab_to_rgb(values_bhwc, ceiling).to(values_bhwc.dtype)


def measure(pixels: torch.Tensor, method: str) -> ReferenceStats:
    """The statistics one distribution's pixels are measured down to for ``method``.

    Args:
        pixels: ``(n, channels)``, every pixel of a batch or a reference, in whatever
            space the caller is matching in.
        method: One of :data:`METHODS`.

    Returns:
        Only the fields ``method`` reads are populated.
    """
    mean = pixels.mean(dim=0)
    if method == "mkl":
        centered = pixels - mean
        covariance = (centered.T @ centered) / max(pixels.shape[0] - 1, 1)
        return ReferenceStats(mean=mean, covariance=covariance)
    if method == "reinhard":
        return ReferenceStats(mean=mean, std=pixels.std(dim=0, unbiased=False))
    sorted_pixels, _ = torch.sort(pixels, dim=0)
    return ReferenceStats(mean=mean, sorted_pixels=sorted_pixels)


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """A symmetric positive-semidefinite matrix's principal square root.

    Args:
        matrix: ``(channels, channels)``, symmetric.

    Returns:
        ``root`` such that ``root @ root`` is ``matrix``, by eigendecomposition. A
        covariance is always symmetric, so this is exact where :func:`torch.linalg.sqrtm`
        would need a Schur form for a general matrix.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    root_eigenvalues = eigenvalues.clamp(min=0.0).sqrt()
    return eigenvectors @ torch.diag(root_eigenvalues) @ eigenvectors.T


def _mkl_map(source: ReferenceStats, target: ReferenceStats) -> tuple[torch.Tensor, torch.Tensor]:
    """The affine map ``x' = A(x - mean_s) + mean_t`` that moves one Gaussian onto another.

    Args:
        source: The batch's own statistics.
        target: The reference's statistics.

    Returns:
        ``(A, mean_t)``, the linear part and the offset, per Pitie, Kokaram and Dahyot
        2007's closed form for the optimal transport map between two Gaussians:
        ``A = Cs^-1/2 (Cs^1/2 Ct Cs^1/2)^1/2 Cs^-1/2``.
    """
    channels = source.mean.shape[0]
    eye = torch.eye(channels, dtype=source.mean.dtype, device=source.mean.device)
    source_cov = source.covariance + COVARIANCE_EPSILON * eye
    target_cov = target.covariance + COVARIANCE_EPSILON * eye

    source_root = _matrix_sqrt(source_cov)
    source_root_inv = torch.linalg.inv(source_root)
    inner = source_root @ target_cov @ source_root
    transform = source_root_inv @ _matrix_sqrt(inner) @ source_root_inv
    return transform, target.mean


def _apply_mkl(pixels: torch.Tensor, source: ReferenceStats, target: ReferenceStats) -> torch.Tensor:
    transform, target_mean = _mkl_map(source, target)
    centered = pixels - source.mean
    return centered @ transform.T + target_mean


def _apply_reinhard(
    pixels: torch.Tensor, source: ReferenceStats, target: ReferenceStats, luminance_only: bool
) -> torch.Tensor:
    """Match mean and standard deviation, per channel.

    Args:
        pixels: ``(n, channels)`` to transform.
        source: The batch's own mean and standard deviation.
        target: The reference's.
        luminance_only: Match only the first channel (``L`` in Lab, or ``R`` alone in
            plain RGB, which is of little use there) and pass the rest through, so a
            colour cast the batch already has is not corrected away along with the drift.

    Returns:
        The transformed pixels.
    """
    scale = (target.std / source.std.clamp(min=1e-6)).clone()
    if luminance_only:
        scale[1:] = 1.0
        target_mean = target.mean.clone()
        target_mean[1:] = source.mean[1:]
    else:
        target_mean = target.mean
    return (pixels - source.mean) * scale + target_mean


def _apply_histogram(pixels: torch.Tensor, source: ReferenceStats, target: ReferenceStats) -> torch.Tensor:
    """Remap every channel's whole distribution onto the reference's, by rank.

    Args:
        pixels: ``(n, channels)`` to transform.
        source: Carries the batch's own sorted pixels, used only to build the rank of
            each input value; the mapped-to values come from ``target``.
        target: Carries the reference's sorted pixels, the inverse CDF this maps onto.

    Returns:
        The transformed pixels. Each channel is handled independently, so a value's rank
        among its own channel's pixels is preserved and its channel's shape is what moves.
    """
    channels = pixels.shape[1]
    out = torch.empty_like(pixels)
    source_sorted = source.sorted_pixels
    target_sorted = target.sorted_pixels
    for channel in range(channels):
        # rank = the fraction of the source's own pixels at or below this value, i.e. the
        # source CDF; reading target_sorted at that same fraction is F_target^-1(F_source(x)).
        rank = torch.searchsorted(
            source_sorted[:, channel].contiguous(), pixels[:, channel].contiguous()
        )
        rank = rank.clamp(max=source_sorted.shape[0] - 1)
        fraction = rank.to(_wide_dtype(rank.device)) / max(source_sorted.shape[0] - 1, 1)
        target_index = (fraction * (target_sorted.shape[0] - 1)).round().long()
        out[:, channel] = target_sorted[target_index, channel]
    return out


def _regrain(original: torch.Tensor, matched: torch.Tensor, strength: float) -> torch.Tensor:
    """Restore local detail a per-channel remap can blotch, in one channel's ``(n,)`` values.

    Args:
        original: The source pixels before matching, ``(n,)``, one channel.
        matched: The same pixels after :func:`_apply_histogram`.
        strength: How much of the original's local value survives, 0 to 1.

    Returns:
        The blended values.
    """
    return matched * (1.0 - strength) + original * strength


def color_match(
    images_bhwc: torch.Tensor,
    reference_bhwc: torch.Tensor,
    method: str,
    color_space: str,
    strength: float,
    luminance_only: bool,
    regrain_strength: float,
) -> torch.Tensor:
    """Match a batch's colour to one reference, by one transform applied to every frame.

    Args:
        images_bhwc: ``(batch, height, width, 3)`` to correct, RGB in ``[0, 1]``.
        reference_bhwc: ``(ref_batch, height, width, 3)`` the batch is matched to. Every
            frame of it is pooled into one distribution, so a reference of more than one
            frame is a target grade rather than a single shot to copy.
        method: One of :data:`METHODS`.
        color_space: One of :data:`COLOR_SPACES`.
        strength: How much of the match is applied, 0 leaves the batch unchanged and 1 is
            the full transform.
        luminance_only: For ``reinhard``, match brightness only and leave chroma as the
            batch's own. Ignored by the other methods.
        regrain_strength: For ``histogram``, how much of each pixel's original local value
            is blended back in afterwards, 0 for none. Ignored by the other methods.

    Returns:
        The corrected batch, same shape and dtype as ``images_bhwc``. A batch or reference
        carrying light above white keeps that peak; one already inside 0 to 1 is held there.
    """
    ceiling = ceiling_of(images_bhwc, reference_bhwc)
    working = to_space(images_bhwc, color_space)
    reference = to_space(reference_bhwc, color_space)

    batch, height, width, channels = working.shape
    source_pixels = working.reshape(-1, channels)
    target_pixels = reference.reshape(-1, channels)

    source_stats = measure(source_pixels, method)
    target_stats = measure(target_pixels, method)

    flat = working.reshape(batch * height * width, channels)
    blended = map_pixels(
        flat, source_stats, target_stats, method, strength, luminance_only, regrain_strength
    )
    return from_space(blended.reshape(batch, height, width, channels), color_space, ceiling)


def map_pixels(
    pixels: torch.Tensor,
    source_stats: ReferenceStats,
    target_stats: ReferenceStats,
    method: str,
    strength: float,
    luminance_only: bool = False,
    regrain_strength: float = 0.0,
) -> torch.Tensor:
    """Carry pixels through one measured match, in whatever space they were measured in.

    Args:
        pixels: ``(n, 3)`` in the working colour space.
        source_stats: What :func:`measure` answered for the images being corrected.
        target_stats: What :func:`measure` answered for the reference.
        method: One of :data:`METHODS`.
        strength: How much of the match to apply, 0 leaves the pixels alone.
        luminance_only: For ``reinhard``, match brightness only.
        regrain_strength: For ``histogram``, how much local value to blend back.

    Returns:
        ``(n, 3)`` mapped pixels. Separate from :func:`color_match` so the same transform
        can be carried by something that is not an image, such as an identity cube being
        baked into a lookup table.
    """
    if method == "mkl":
        matched = _apply_mkl(pixels, source_stats, target_stats)
    elif method == "reinhard":
        matched = _apply_reinhard(pixels, source_stats, target_stats, luminance_only)
    else:
        matched = _apply_histogram(pixels, source_stats, target_stats)
        if regrain_strength > 0.0:
            for channel in range(pixels.shape[-1]):
                matched[:, channel] = _regrain(
                    pixels[:, channel], matched[:, channel], regrain_strength
                )
    return pixels * (1.0 - strength) + matched * strength
