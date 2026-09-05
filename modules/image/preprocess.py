"""Images prepared for a ControlNet, computed in torch.

Every function takes and returns a ``(batch, channels, height, width)`` float tensor on a 0
to 255 scale. A result is three channels whatever the source carried.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .convolve import canny as canny_edges
from .convolve import gaussian_blur, luminance

__all__ = [
    "binary",
    "lineart_simple",
    "normal_from_depth",
    "pyramid_canny",
    "scribble_xdog",
    "shuffle",
]

#: Scales the pyramid canny accumulates over, coarsest first.
PYRAMID_STEPS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Weight a finer pyramid level takes against everything coarser already accumulated.
PYRAMID_BLEND = 0.25

#: Percentiles the pyramid result is stretched between.
PYRAMID_RANGE = (1.0, 99.0)

#: Shortest side a pyramid level may have. Tracing pads by one on each edge, so a level
#: narrower than this has nothing left in the middle.
PYRAMID_SMALLEST = 3


def _as_three(x: torch.Tensor) -> torch.Tensor:
    """Repeat a single channel up to three, or drop alpha from four."""
    if x.shape[1] == 3:
        return x
    if x.shape[1] >= 3:
        return x[:, :3]
    return x[:, :1].repeat(1, 3, 1, 1)


def pyramid_canny(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Trace edges at nine scales and stack them, so a soft edge survives.

    Args:
        x: Source images.
        low: Gradient strength an edge continues at, 0 to 255.
        high: Gradient strength an edge starts at, 0 to 255.

    Returns:
        A three-channel result, stretched between its 1st and 99th percentiles.
    """
    height, width = x.shape[-2:]
    accumulated = None
    for scale in PYRAMID_STEPS:
        size = (max(1, int(height * scale)), max(1, int(width * scale)))
        if min(size) < PYRAMID_SMALLEST:
            continue
        small = F.interpolate(x, size=size, mode="area")
        # Each channel is traced on its own, as the pyramid this reproduces does.
        edge = torch.cat(
            [canny_edges(small[:, c: c + 1], float(low), float(high)).to(x.dtype)
             for c in range(min(3, small.shape[1]))],
            dim=1,
        )
        if accumulated is None:
            accumulated = edge
            continue
        accumulated = F.interpolate(accumulated, size=size, mode="bilinear", align_corners=False)
        accumulated = accumulated * (1.0 - PYRAMID_BLEND) + edge * PYRAMID_BLEND
    if accumulated is None:
        # Every level was narrower than one can be traced, so the frame carries no edge.
        return _as_three(x.new_zeros((x.shape[0], 1, height, width)))
    summed = accumulated.sum(dim=1, keepdim=True)
    summed = F.interpolate(summed, size=(height, width), mode="bilinear", align_corners=False)
    return _as_three(_stretch(summed, *PYRAMID_RANGE))


def _stretch(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Stretch each image between two percentiles onto 0 to 255.

    Args:
        x: One channel per image, of any range.
        low: Percentile taken as the new black.
        high: Percentile taken as the new white.

    Returns:
        A tensor of the same shape on a 0 to 255 scale. A plane with nothing between the two
        percentiles is answered by what it holds rather than by the black a zero-width
        stretch would give.
    """
    out = torch.empty_like(x)
    for index in range(x.shape[0]):
        flat = x[index].flatten()
        bottom, top = torch.quantile(
            flat.float(), torch.tensor([low / 100.0, high / 100.0], device=x.device)
        )
        span = top - bottom
        if float(span) <= 1e-5:
            # Every sample sits at one level, so the plane is all edge or none of it.
            out[index] = torch.where(x[index] > 0, 255.0, 0.0).to(x.dtype)
            continue
        out[index] = ((x[index] - bottom) / span * 255.0).clamp(0.0, 255.0)
    return out


def binary(x: torch.Tensor, threshold: int) -> torch.Tensor:
    """Reduce to two tones, picking the split per image where none is given.

    Args:
        x: Source images.
        threshold: Brightness the split is made at, 0 to 255. 0 picks one per image with
            Otsu's method.

    Returns:
        A three-channel result.
    """
    grey = luminance(x)
    if int(threshold) <= 0 or int(threshold) >= 255:
        level = torch.stack([_otsu(grey[i]) for i in range(grey.shape[0])]).view(-1, 1, 1, 1)
    else:
        level = torch.tensor(float(threshold), dtype=x.dtype, device=x.device)
    return _as_three(torch.where(grey > level, 255.0, 0.0).to(x.dtype))


def _otsu(plane: torch.Tensor) -> torch.Tensor:
    """The brightness that best separates one plane into two groups."""
    counts = torch.histc(plane.float(), bins=256, min=0.0, max=255.0)
    weights = counts / counts.sum().clamp_min(1.0)
    levels = torch.arange(256, dtype=torch.float32, device=plane.device)
    below = torch.cumsum(weights, 0)
    above = 1.0 - below
    mean_below = torch.cumsum(weights * levels, 0) / below.clamp_min(1e-9)
    total = (weights * levels).sum()
    mean_above = (total - torch.cumsum(weights * levels, 0)) / above.clamp_min(1e-9)
    variance = below * above * (mean_below - mean_above) ** 2
    return levels[int(torch.argmax(variance))].to(plane.dtype)


def scribble_xdog(x: torch.Tensor, threshold: int) -> torch.Tensor:
    """Trace strokes with a difference of Gaussians, as black on white.

    Args:
        x: Source images.
        threshold: How much difference counts as a stroke, 1 to 64. Lower keeps more.

    Returns:
        A three-channel result.
    """
    colour = _as_three(x)
    difference = gaussian_blur(colour, sigma=5.0) - gaussian_blur(colour, sigma=0.5)
    dog = (255.0 - difference.amin(dim=1, keepdim=True)).clamp(0.0, 255.0)
    return _as_three(torch.where(2.0 * (255.0 - dog) > float(threshold), 0.0, 255.0).to(x.dtype))


def lineart_simple(x: torch.Tensor, sigma: float, threshold: int) -> torch.Tensor:
    """Trace strokes by how far each pixel sits below a blur of its surroundings.

    Args:
        x: Source images.
        sigma: Radius of the blur compared against, in pixels.
        threshold: Strength below which a difference is not counted when the result is
            normalised, 0 to 255.

    Returns:
        A three-channel result, black strokes on white.
    """
    colour = _as_three(x)
    blurred = gaussian_blur(colour, sigma=float(sigma))
    intensity = (blurred - colour).amin(dim=1, keepdim=True).clamp(0.0, 255.0)
    out = torch.empty_like(intensity)
    for index in range(intensity.shape[0]):
        plane = intensity[index]
        kept = plane[plane > float(threshold)]
        median = torch.median(kept) if kept.numel() else torch.tensor(0.0, device=plane.device)
        out[index] = plane / torch.clamp(median, min=16.0) * 127.0
    return _as_three(255.0 - out.clamp(0.0, 255.0))


def shuffle(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Displace the image along a smooth random flow, keeping its colours and losing its layout.

    Args:
        x: Source images.
        seed: Chooses the flow. The same seed always displaces the same way.

    Returns:
        A three-channel result.
    """
    colour = _as_three(x)
    batch, _, height, width = colour.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    coarse = torch.rand((batch, 2, 4, 4), generator=generator, dtype=torch.float32)
    flow = F.interpolate(
        coarse.to(colour.device), size=(height, width), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    # grid_sample samples in [-1, 1] with x first, so the two planes are scaled and stacked.
    grid = torch.stack([flow[:, 0] * 2.0 - 1.0, flow[:, 1] * 2.0 - 1.0], dim=-1)
    return F.grid_sample(
        colour, grid.to(colour.dtype), mode="bilinear", padding_mode="reflection",
        align_corners=False,
    )


#: Half-width of the window the surface slope is fitted over, so the window is 7 pixels a side.
SLOPE_RADIUS = 3


def _slope_kernels(radius: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """The pair of filters that read the slope of the best-fit plane through a window.

    Args:
        radius: Half-width of the window, so the window is ``2 * radius + 1`` a side.
        device: Where the filters are built.
        dtype: The filters' element type.

    Returns:
        The across and down filters, each shaped ``(1, 1, side, side)``.
    """
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    across = offsets.view(1, -1).expand(2 * radius + 1, -1)
    across = across / across.square().sum().clamp_min(1e-6)
    return across.view(1, 1, *across.shape), across.t().reshape(1, 1, *across.shape)


def normal_from_depth(
    depth: torch.Tensor, strength: float, radius: int = SLOPE_RADIUS
) -> torch.Tensor:
    """Read a depth map as a surface and answer the direction that surface faces.

    Args:
        depth: ``(batch, 1, height, width)`` depth on a 0 to 255 scale, bright for near.
        strength: How much the surface is tilted by a given change in depth. Larger leans
            the normals further from straight on.
        radius: Half-width of the window the slope is fitted over. 0 takes the slope from
            a three-tap difference instead.

    Returns:
        A three-channel result on a 0 to 255 scale, in the tangent-space layout a normal
        ControlNet reads: red across, green up, blue towards the viewer.
    """
    scaled = depth / 255.0 * max(float(strength), 1e-3)
    reach = max(int(radius), 0)

    if reach < 1:
        padded = F.pad(scaled, (1, 1, 1, 1), mode="replicate")
        across = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) * 0.5
        down = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) * 0.5
    else:
        horizontal, vertical = _slope_kernels(reach, scaled.device, scaled.dtype)
        padded = F.pad(scaled, (reach,) * 4, mode="replicate")
        across = F.conv2d(padded, horizontal)
        down = F.conv2d(padded, vertical)

    facing = torch.ones_like(across)
    normal = torch.cat([-across, down, facing], dim=1)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return ((normal + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0)
