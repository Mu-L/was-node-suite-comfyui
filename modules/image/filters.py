"""Tonal and stylistic image filters.

:func:`shadows_and_highlights`, :func:`dragan_filter` and :func:`sparkle` each take a PIL
image and return a new one. Between those two conversions the work is float tensors shaped
``(height, width, channels)`` on a 0 to 1 scale.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw

from .blend_modes import blend, ceiling_of

__all__ = ["dragan_filter", "shadows_and_highlights", "sparkle"]

#: Box blur passes one gaussian radius is spread over.
_PASSES = 3

#: Weights an 8-bit greyscale conversion sums the channels with, over 65536.
_GREY = (19595, 38470, 7471)

#: The 3x3 kernel a sharpness enhancement smooths its degenerate copy with, over 13.
_SMOOTH = (1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0)

#: Points drawn in each glitter layer.
_PARTICLES = 5000

#: Coverage the bloom and both glitter layers composite at.
_LAYER_ALPHA = 128.0 / 255.0

#: Grey the high-pass layer is screened over.
_MID_GREY = 127.0 / 255.0


def _planes(image: Image.Image) -> torch.Tensor:
    """Read a PIL image into a float tensor.

    Args:
        image: Any 8-bit PIL image.

    Returns:
        ``(height, width, channels)`` floats on a 0 to 1 scale.
    """
    values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    width, height = image.size
    return values.view(height, width, len(image.getbands())).float().div(255.0)


def _picture(plane: torch.Tensor, mode: str) -> Image.Image:
    """Write a float tensor back into a PIL image.

    Args:
        plane: ``(height, width, channels)`` or ``(height, width)`` floats on a 0 to 1 scale.
        mode: PIL mode matching the channel count.

    Returns:
        An 8-bit PIL image. Values outside 0 to 1 are clipped.
    """
    data = plane.mul(255.0).round().clamp(0.0, 255.0).to(torch.uint8).cpu().reshape(-1)
    buffer = bytearray(data.numel())
    torch.frombuffer(buffer, dtype=torch.uint8).copy_(data)
    size = (int(plane.shape[1]), int(plane.shape[0]))
    return Image.frombytes(mode, size, buffer)


def _grey(plane: torch.Tensor) -> torch.Tensor:
    """The brightness a greyscale conversion reads off a picture.

    Args:
        plane: ``(height, width, channels)`` colours. A plane with fewer than three
            channels answers its first channel.

    Returns:
        ``(height, width)`` brightness on a 0 to 1 scale.
    """
    if plane.shape[-1] < 3:
        return plane[..., 0]
    weights = plane.new_tensor(_GREY).div(65536.0)
    return (plane[..., :3] * weights).sum(dim=-1)


def _grey8(plane: torch.Tensor) -> torch.Tensor:
    """The brightness an 8-bit greyscale conversion reads off a picture.

    Args:
        plane: ``(height, width, channels)`` colours with at least three channels.

    Returns:
        ``(height, width)`` brightness on a 0 to 255 scale, rounded to whole steps.
    """
    weights = plane.new_tensor(_GREY)
    totals = ((plane[..., :3] * 255.0).round() * weights).sum(dim=-1)
    return torch.floor((totals + 32768.0) / 65536.0)


def _held(plane: torch.Tensor, reach: int, axis: int) -> torch.Tensor:
    """Extend a plane along one axis by repeating its edge pixel.

    Args:
        plane: The plane to extend.
        reach: Pixels added at each end.
        axis: Axis to extend along.

    Returns:
        The plane, longer by ``2 * reach`` along ``axis``.
    """
    sizes = [1] * plane.ndim
    sizes[axis] = reach
    lead = plane.narrow(axis, 0, 1).repeat(sizes)
    tail = plane.narrow(axis, plane.shape[axis] - 1, 1).repeat(sizes)
    return torch.cat((lead, plane, tail), dim=axis)


def _box_radius(radius: float) -> float:
    """The box radius three passes carry a gaussian radius as.

    Args:
        radius: Gaussian blur radius in pixels.

    Returns:
        The radius of each box pass, whole part plus the weight of its two outer taps.
    """
    sigma = radius * radius / _PASSES
    span = math.sqrt(12.0 * sigma + 1.0)
    whole = math.floor((span - 1.0) / 2.0)
    part = (2 * whole + 1) * (whole * (whole + 1) - 3 * sigma)
    return whole + part / (6 * (sigma - (whole + 1) ** 2))


def _box_pass(plane: torch.Tensor, radius: float, axis: int) -> torch.Tensor:
    """One box blur pass along an axis.

    Args:
        plane: The plane to blur.
        radius: Box radius. The fraction beyond its whole part weights the two outer taps.
        axis: Axis to blur along.

    Returns:
        The blurred plane, the shape that went in.
    """
    span = int(radius)
    padded = _held(plane, span + 1, axis)
    running = padded.cumsum(axis)
    running = torch.cat((torch.zeros_like(running.narrow(axis, 0, 1)), running), dim=axis)
    count = plane.shape[axis]
    inner = running.narrow(axis, span * 2 + 2, count) - running.narrow(axis, 1, count)
    outer = padded.narrow(axis, 0, count) + padded.narrow(axis, span * 2 + 2, count)
    return (inner + outer * (radius - span)) / (2.0 * radius + 1.0)


def _blurred(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Blur a plane by a gaussian radius.

    Args:
        plane: The plane to blur, with the height on axis 0 and the width on axis 1.
        radius: Radius in pixels. A negative radius blurs by its magnitude, 0 does nothing.

    Returns:
        The blurred plane, the shape that went in.
    """
    if not radius:
        return plane
    box = _box_radius(abs(float(radius)))
    for axis in (1, 0):
        for _ in range(_PASSES):
            plane = _box_pass(plane, box, axis)
    return plane


def _smoothed(plane: torch.Tensor) -> torch.Tensor:
    """Smooth a plane with a 3x3 kernel, holding its border.

    Args:
        plane: ``(height, width, channels)`` colours.

    Returns:
        The smoothed plane, its outermost row and column carried over untouched.
    """
    height, width, channels = plane.shape
    if height < 3 or width < 3:
        return plane
    weight = plane.new_tensor(_SMOOTH).view(1, 1, 3, 3).div(13.0).repeat(channels, 1, 1, 1)
    stack = plane.permute(2, 0, 1).unsqueeze(0)
    inner = functional.conv2d(stack, weight, groups=channels)
    smoothed = plane.clone()
    smoothed[1:-1, 1:-1] = inner.squeeze(0).permute(1, 2, 0)
    return smoothed


def _towards(plane: torch.Tensor, degenerate: torch.Tensor, factor: float,
             ceiling: float) -> torch.Tensor:
    """Interpolate a plane away from a degenerate copy of itself.

    Args:
        plane: The plane being enhanced.
        degenerate: The plane the factor measures distance from.
        factor: 0 answers ``degenerate``, 1 answers ``plane``, more overshoots.
        ceiling: The largest value the result may hold.

    Returns:
        The enhanced plane, inside 0 to ``ceiling``.
    """
    return (degenerate + (plane - degenerate) * factor).clamp(0.0, ceiling)


def _contrasted(plane: torch.Tensor, factor: float, ceiling: float) -> torch.Tensor:
    """Enhance contrast, measuring from the plane's mean grey.

    Args:
        plane: ``(height, width, channels)`` colours.
        factor: Enhancement factor.
        ceiling: The largest value the result may hold.

    Returns:
        The enhanced plane.
    """
    return _towards(plane, _grey(plane).mean(), factor, ceiling)


def _saturated(plane: torch.Tensor, factor: float, ceiling: float) -> torch.Tensor:
    """Enhance colour, measuring from a greyscale copy of the plane.

    Args:
        plane: ``(height, width, channels)`` colours. A plane with fewer than three
            channels comes back unchanged.
        factor: Enhancement factor.
        ceiling: The largest value the result may hold.

    Returns:
        The enhanced plane.
    """
    if plane.shape[-1] < 3:
        return plane
    return _towards(plane, _grey(plane).unsqueeze(-1), factor, ceiling)


def _sharpened(plane: torch.Tensor, factor: float, ceiling: float) -> torch.Tensor:
    """Enhance sharpness, measuring from a smoothed copy of the plane.

    Args:
        plane: ``(height, width, channels)`` colours.
        factor: Enhancement factor.
        ceiling: The largest value the result may hold.

    Returns:
        The enhanced plane.
    """
    return _towards(plane, _smoothed(plane), factor, ceiling)


def _brightened(plane: torch.Tensor, factor: float, ceiling: float) -> torch.Tensor:
    """Enhance brightness, measuring from black.

    Args:
        plane: ``(height, width, channels)`` colours.
        factor: Enhancement factor.
        ceiling: The largest value the result may hold.

    Returns:
        The enhanced plane.
    """
    return (plane * factor).clamp(0.0, ceiling)


def _blended(backdrop: torch.Tensor, source: torch.Tensor, mode: str, ceiling: float,
             backdrop_alpha=None, source_alpha=None) -> torch.Tensor:
    """Blend two colour planes and carry the result through their coverage.

    Args:
        backdrop: ``(height, width, 3)`` colours already in place.
        source: ``(height, width, 3)`` colours going over them.
        mode: A blend mode name from :mod:`.blend_modes`.
        ceiling: The largest value the result may hold.
        backdrop_alpha: The backdrop's coverage on a 0 to 1 scale, or None for an opaque
            backdrop.
        source_alpha: The source's coverage, or None for an opaque source.

    Returns:
        ``(height, width, 3)`` colours, with no coverage of their own.
    """
    mixed = blend(backdrop, source, mode)
    if backdrop_alpha is None and source_alpha is None:
        return mixed
    under = 1.0 if backdrop_alpha is None else backdrop_alpha
    over = 1.0 if source_alpha is None else source_alpha
    both = under * over
    mixed = source * (over - both) + mixed * both + backdrop * (under - both)
    return mixed.clamp(0.0, ceiling)


def _composited(backdrop: torch.Tensor, backdrop_alpha: torch.Tensor, source: torch.Tensor,
                source_alpha) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay a source over a backdrop, both carrying coverage.

    Args:
        backdrop: ``(height, width, 3)`` colours underneath.
        backdrop_alpha: ``(height, width, 1)`` coverage on a 0 to 1 scale.
        source: ``(height, width, 3)`` colours on top.
        source_alpha: The source's coverage, a tensor or a number.

    Returns:
        ``(colours, alpha)``, the colours not premultiplied.
    """
    alpha = source_alpha + backdrop_alpha * (1.0 - source_alpha)
    weighted = source * source_alpha + backdrop * backdrop_alpha * (1.0 - source_alpha)
    return weighted / alpha.clamp(min=1e-6), alpha


def _glitter(size: tuple[int, int]) -> torch.Tensor:
    """Draw a layer of randomly coloured points and blur it.

    Args:
        size: ``(width, height)`` of the layer.

    Returns:
        ``(height, width, 3)`` colours on a 0 to 1 scale, on black.
    """
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    width, height = size
    # Both ranges reach one past the last pixel; PIL discards the points that fall outside.
    for _ in range(_PARTICLES):
        point = (random.randint(0, width), random.randint(0, height))
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        draw.point(point, fill=colour)
    return _blurred(_planes(canvas)[..., :3], 1)


def shadows_and_highlights(image: Image.Image, shadow_thresh: float = 30,
                           highlight_thresh: float = 220, shadow_factor: float = 0.5,
                           highlight_factor: float = 1.5, shadow_smooth: float | None = None,
                           highlight_smooth: float | None = None,
                           simplify_masks: float | None = None) -> tuple:
    """Darken shadows and lift highlights independently, returning both masks.

    Args:
        image: Source image. An alpha channel is detached first and reattached to the
            result; the adjustment itself runs on ``RGB``.
        shadow_thresh: Luminance below which a pixel counts as shadow, on a 0 to 255 scale.
        highlight_thresh: Luminance above which a pixel counts as highlight, same scale.
        shadow_factor: Multiplier applied to every channel of the shadow copy.
        highlight_factor: Multiplier applied to every channel of the highlight copy.
        shadow_smooth: Gaussian radius for the shadow mask, or ``None`` to leave it hard.
            The blur runs twice.
        highlight_smooth: Gaussian radius for the highlight mask, or ``None``.
        simplify_masks: Gaussian radius applied to the luminance before either mask is
            cut, which merges specks into larger regions. It is read only when one of the
            two mask radii is given.

    Returns:
        ``(result, shadow_mask, highlight_mask)``. The masks are mode ``L`` at the size of
        the source.
    """
    alpha = _planes(image.getchannel('A')) if image.mode.endswith('A') else None
    plane = _planes(image if image.mode == 'RGB' else image.convert('RGB'))
    ceiling = ceiling_of(plane)

    grey = _grey8(plane)
    if simplify_masks is not None and (shadow_smooth is not None or highlight_smooth is not None):
        # A blurred luminance is read on the same whole steps as an unblurred one.
        grey = _blurred(grey, float(simplify_masks)).round()

    shadow_mask = (grey < shadow_thresh).to(plane.dtype)
    highlight_mask = (grey > highlight_thresh).to(plane.dtype)
    for _ in range(2):
        if shadow_smooth is not None:
            shadow_mask = _blurred(shadow_mask, shadow_smooth)
        if highlight_smooth is not None:
            highlight_mask = _blurred(highlight_mask, highlight_smooth)

    shadow = (plane * shadow_factor).clamp(0.0, ceiling)
    highlight = (plane * highlight_factor).clamp(0.0, ceiling)
    result = plane + (shadow - plane) * shadow_mask.unsqueeze(-1)
    result = result + (highlight - result) * highlight_mask.unsqueeze(-1)
    result = blend(result, plane, "color")

    if alpha is not None:
        result = torch.cat((result, alpha), dim=-1)
    return (
        _picture(result, 'RGBA' if alpha is not None else 'RGB'),
        _picture(shadow_mask, 'L'),
        _picture(highlight_mask, 'L'),
    )


def dragan_filter(image: Image.Image, saturation: float = 1, contrast: float = 1,
                  sharpness: float = 1, brightness: float = 1, highpass_radius: float = 3,
                  highpass_samples: int = 1, highpass_strength: float = 1,
                  colorize: bool = True) -> Image.Image:
    """Apply the high-contrast, heavily textured Dragan portrait look.

    Args:
        image: Source image. An alpha channel is detached first and reattached to the
            result; the filter itself runs on ``RGB``.
        saturation: Colour enhancement factor, applied to the recoloured result. 0.0
            drains the colour, 1.0 leaves it, 2.0 doubles it. Needs ``colorize``.
        contrast: Contrast enhancement factor.
        sharpness: Sharpness enhancement factor.
        brightness: Brightness enhancement factor.
        highpass_radius: Blur radius used to build the high-pass layer.
        highpass_samples: Extra overlay passes of the high-pass layer. Values below one
            are treated as one.
        highpass_strength: Blend weight between the enhanced image and the overlaid one.
        colorize: Keep the high-pass layer in colour and recolour the result from the
            source.

    Returns:
        An ``RGB`` image, or ``RGBA`` when the source had alpha.
    """
    alpha = _planes(image.getchannel('A')) if image.mode.endswith('A') else None
    plane = _planes(image if image.mode == 'RGB' else image.convert('RGB'))
    ceiling = ceiling_of(plane)

    grey = _grey(plane).unsqueeze(-1)
    grey = _contrasted(grey, contrast, ceiling)
    grey = _sharpened(grey, sharpness, ceiling)
    grey = _brightened(grey, brightness, ceiling)
    enhanced = grey.expand(-1, -1, 3)

    detail = (plane - _blurred(grey, highpass_radius)).clamp(0.0, ceiling)
    highpass = blend(torch.full_like(plane, _MID_GREY), detail, "screen")
    if not colorize:
        highpass = _grey(highpass).unsqueeze(-1).expand(-1, -1, 3)

    overlaid = blend(enhanced, highpass, "overlay")
    for _ in range(max(1, highpass_samples)):
        overlaid = blend(overlaid, highpass, "overlay")

    result = (enhanced + (overlaid - enhanced) * highpass_strength).clamp(0.0, ceiling)
    if colorize:
        # Saturation acts here, on the recoloured result. The pass above it is a single
        # brightness plane, which is its own greyscale and so cannot be saturated.
        result = _saturated(blend(result, plane, "color"), saturation, ceiling)

    if alpha is not None:
        result = torch.cat((result, alpha), dim=-1)
    return _picture(result, 'RGBA' if alpha is not None else 'RGB')


def sparkle(image: Image.Image) -> Image.Image:
    """Add bloom and two layers of coloured glitter.

    Args:
        image: Source image, read as ``RGBA``.

    Returns:
        An ``RGB`` image the same size as the source.
    """
    plane = _planes(image if image.mode == 'RGBA' else image.convert('RGBA'))
    colour, alpha = plane[..., :3], plane[..., 3:]
    ceiling = ceiling_of(colour)

    colour = _contrasted(colour, 1.25, ceiling)
    colour = _saturated(colour, 1.5, ceiling)

    bloom = _brightened(_blurred(colour, 20), 1.2, ceiling)
    colour, alpha = _composited(colour, alpha, bloom, _LAYER_ALPHA)

    colour = _blended(
        colour, _glitter(image.size), "color-dodge", ceiling, alpha, _LAYER_ALPHA,
    )
    colour = _blended(
        colour, _glitter(image.size), "lighten", ceiling, None, _LAYER_ALPHA,
    )
    return _picture(colour, 'RGB')
