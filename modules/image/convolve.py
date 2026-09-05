"""Convolution, edge and morphology primitives evaluated in torch.

Tensors are ``(batch, channels, height, width)`` floats and padding reflects at the edges.
Kernel sizes are odd, and an even one is rounded down. Edge work reads a 0 to 255 scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "bilateral_blur",
    "canny",
    "dilate",
    "ellipse_kernel",
    "erode",
    "gaussian_blur",
    "gaussian_kernel",
    "gradients",
    "hysteresis",
    "luminance",
    "thin_edges",
]

#: Kernels substituted when a sigma is derived rather than given and the kernel is small.
#: Each is a whole number of 256ths and none is quite the Gaussian its width implies.
BINOMIAL = {
    1: (1.0,),
    3: (0.25, 0.5, 0.25),
    5: (0.0625, 0.25, 0.375, 0.25, 0.0625),
    7: (0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125),
    9: (0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375,
        0.19921875, 0.1171875, 0.05078125, 0.015625),
}

#: Rec. 601 luminance weights, in red, green, blue order.
GRAY_WEIGHTS = (0.299, 0.587, 0.114)

#: Horizontal 3x3 Sobel kernel. The vertical one is its transpose.
SOBEL = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))

#: Neighbour offsets as ``(row, column)`` pairs, one pair per 45 degree gradient sector,
#: from a horizontal gradient through to one 135 degrees around from it.
SECTORS = (
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 0), (1, 0)),
    ((-1, 1), (1, -1)),
)

#: Passes an edge may grow over before hysteresis stops, on top of the image diagonal.
GROWTH_MARGIN = 8


def luminance(x: torch.Tensor) -> torch.Tensor:
    """Reduce colour channels to a single luminance channel.

    Args:
        x: ``(batch, channels, height, width)`` tensor. One and two channel tensors are
            already luminance, and only their first channel is read.

    Returns:
        A ``(batch, 1, height, width)`` tensor on the scale of the source.
    """
    if x.shape[1] < 3:
        return x[:, :1]
    weights = torch.tensor(GRAY_WEIGHTS, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x[:, :3] * weights).sum(1, keepdim=True)


def gaussian_kernel(
    size: int, sigma: float = 0.0, dtype=torch.float32, device=None
) -> torch.Tensor:
    """Build a normalised one-dimensional Gaussian.

    Args:
        size: Odd kernel length in samples.
        sigma: Standard deviation in pixels. At or below 0 it is derived from ``size``.
        dtype: Element type of the result.
        device: Where the kernel is built.

    Returns:
        A ``(size,)`` tensor summing to 1.
    """
    size = _odd(size)
    if sigma <= 0:
        if size in BINOMIAL:
            return torch.tensor(BINOMIAL[size], dtype=dtype, device=device)
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    offsets = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    weights = torch.exp(-offsets * offsets / (2.0 * sigma * sigma))
    return weights / weights.sum()


def gaussian_blur(x: torch.Tensor, size: int = 0, sigma: float = 0.0) -> torch.Tensor:
    """Blur with a separable Gaussian.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        size: Odd kernel length. 0 derives one spanning four sigma, and a kernel wider
            than the image is trimmed to fit it.
        sigma: Standard deviation in pixels. 0 derives one from ``size``.

    Returns:
        A tensor of the same shape.

    Raises:
        ValueError: Neither ``size`` nor ``sigma`` was given.
    """
    if size <= 0 and sigma <= 0:
        raise ValueError("gaussian_blur needs a kernel size, a sigma, or both")
    if size <= 0:
        size = int(round(sigma * 4.0)) * 2 + 1
    size = _fit(size, x)
    if size < 3:
        return x
    kernel = gaussian_kernel(size, sigma, x.dtype, x.device)
    return _separable(_separable(x, kernel, True), kernel, False)


def gradients(
    x: torch.Tensor, l2: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take Sobel gradients and their magnitude.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        l2: ``True`` measures magnitude as the hypotenuse, ``False`` as the sum of the
            two absolute gradients.

    Returns:
        ``(magnitude, horizontal, vertical)``, each the shape of the source.
    """
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = _taps(padded, SOBEL, x.shape[-2:])
    gy = _taps(padded, tuple(zip(*SOBEL)), x.shape[-2:])
    magnitude = torch.sqrt(gx * gx + gy * gy) if l2 else gx.abs() + gy.abs()
    return magnitude, gx, gy


def thin_edges(
    magnitude: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor
) -> torch.Tensor:
    """Suppress every gradient that is not a peak along its own direction.

    Args:
        magnitude: Gradient strengths, ``(batch, channels, height, width)``.
        gx: Horizontal gradients, the shape of ``magnitude``.
        gy: Vertical gradients, the shape of ``magnitude``.

    Returns:
        ``magnitude`` with everything off a ridge set to zero.
    """
    angle = torch.rad2deg(torch.atan2(gy, gx)) % 180.0
    sector = torch.div(angle + 22.5, 45.0, rounding_mode="floor").long() % 4
    padded = F.pad(magnitude, (1, 1, 1, 1))
    keep = torch.zeros_like(magnitude, dtype=torch.bool)
    for index, (first, second) in enumerate(SECTORS):
        # One side compares strictly, so a ridge two pixels wide keeps one of them.
        peak = (magnitude > _shift(padded, *first)) & (magnitude >= _shift(padded, *second))
        keep |= (sector == index) & peak
    keep[..., 0, :] = keep[..., -1, :] = keep[..., :, 0] = keep[..., :, -1] = False
    return magnitude * keep


def hysteresis(magnitude: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Keep the gradients above ``high``, and those above ``low`` that touch them.

    Args:
        magnitude: Gradient strengths, ``(batch, channels, height, width)``.
        low: Strength a gradient has to reach to continue an edge, as a number or as a
            tensor broadcasting over ``magnitude`` to set it per image.
        high: Strength a gradient has to reach to start one, in the same two forms. At or
            below ``low`` every gradient above ``low`` starts one.

    Returns:
        A boolean tensor the shape of ``magnitude``.
    """
    weak = magnitude >= low
    edges = (magnitude >= high) & weak
    limit = int(math.hypot(magnitude.shape[-2], magnitude.shape[-1])) + GROWTH_MARGIN
    for _ in range(limit):
        grown = (F.max_pool2d(edges.to(magnitude.dtype), 3, stride=1, padding=1) > 0) & weak
        if torch.equal(grown, edges):
            break
        edges = grown
    return edges


def canny(gray: torch.Tensor, low: float, high: float, l2: bool = False) -> torch.Tensor:
    """Trace the edges of a luminance image and thin them to single-pixel lines.

    Args:
        gray: ``(batch, 1, height, width)`` tensor on a 0 to 255 scale, blurred already.
        low: Gradient strength an edge continues at.
        high: Gradient strength an edge starts at.
        l2: ``True`` measures gradient magnitude as the hypotenuse.

    Returns:
        A boolean tensor the shape of the source, true on an edge.
    """
    magnitude, gx, gy = gradients(gray, l2)
    return hysteresis(thin_edges(magnitude, gx, gy), low, high)


def ellipse_kernel(radius: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """Build a filled ellipse to dilate or erode through.

    Args:
        radius: Distance in pixels from the centre to the edge of the ellipse.
        dtype: Element type of the result.
        device: Where the kernel is built.

    Returns:
        A ``(2 * radius + 1, 2 * radius + 1)`` tensor holding 1 inside the ellipse.
    """
    radius = max(int(radius), 0)
    size = radius * 2 + 1
    kernel = torch.zeros((size, size), dtype=dtype, device=device)
    inverse = 1.0 / (radius * radius) if radius else 0.0
    for row in range(size):
        dy = row - radius
        reach = max(0.0, (radius * radius - dy * dy) * inverse)
        span = int(round(radius * math.sqrt(reach)))
        kernel[row, max(radius - span, 0):min(radius + span + 1, size)] = 1.0
    return kernel


def dilate(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Spread each value out to the brightest one under the kernel.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        kernel: Structuring element, non-zero where a neighbour counts.

    Returns:
        A tensor of the same shape.
    """
    return _morph(x, kernel, torch.maximum)


def erode(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Pull each value back to the darkest one under the kernel.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        kernel: Structuring element, non-zero where a neighbour counts.

    Returns:
        A tensor of the same shape.
    """
    return _morph(x, kernel, torch.minimum)


def bilateral_blur(
    x: torch.Tensor, diameter: int, sigma_color: float, sigma_space: float
) -> torch.Tensor:
    """Blur within areas of similar colour, leaving the boundaries between them crisp.

    Args:
        x: ``(batch, channels, height, width)`` tensor on a 0 to 255 scale.
        diameter: Width in pixels of the area each pixel is mixed over. At or below 0 it
            is derived from ``sigma_space``.
        sigma_color: How far apart in colour two pixels may be and still mix, on the same
            0 to 255 scale.
        sigma_space: How far apart in pixels two pixels may be and still mix.

    Returns:
        A tensor of the same shape.
    """
    sigma_color = max(float(sigma_color), 1.0)
    sigma_space = max(float(sigma_space), 1.0)
    radius = int(diameter) // 2 if diameter > 0 else int(round(sigma_space * 1.5))
    radius = max(min(radius, min(x.shape[-2], x.shape[-1]) - 1), 1)

    height, width = x.shape[-2:]
    padded = F.pad(x, (radius,) * 4, mode="reflect")
    colour_scale = -0.5 / (sigma_color * sigma_color)
    space_scale = -0.5 / (sigma_space * sigma_space)

    total = torch.zeros_like(x)
    divisor = torch.zeros_like(x[:, :1])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            square = dy * dy + dx * dx
            if square > radius * radius:
                continue
            row, column = radius + dy, radius + dx
            shifted = padded[:, :, row:row + height, column:column + width]
            distance = (shifted - x).abs().sum(1, keepdim=True)
            nearness = torch.exp(colour_scale * distance * distance)
            weight = math.exp(space_scale * square) * nearness
            total += shifted * weight
            divisor += weight
    return total / divisor


def _separable(x: torch.Tensor, kernel: torch.Tensor, horizontal: bool) -> torch.Tensor:
    """Run one axis of a separable filter as a sum of shifted, weighted slices.

    Args:
        x: ``(batch, channels, height, width)`` tensor.
        kernel: One-dimensional weights, odd in length.
        horizontal: ``True`` filters along the width, ``False`` along the height.

    Returns:
        A tensor of the same shape.
    """
    # The slices are summed in kernel order rather than convolved, so the result of one
    # frame does not depend on how many frames came with it.
    size = kernel.numel()
    pad = size // 2
    height, width = x.shape[-2:]
    padding = (pad, pad, 0, 0) if horizontal else (0, 0, pad, pad)
    padded = F.pad(x, padding, mode="reflect")
    total = None
    for index in range(size):
        if horizontal:
            piece = padded[:, :, :, index:index + width]
        else:
            piece = padded[:, :, index:index + height, :]
        term = piece * kernel[index]
        total = term if total is None else total + term
    return total


def _taps(padded: torch.Tensor, kernel, size) -> torch.Tensor:
    """Correlate a padded tensor with a small kernel, one non-zero tap at a time.

    Args:
        padded: The tensor, already padded by half the kernel on each side.
        kernel: Rows of weights.
        size: ``(height, width)`` of the result.

    Returns:
        A tensor of that height and width.
    """
    height, width = size
    total = None
    for row, weights in enumerate(kernel):
        for column, weight in enumerate(weights):
            if not weight:
                continue
            term = padded[:, :, row:row + height, column:column + width] * weight
            total = term if total is None else total + term
    return total if total is not None else padded[:, :, :height, :width] * 0.0


def _morph(x: torch.Tensor, kernel: torch.Tensor, pick) -> torch.Tensor:
    """Take ``pick`` of every neighbour the kernel selects."""
    spans = _row_spans(kernel)
    if spans is None:
        return _morph_offsets(x, kernel, pick)
    height, width = x.shape[-2:]
    pad_y, pad_x = kernel.shape[-2] // 2, kernel.shape[-1] // 2
    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    sign = 1.0 if pick is torch.maximum else -1.0
    result = None
    for row, first, last in spans:
        span = last - first + 1
        band = padded[:, :, row:row + height, first:first + width + span - 1]
        pooled = sign * F.max_pool2d(sign * band, (1, span), stride=1)
        result = pooled if result is None else pick(result, pooled)
    return result


def _morph_offsets(x: torch.Tensor, kernel: torch.Tensor, pick) -> torch.Tensor:
    """Take ``pick`` one kernel entry at a time, for a kernel with a gap in a row."""
    offsets = torch.nonzero(kernel, as_tuple=False).tolist()
    if not offsets:
        return x
    height, width = x.shape[-2:]
    pad_y, pad_x = kernel.shape[-2] // 2, kernel.shape[-1] // 2
    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    result = None
    for row, column in offsets:
        window = padded[:, :, row:row + height, column:column + width]
        result = window if result is None else pick(result, window)
    return result


def _row_spans(kernel: torch.Tensor) -> list[tuple[int, int, int]] | None:
    """Each occupied kernel row as ``(row, first, last)``, or ``None`` if one has a gap."""
    spans = []
    for row in range(kernel.shape[-2]):
        columns = torch.nonzero(kernel[row], as_tuple=False).flatten().tolist()
        if not columns:
            continue
        if columns[-1] - columns[0] + 1 != len(columns):
            return None
        spans.append((row, columns[0], columns[-1]))
    return spans or None


def _shift(padded: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """The view of a one-pixel-padded tensor offset by ``(dy, dx)``."""
    height, width = padded.shape[-2] - 2, padded.shape[-1] - 2
    return padded[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


def _odd(size: int) -> int:
    """``size`` rounded down to an odd number, and never below 1."""
    size = int(size)
    return max(1, size - 1 if size % 2 == 0 else size)


def _fit(size: int, x: torch.Tensor) -> int:
    """``size`` trimmed to a kernel a reflecting pad of this tensor can carry."""
    return _odd(min(_odd(size), 2 * min(x.shape[-2], x.shape[-1]) - 1))
