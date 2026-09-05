"""Closed-form alpha matting, solved in torch on whatever device the input is on.

Frames are ``(height, width, 3)`` and mattes ``(height, width)``, float on a 0 to 1 scale.
:func:`trimap` marks the band, :func:`alpha` solves it, :func:`foreground` unmixes it.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

__all__ = [
    "BACKGROUND",
    "BACKGROUND_LEVEL",
    "CHUNK",
    "EPSILON",
    "FOREGROUND",
    "FOREGROUND_LEVEL",
    "GRADIENT_WEIGHT",
    "LARGE_SWEEPS",
    "MAX_ITERATIONS",
    "NEIGHBOURS",
    "RADIUS",
    "REGULARIZATION",
    "SMALL_SIZE",
    "SMALL_SWEEPS",
    "SOLVE_DEVICES",
    "TOLERANCE",
    "UNKNOWN",
    "alpha",
    "foreground",
    "trimap",
]

#: Half the window the Laplacian is built over, so nine pixels around each one.
RADIUS = 1

#: Added to the diagonal of a window's colour covariance before it is inverted.
EPSILON = 1e-7

#: Relative residual the conjugate gradient solve stops at, and the iterations it may take
#: to reach it.
TOLERANCE = 1e-7
MAX_ITERATIONS = 10000

#: The three values a trimap holds.
BACKGROUND = 0.0
UNKNOWN = 128 / 255
FOREGROUND = 1.0

#: Trimap value from which a pixel counts as known, either way.
FOREGROUND_LEVEL = 0.9
BACKGROUND_LEVEL = 0.1

#: Colour estimate: the constant added to every neighbour weight, and the weight the alpha
#: gradient carries on top of it.
REGULARIZATION = 1e-5
GRADIENT_WEIGHT = 1.0

#: Side below which a level counts as small, and the sweeps run over a small level and over
#: a large one.
SMALL_SIZE = 32
SMALL_SWEEPS = 10
LARGE_SWEEPS = 2

#: Offsets of the four neighbours a colour estimate reads, as ``(row, column)``.
NEIGHBOURS = ((0, -1), (0, 1), (-1, 0), (1, 0))

#: Windows, and Laplacian rows, handled in one pass.
CHUNK = 1 << 15

#: Device types the sparse solve runs on. It runs on the CPU for anything else.
SOLVE_DEVICES = frozenset({"cpu", "cuda"})


def trimap(
    mask,
    foreground_threshold: float = 240 / 255,
    background_threshold: float = 10 / 255,
    erode_size: int = 10,
):
    """Mark every pixel of a mask as foreground, background or unknown.

    Args:
        mask: ``(height, width)`` tensor on a 0 to 1 scale.
        foreground_threshold: Level above which a pixel is foreground, on a 0 to 1 scale.
        background_threshold: Level below which a pixel is background, on a 0 to 1 scale.
        erode_size: Side of the square both regions are eroded by, 0 to erode neither.

    Returns:
        A ``(height, width)`` float tensor holding :data:`BACKGROUND`, :data:`UNKNOWN` or
        :data:`FOREGROUND`, on the device of ``mask``.
    """
    values = mask.to(torch.float32)
    is_foreground = _erode(values > foreground_threshold, erode_size, False)
    is_background = _erode(values < background_threshold, erode_size, True)
    marked = torch.full_like(values, UNKNOWN)
    marked[is_foreground] = FOREGROUND
    marked[is_background] = BACKGROUND
    return marked


def alpha(image, trimap):
    """Solve the matte the trimap constrains, over 3x3 windows of the frame.

    Args:
        image: ``(height, width, 3)`` tensor on a 0 to 1 scale.
        trimap: ``(height, width)`` tensor, as :func:`trimap` answers it.

    Returns:
        A ``(height, width)`` tensor on a 0 to 1 scale, on the device and in the dtype of
        ``image``.
    """
    device = image.device
    solve_on = device if device.type in SOLVE_DEVICES else torch.device("cpu")
    frame = image.to(solve_on, torch.float64)
    marked = trimap.to(solve_on, torch.float64).reshape(-1)
    is_foreground = marked >= FOREGROUND_LEVEL
    is_known = is_foreground | (marked <= BACKGROUND_LEVEL)
    unknown = (~is_known).nonzero().reshape(-1)
    matte = marked.clone()
    if unknown.numel():
        slot = _slots(unknown, marked.numel())
        packed = _rows(frame, is_known, slot, unknown.numel())
        matrix, rhs, diagonal = _system(
            packed, unknown, slot, is_foreground, frame.shape[0], frame.shape[1]
        )
        matte[unknown] = _solve(matrix, rhs, diagonal)
    return matte.reshape(trimap.shape).clamp(0, 1).to(device, image.dtype)


def foreground(image, alpha):
    """Estimate the foreground colour under a matte, one level of detail at a time.

    Args:
        image: ``(height, width, 3)`` tensor on a 0 to 1 scale.
        alpha: ``(height, width)`` matte on a 0 to 1 scale.

    Returns:
        A ``(height, width, 3)`` tensor on a 0 to 1 scale, in the dtype of ``image``.
    """
    frame = image.to(torch.float32)
    matte = alpha.to(device=frame.device, dtype=torch.float32)
    height, width, _ = frame.shape
    front = _mean_colour(frame, matte > FOREGROUND_LEVEL)
    back = _mean_colour(frame, matte < BACKGROUND_LEVEL)
    levels = max(1, math.ceil(math.log2(max(width, height))))
    for level in range(levels + 1):
        wide = round(width ** (level / levels))
        tall = round(height ** (level / levels))
        small = _resize(frame, tall, wide)
        band = _resize(matte, tall, wide)
        front = _resize(front, tall, wide)
        back = _resize(back, tall, wide)
        small_level = wide <= SMALL_SIZE and tall <= SMALL_SIZE
        for _ in range(SMALL_SWEEPS if small_level else LARGE_SWEEPS):
            front, back = _sweep(small, band, front, back)
    return front.to(image.dtype)


def _erode(flags, size: int, border: bool):
    """Shrink a boolean plane by a square, as ``scipy.ndimage.binary_erosion`` does.

    Args:
        flags: ``(height, width)`` boolean tensor.
        size: Side of the structuring element, 0 to shrink nothing.
        border: What lies outside the plane.

    Returns:
        A ``(height, width)`` boolean tensor.
    """
    if size <= 0:
        return flags
    values = flags.to(torch.float32)[None, None]
    outside = float(border)
    # An even side puts the origin left of centre.
    before, after = size // 2, size - 1 - size // 2
    padded = functional.pad(values, (before, after, 0, 0), value=outside)
    eroded = -functional.max_pool2d(-padded, (1, size), stride=1)
    padded = functional.pad(eroded, (0, 0, before, after), value=outside)
    eroded = -functional.max_pool2d(-padded, (size, 1), stride=1)
    return eroded[0, 0] > 0.5


def _slots(unknown, total: int):
    """Map each pixel to its row of the system, and each known pixel past the last row.

    Args:
        unknown: Flat index of every unknown pixel.
        total: How many pixels the frame holds.

    Returns:
        A ``(total,)`` tensor of row indices.
    """
    count = unknown.numel()
    slot = torch.full((total,), count, dtype=torch.long, device=unknown.device)
    slot[unknown] = torch.arange(count, device=unknown.device)
    return slot


def _centres(is_known, height: int, width: int):
    """Flat index of every window carrying at least one unknown pixel.

    Args:
        is_known: ``(height * width,)`` boolean tensor.
        height: Frame height.
        width: Frame width.

    Returns:
        A one-dimensional tensor of pixel indices, each the centre of a window.
    """
    size = 2 * RADIUS + 1
    if height < size or width < size:
        return torch.empty(0, dtype=torch.long, device=is_known.device)
    known = is_known.reshape(1, 1, height, width).to(torch.float32)
    covered = -functional.max_pool2d(-known, size, stride=1)
    inside = (covered.reshape(-1) < 0.5).nonzero().reshape(-1)
    span = width - 2 * RADIUS
    return (inside // span + RADIUS) * width + (inside % span) + RADIUS


def _rows(image, is_known, slot, count: int):
    """Accumulate the Laplacian entries of every unknown pixel, window by window.

    Args:
        image: ``(height, width, 3)`` float64 tensor.
        is_known: ``(height * width,)`` boolean tensor.
        slot: Row of each pixel, as :func:`_slots` answers it.
        count: How many rows there are.

    Returns:
        A ``(count, (4 * RADIUS + 1) ** 2)`` tensor, each row holding one pixel's entries
        by the offset of the column pixel from it.
    """
    height, width, _ = image.shape
    size = 2 * RADIUS + 1
    area = size * size
    span = 4 * RADIUS + 1
    steps = torch.arange(-RADIUS, RADIUS + 1, device=image.device)
    down, across = torch.meshgrid(steps, steps, indexing="ij")
    down, across = down.reshape(-1), across.reshape(-1)
    member = down * width + across
    block = (down[None, :] - down[:, None] + 2 * RADIUS) * span + (
        across[None, :] - across[:, None] + 2 * RADIUS
    )
    identity = torch.eye(area, dtype=image.dtype, device=image.device)
    values = torch.zeros((count + 1) * span * span, dtype=image.dtype, device=image.device)
    flat = image.reshape(-1, 3)
    for centres in _centres(is_known, height, width).split(CHUNK):
        pixels = centres[:, None] + member[None, :]
        colours = flat[pixels]
        centred = colours - colours.mean(dim=1, keepdim=True)
        inverse = _inverse_covariance(centred, area)
        weights = centred @ inverse @ centred.transpose(1, 2)
        entries = identity - (1.0 + weights) / area
        target = slot[pixels][:, :, None] * (span * span) + block[None, :, :]
        values.index_add_(0, target.reshape(-1), entries.reshape(-1))
    return values.reshape(count + 1, span * span)[:count]


def _inverse_covariance(centred, area: int):
    """Invert the regularised colour covariance of every window.

    Args:
        centred: ``(windows, pixels, 3)`` tensor of window colours, less the window mean.
        area: How many pixels one window holds.

    Returns:
        A ``(windows, 3, 3)`` tensor.
    """
    covariance = centred.transpose(1, 2) @ centred
    a00 = (covariance[:, 0, 0] + EPSILON) / area
    a01 = covariance[:, 0, 1] / area
    a02 = covariance[:, 0, 2] / area
    a11 = (covariance[:, 1, 1] + EPSILON) / area
    a12 = covariance[:, 1, 2] / area
    a22 = (covariance[:, 2, 2] + EPSILON) / area
    determinant = (
        a00 * a12 * a12
        + a01 * a01 * a22
        + a02 * a02 * a11
        - a00 * a11 * a22
        - 2 * a01 * a02 * a12
    )
    scale = 1.0 / determinant
    m00 = (a12 * a12 - a11 * a22) * scale
    m01 = (a01 * a22 - a02 * a12) * scale
    m02 = (a02 * a11 - a01 * a12) * scale
    m11 = (a02 * a02 - a00 * a22) * scale
    m12 = (a00 * a12 - a01 * a02) * scale
    m22 = (a01 * a01 - a00 * a11) * scale
    entries = [m00, m01, m02, m01, m11, m12, m02, m12, m22]
    return torch.stack(entries, dim=-1).reshape(-1, 3, 3)


def _system(packed, unknown, slot, is_foreground, height: int, width: int):
    """Split the Laplacian rows into the unknown block and the vector it is solved against.

    Args:
        packed: Laplacian entries, as :func:`_rows` answers them.
        unknown: Flat index of every unknown pixel.
        slot: Row of each pixel, as :func:`_slots` answers it.
        is_foreground: ``(height * width,)`` boolean tensor.
        height: Frame height.
        width: Frame width.

    Returns:
        ``(matrix, rhs, diagonal)``: the unknown block as a sparse CSR tensor, the vector
        the solve runs against, and the diagonal of the block.
    """
    count = unknown.numel()
    span = 4 * RADIUS + 1
    device = packed.device
    steps = torch.arange(-2 * RADIUS, 2 * RADIUS + 1, device=device)
    down, across = torch.meshgrid(steps, steps, indexing="ij")
    down, across = down.reshape(-1), across.reshape(-1)
    indices, entries = [], []
    rhs = torch.zeros(count, dtype=packed.dtype, device=device)
    for start in range(0, count, CHUNK):
        part = unknown[start : start + CHUNK]
        values = packed[start : start + CHUNK]
        rows = part[:, None] // width + down[None, :]
        columns = part[:, None] % width + across[None, :]
        keep = (
            (rows >= 0)
            & (rows < height)
            & (columns >= 0)
            & (columns < width)
            & (values != 0)
        )
        row = torch.arange(start, start + part.numel(), device=device)
        row = row[:, None].expand_as(values)[keep]
        pixel = (rows * width + columns)[keep]
        entry = values[keep]
        column = slot[pixel]
        unsolved = column < count
        indices.append(torch.stack([row[unsolved], column[unsolved]]))
        entries.append(entry[unsolved])
        constrained = ~unsolved & is_foreground[pixel]
        rhs.index_add_(0, row[constrained], -entry[constrained])
    matrix = torch.sparse_coo_tensor(
        torch.cat(indices, dim=1), torch.cat(entries), (count, count)
    )
    diagonal = packed[:, 2 * RADIUS * span + 2 * RADIUS]
    return matrix.coalesce().to_sparse_csr(), rhs, diagonal


def _solve(matrix, rhs, diagonal):
    """Solve a symmetric system by conjugate gradients, preconditioned on the diagonal.

    Args:
        matrix: Sparse CSR tensor.
        rhs: The vector to solve against.
        diagonal: The diagonal of ``matrix``.

    Returns:
        A vector the length of ``rhs``.

    Raises:
        ArithmeticError: The residual did not fall to :data:`TOLERANCE` within
            :data:`MAX_ITERATIONS`, or the matrix stopped being positive definite.
    """
    solution = torch.zeros_like(rhs)
    target = rhs.norm()
    residual = rhs.clone()
    if residual.norm() <= TOLERANCE * target:
        return solution
    direction = residual.clone()
    projection = residual.dot(direction)
    for _ in range(MAX_ITERATIONS):
        applied = (matrix @ direction.unsqueeze(1)).reshape(-1)
        curvature = direction.dot(applied)
        if curvature <= 0:
            raise ArithmeticError(
                f"the matting solve met a curvature of {float(curvature):.3e}, so the "
                f"system is no longer positive definite and the matte it would answer is "
                f"not a solution. Widen the trimap's known regions and try again."
            )
        step = projection / curvature
        solution += step * direction
        residual -= step * applied
        if residual.norm() <= TOLERANCE * target:
            break
        preconditioned = residual
        previous = projection
        projection = residual.dot(preconditioned)
        direction = direction * (projection / previous) + preconditioned
    else:
        raise ArithmeticError(
            f"the matting solve did not settle within {MAX_ITERATIONS} steps: the residual "
            f"stands at {float(residual.norm() / target.clamp_min(1e-30)):.3e} against a "
            f"tolerance of {TOLERANCE:.0e}. A narrower unknown band settles sooner, so "
            f"widen the foreground and background thresholds, or matte a smaller frame."
        )
    return solution


def _mean_colour(frame, selected):
    """The mean colour over the selected pixels, as a ``(1, 1, 3)`` tensor.

    Args:
        frame: ``(height, width, 3)`` tensor.
        selected: ``(height, width)`` boolean tensor.

    Returns:
        A ``(1, 1, 3)`` tensor.
    """
    weights = selected.to(frame.dtype).unsqueeze(-1)
    return ((frame * weights).sum(dim=(0, 1)) / (weights.sum() + 1e-5)).reshape(1, 1, -1)


def _resize(values, height: int, width: int):
    """Resample the first two axes to ``(height, width)`` by nearest neighbour.

    Args:
        values: ``(rows, columns)`` or ``(rows, columns, channels)`` tensor.
        height: Rows wanted.
        width: Columns wanted.

    Returns:
        A tensor of the same rank, ``height`` by ``width`` over its first two axes.
    """
    rows = (torch.arange(height, device=values.device) * values.shape[0]) // height
    columns = (torch.arange(width, device=values.device) * values.shape[1]) // width
    return values[rows][:, columns]


def _shift(values, rows: int, columns: int):
    """The neighbour ``(rows, columns)`` away from every pixel, clamped at the edges.

    Args:
        values: ``(rows, columns)`` or ``(rows, columns, channels)`` tensor.
        rows: Offset down.
        columns: Offset across.

    Returns:
        A tensor of the same shape.
    """
    height, width = values.shape[0], values.shape[1]
    down = (torch.arange(height, device=values.device) + rows).clamp(0, height - 1)
    across = (torch.arange(width, device=values.device) + columns).clamp(0, width - 1)
    return values[down][:, across]


def _sweep(image, matte, front, back):
    """Refit both colours of every pixel, one half of the checkerboard at a time.

    Args:
        image: ``(height, width, 3)`` tensor for this level.
        matte: ``(height, width)`` matte for this level.
        front: ``(height, width, 3)`` foreground estimate.
        back: ``(height, width, 3)`` background estimate.

    Returns:
        ``(front, back)``, each ``(height, width, 3)`` on a 0 to 1 scale.
    """
    height, width = matte.shape
    rows = torch.arange(height, device=matte.device)[:, None]
    columns = torch.arange(width, device=matte.device)[None, :]
    half = ((rows + columns) % 2 == 0).unsqueeze(-1)
    for selected in (half, ~half):
        fitted_front, fitted_back = _fit(image, matte, front, back)
        front = torch.where(selected, fitted_front, front)
        back = torch.where(selected, fitted_back, back)
    return front, back


def _fit(image, matte, front, back):
    """Solve the two colours of every pixel against the estimate its neighbours hold.

    Args:
        image: ``(height, width, 3)`` tensor for this level.
        matte: ``(height, width)`` matte for this level.
        front: ``(height, width, 3)`` foreground estimate.
        back: ``(height, width, 3)`` background estimate.

    Returns:
        ``(front, back)``, each ``(height, width, 3)`` on a 0 to 1 scale.
    """
    weight = matte.unsqueeze(-1)
    rest = 1.0 - weight
    a00 = weight * weight
    a01 = weight * rest
    a11 = rest * rest
    b0 = weight * image
    b1 = rest * image
    for shift_row, shift_column in NEIGHBOURS:
        gradient = (weight - _shift(weight, shift_row, shift_column)).abs()
        step = REGULARIZATION + GRADIENT_WEIGHT * gradient
        a00 = a00 + step
        a11 = a11 + step
        b0 = b0 + step * _shift(front, shift_row, shift_column)
        b1 = b1 + step * _shift(back, shift_row, shift_column)
    determinant = a00 * a11 - a01 * a01
    return (
        ((a11 * b0 - a01 * b1) / determinant).clamp(0, 1),
        ((a00 * b1 - a01 * b0) / determinant).clamp(0, 1),
    )
