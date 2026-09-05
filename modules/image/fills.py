"""Solid colours, gradients and masks, as torch planes.

A colour is three floats in 0 to 1, a plane is ``(..., 3)`` and a mask ``(height, width, 1)``.
``ceiling`` is the value white sits at.
"""

from __future__ import annotations

__all__ = [
    "GRADIENT_SIZE",
    "MIN_SPREAD",
    "blend_colour",
    "blend_fill",
    "blend_hue",
    "blend_image",
    "blend_opacity",
    "composite_mask",
    "fill",
    "linear_gradient",
    "linear_mask",
    "quarter_turn",
    "radial_gradient",
    "radial_mask",
    "rgb8",
]

import math
from collections.abc import Sequence

import torch

from .blend_modes import blend, ceiling_of

#: Steps in the fixed ramp a linear mask is cut from.
GRADIENT_SIZE = 256

#: Narrowest a radial ramp may be, so two stops at one position still divide.
MIN_SPREAD = 0.001


def rgb8(red: int, green: int, blue: int) -> tuple[float, float, float]:
    """A colour written in 0 to 255 steps, as floats in 0 to 1.

    Args:
        red: Red, 0 to 255.
        green: Green, 0 to 255.
        blue: Blue, 0 to 255.

    Returns:
        Three floats in 0 to 1.
    """
    return (red / 255.0, green / 255.0, blue / 255.0)


def fill(
    plane: torch.Tensor, colour: tuple[float, float, float], ceiling: float = 1.0
) -> torch.Tensor:
    """One solid colour, ready to mix with a plane.

    Args:
        plane: The plane it is built for, which sets the dtype and device.
        colour: Three floats in 0 to 1.
        ceiling: The value white sits at.

    Returns:
        A ``(1, 1, 3)`` tensor, which broadcasts over any picture.
    """
    return torch.tensor(
        [float(channel) * ceiling for channel in colour],
        dtype=plane.dtype,
        device=plane.device,
    ).view(1, 1, 3)


def blend_opacity(
    backdrop: torch.Tensor, source: torch.Tensor, opacity: float
) -> torch.Tensor:
    """Mix two planes by opacity.

    Args:
        backdrop: ``(..., 3)`` colours underneath.
        source: ``(..., 3)`` colours on top, at the backdrop's own scale.
        opacity: How much of the source lands, 0.0 to 1.0.

    Returns:
        A tensor of the two shapes broadcast together.
    """
    return backdrop + (source - backdrop) * opacity


def composite_mask(
    over: torch.Tensor, under: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Choose between two planes through a mask.

    Args:
        over: ``(..., 3)`` colours taken where the mask is 1.
        under: ``(..., 3)`` colours taken where the mask is 0.
        mask: ``(height, width, 1)`` weights in 0 to 1.

    Returns:
        A tensor of the three shapes broadcast together.
    """
    return under + (over - under) * mask


def blend_image(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    mode: str,
    ceiling: float | None = None,
) -> torch.Tensor:
    """Mix two planes through one blend mode, over the range in use.

    Args:
        backdrop: ``(..., 3)`` colours underneath.
        source: ``(..., 3)`` colours on top, at the backdrop's own scale.
        mode: One of :data:`modules.image.blend_modes.MODES`.
        ceiling: The value white sits at, or None to read it off the two planes.

    Returns:
        A tensor of the two shapes broadcast together.
    """
    top = ceiling_of(backdrop, source) if ceiling is None else ceiling
    if top == 1.0:
        mixed = blend(backdrop, source, mode)
    else:
        mixed = blend(backdrop / top, source / top, mode) * top
    # A mode that answers its source verbatim answers it at the source's own shape.
    shape = torch.broadcast_shapes(backdrop.shape, source.shape)
    return mixed if mixed.shape == shape else mixed.expand(shape)


def blend_fill(
    backdrop: torch.Tensor,
    mode: str,
    colour: tuple[float, float, float],
    alpha: float = 1.0,
    ceiling: float | None = None,
) -> torch.Tensor:
    """Blend one solid colour over a plane.

    Args:
        backdrop: ``(..., 3)`` colours underneath.
        mode: One of :data:`modules.image.blend_modes.MODES`.
        colour: Three floats in 0 to 1.
        alpha: How much of the blend lands, 0.0 to 1.0, on the 1 in 255 grid.
        ceiling: The value white sits at, or None to read it off the backdrop.

    Returns:
        A tensor shaped as the backdrop.
    """
    top = ceiling_of(backdrop) if ceiling is None else ceiling
    mixed = blend_image(backdrop, fill(backdrop, colour, top), mode, top)
    weight = round(alpha * 255.0) / 255.0
    if weight >= 1.0:
        return mixed
    return backdrop + (mixed - backdrop) * weight


def blend_colour(
    backdrop: torch.Tensor,
    colour: tuple[float, float, float],
    alpha: float = 1.0,
    ceiling: float | None = None,
) -> torch.Tensor:
    """Give a plane the hue and saturation of one solid colour.

    Args:
        backdrop: ``(..., 3)`` colours underneath.
        colour: Three floats in 0 to 1.
        alpha: How much of the blend lands, 0.0 to 1.0.
        ceiling: The value white sits at, or None to read it off the backdrop.

    Returns:
        A tensor shaped as the backdrop.
    """
    return blend_fill(backdrop, "color", colour, alpha, ceiling)


def blend_hue(
    backdrop: torch.Tensor,
    colour: tuple[float, float, float],
    alpha: float = 1.0,
    ceiling: float | None = None,
) -> torch.Tensor:
    """Give a plane the hue of one solid colour.

    Args:
        backdrop: ``(..., 3)`` colours underneath.
        colour: Three floats in 0 to 1.
        alpha: How much of the blend lands, 0.0 to 1.0.
        ceiling: The value white sits at, or None to read it off the backdrop.

    Returns:
        A tensor shaped as the backdrop.
    """
    return blend_fill(backdrop, "hue", colour, alpha, ceiling)


def radial_mask(
    plane: torch.Tensor,
    length: float = 0.0,
    scale: float = 1.0,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
) -> torch.Tensor:
    """A round ramp falling from a centre out to the farthest corner.

    Args:
        plane: The plane it is built for, which sets the size, dtype and device.
        length: How far out the ramp holds 1.0, as a fraction of the way to the corner.
        scale: How far out it reaches 0.0. 1.0 reaches the farthest corner.
        centre_x: Middle of the ramp across the frame, 0.0 to 1.0.
        centre_y: Middle of the ramp down the frame, 0.0 to 1.0.

    Returns:
        A ``(height, width, 1)`` tensor in 0 to 1, 1.0 at the centre.
    """
    height, width = int(plane.shape[-3]), int(plane.shape[-2])
    if length >= 1.0:
        return torch.ones((height, width, 1), dtype=plane.dtype, device=plane.device)
    if scale <= 0.0:
        return torch.zeros((height, width, 1), dtype=plane.dtype, device=plane.device)

    left, right = width * centre_x, width * (1.0 - centre_x)
    above, below = height * centre_y, height * (1.0 - centre_y)
    across = torch.linspace(-left, right, width, dtype=torch.float32, device=plane.device)
    down = torch.linspace(
        -above, below, height, dtype=torch.float32, device=plane.device
    ).unsqueeze(-1)

    reach = math.hypot(max(left, right), max(above, below))
    spread = max(scale - length, MIN_SPREAD)
    ramp = 1.0 - ((across * across + down * down).sqrt() / reach - length) / spread
    return ramp.clamp(0.0, 1.0).unsqueeze(-1).to(plane.dtype)


def linear_mask(
    plane: torch.Tensor,
    start: float = 0.0,
    stop: float = 1.0,
    horizontal: bool = True,
) -> torch.Tensor:
    """A straight ramp falling across or down the frame.

    Args:
        plane: The plane it is built for, which sets the size, dtype and device.
        start: How far down from white the ramp begins, 0.0 to 1.0. 0.8 begins at 0.2.
        stop: How far down it ends. 1.0 ends at black.
        horizontal: Whether the ramp runs across the frame rather than down it.

    Returns:
        A ``(height, width, 1)`` tensor in 0 to 1, highest at the left or the top.
    """
    height, width = int(plane.shape[-3]), int(plane.shape[-2])
    first = round(GRADIENT_SIZE * start)
    last = round(GRADIENT_SIZE * stop)
    count = width if horizontal else height

    steps = torch.arange(count, dtype=torch.float32, device=plane.device) + 0.5
    read = first + steps * (last - first) / count
    ramp = ((GRADIENT_SIZE - 0.5) - read) / (GRADIENT_SIZE - 1)
    ramp = ramp.clamp(0.0, 1.0).to(plane.dtype)

    shaped = ramp.view(1, count, 1) if horizontal else ramp.view(count, 1, 1)
    return shaped.expand(height, width, 1)


def quarter_turn(mask: torch.Tensor) -> torch.Tensor:
    """A mask turned a quarter anticlockwise on a canvas of the same size.

    Args:
        mask: A ``(height, width, 1)`` tensor.

    Returns:
        A ``(height, width, 1)`` tensor, 0.0 wherever the turn reaches off the canvas.
    """
    height, width = int(mask.shape[-3]), int(mask.shape[-2])
    rows = torch.arange(height, dtype=torch.float32, device=mask.device)
    columns = torch.arange(width, dtype=torch.float32, device=mask.device)
    across = torch.floor(-(rows + 0.5) + (height + width) / 2.0)
    down = torch.floor(columns + 0.5 + (height - width) / 2.0)

    taken = mask.index_select(
        0, down.clamp(0, height - 1).long()
    ).index_select(1, across.clamp(0, width - 1).long()).transpose(0, 1)
    reached = ((across >= 0) & (across < width)).view(height, 1, 1) & (
        (down >= 0) & (down < height)
    ).view(1, width, 1)
    return torch.where(reached, taken, torch.zeros_like(taken))


def radial_gradient(
    plane: torch.Tensor,
    colours: Sequence[tuple[float, float, float]],
    positions: Sequence[float] | None = None,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
    ceiling: float = 1.0,
) -> torch.Tensor:
    """A round ramp between two or more colour stops.

    Args:
        plane: The plane it is built for, which sets the size, dtype and device.
        colours: Two or more colours, each three floats in 0 to 1, centre outwards.
        positions: Where each stop sits, as a fraction of the way to the farthest
            corner, or None to space them evenly.
        centre_x: Middle of the ramp across the frame, 0.0 to 1.0.
        centre_y: Middle of the ramp down the frame, 0.0 to 1.0.
        ceiling: The value white sits at.

    Returns:
        A ``(height, width, 3)`` tensor.

    Raises:
        ValueError: Fewer than two colours, or a position for every colour is missing.
    """
    stops = [fill(plane, colour, ceiling) for colour in colours]
    if len(stops) < 2:
        raise ValueError(
            f"a radial gradient needs at least two colours, and {len(stops)} was given. "
            f"Add a second colour for the outer edge."
        )
    if positions is None:
        last = len(stops) - 1
        positions = [index / last for index in range(len(stops))]
    elif len(positions) != len(stops):
        raise ValueError(
            f"a radial gradient needs one position per colour, and {len(positions)} "
            f"positions were given for {len(stops)} colours. Give every colour a position "
            f"or none at all."
        )

    ramp = stops[0]
    for index in range(1, len(stops)):
        mask = radial_mask(
            plane, positions[index - 1], positions[index], centre_x, centre_y
        )
        ramp = composite_mask(ramp, stops[index], mask)
    return ramp


def linear_gradient(
    plane: torch.Tensor,
    first: tuple[float, float, float],
    last: tuple[float, float, float],
    horizontal: bool = True,
    ceiling: float = 1.0,
) -> torch.Tensor:
    """A straight ramp between two colours.

    Args:
        plane: The plane it is built for, which sets the size, dtype and device.
        first: The colour at the left or the top, three floats in 0 to 1.
        last: The colour at the right or the bottom.
        horizontal: Whether the ramp runs across the frame rather than down it.
        ceiling: The value white sits at.

    Returns:
        A ``(height, width, 3)`` tensor.
    """
    mask = linear_mask(plane, horizontal=horizontal)
    return composite_mask(fill(plane, first, ceiling), fill(plane, last, ceiling), mask)
