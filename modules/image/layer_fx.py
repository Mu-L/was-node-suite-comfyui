"""Layer effects baked into one layer of a stack: strokes, shadows, glows, fills, bevels.

Planes are ``(height, width)`` or ``(height, width, 3)`` floats from 0 to 1. Coverage is 1
where a layer paints.
"""

from __future__ import annotations

__all__ = [
    "BLEND_MODES",
    "BEVEL_DIRECTIONS",
    "BEVEL_STYLES",
    "EPSILON",
    "GLOW_MODES",
    "OVERLAY_FILLS",
    "SHADOW_MODES",
    "STROKE_POSITIONS",
    "applied",
    "bevel",
    "bevel_margin",
    "blend",
    "blur",
    "chosen",
    "colour",
    "coverage",
    "fill",
    "frames",
    "glow",
    "glow_margin",
    "grow",
    "inside",
    "offset",
    "over",
    "overlay",
    "padded",
    "ramp",
    "replaced",
    "report",
    "shadow",
    "shadow_margin",
    "shrink",
    "slopes",
    "shifted",
    "stack",
    "stroke",
    "stroke_margin",
    "trimmed",
]

import math

import torch
import torch.nn.functional as F

from .. import log
from ..interface import run_result
from . import blend_modes, convolve, layer_ops

logger = log.get_logger("image.layer_fx")

#: Smallest value treated as non-zero when a division or a test would otherwise be unstable.
EPSILON = blend_modes.EPSILON

#: Every blend mode a layer stack names, which an effect mixes with the layer through.
BLEND_MODES = blend_modes.MODES

#: Where a stroke sits against the edge of the coverage it follows.
STROKE_POSITIONS = ("outer", "centre", "inner")

#: Which side of the coverage a shadow falls on.
SHADOW_MODES = ("drop", "inner")

#: Which side of the coverage a glow spreads over.
GLOW_MODES = ("outer", "inner")

#: What an overlay paints the coverage with.
OVERLAY_FILLS = ("flat", "gradient")

#: How a bevel sits against the edge of the coverage.
BEVEL_STYLES = ("inner", "outer", "emboss")

#: Whether a bevel reads as raised or as pressed in.
BEVEL_DIRECTIONS = ("up", "down")

#: Luminance weights the blend modes that carry luminance read, in red, green, blue order.
LUMINANCE = blend_modes.LUMINANCE

#: Hexadecimal digits a colour string may hold.
HEX_DIGITS = "0123456789abcdefABCDEF"


def _whole(value, fallback: int = 0) -> int:
    """One number as a whole number, or ``fallback`` where it is not one."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback



def blend(backdrop: torch.Tensor, source: torch.Tensor, mode: str = "normal") -> torch.Tensor:
    """Mix two colour planes through one blend mode.

    Args:
        backdrop: ``(height, width, 3)`` colours already in place.
        source: ``(height, width, 3)`` colours going over them.
        mode: One of :data:`BLEND_MODES`. An unknown name is read as ``"normal"``.

    Returns:
        A ``(height, width, 3)`` tensor.
    """
    return blend_modes.blend(backdrop, source, mode)


def over(backdrop: torch.Tensor, backdrop_alpha: torch.Tensor, source: torch.Tensor,
         source_alpha: torch.Tensor, mode: str = "normal", empty=None):
    """Lay a source over a backdrop, adding coverage where the source paints.

    Args:
        backdrop: ``(height, width, 3)`` colours underneath.
        backdrop_alpha: ``(height, width)`` coverage of the backdrop.
        source: ``(height, width, 3)`` colours going over them.
        source_alpha: ``(height, width)`` coverage of the source.
        mode: One of :data:`BLEND_MODES`.
        empty: ``(height, width, 3)`` colours held where neither side covers anything.
            None holds the backdrop's.

    Returns:
        ``(colours, coverage)``, each the shape it arrived in.
    """
    back_a = backdrop_alpha.unsqueeze(-1)
    src_a = source_alpha.unsqueeze(-1)
    mixed = source + (blend(backdrop, source, mode) - source) * back_a
    answer_alpha = source_alpha + backdrop_alpha * (1.0 - source_alpha)
    weighted = src_a * mixed + (1.0 - src_a) * back_a * backdrop
    reach = answer_alpha.unsqueeze(-1)
    usable = reach > EPSILON
    bare = backdrop if empty is None else empty
    return torch.where(usable, weighted / torch.where(usable, reach, torch.ones_like(reach)),
                       bare), answer_alpha


def inside(backdrop: torch.Tensor, source: torch.Tensor, strength: torch.Tensor,
           mode: str = "normal") -> torch.Tensor:
    """Mix a source into a backdrop without changing what the backdrop covers.

    Args:
        backdrop: ``(height, width, 3)`` colours to paint into.
        source: ``(height, width, 3)`` colours going into them.
        strength: ``(height, width)`` share of the mix, 0 to 1, already clipped to the
            backdrop's coverage.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        A ``(height, width, 3)`` tensor.
    """
    mixed = blend(backdrop, source, mode)
    return backdrop + (mixed - backdrop) * strength.unsqueeze(-1)


def colour(text, name: str = "color") -> tuple[float, float, float]:
    """Read a hex colour string as three floats.

    Args:
        text: A colour such as ``"#ff8800"``, ``"ff8800"`` or the three digit ``"#f80"``.
        name: Which input the string came from, for the message a bad one raises.

    Returns:
        ``(red, green, blue)``, each 0.0 to 1.0.

    Raises:
        ValueError: The string is not three or six hexadecimal digits.
    """
    digits = str(text).strip().lstrip("#").strip()
    if len(digits) == 3 and all(digit in HEX_DIGITS for digit in digits):
        digits = "".join(digit * 2 for digit in digits)
    if len(digits) == 6 and all(digit in HEX_DIGITS for digit in digits):
        packed = int(digits, 16)
        return ((packed >> 16 & 255) / 255.0, (packed >> 8 & 255) / 255.0,
                (packed & 255) / 255.0)
    raise ValueError(
        f"{name} was given {text!r}, which is not a colour. Write six hexadecimal digits "
        f"such as #ff8800, or three such as #f80."
    )


def fill(tint, height: int, width: int, dtype=None, device=None) -> torch.Tensor:
    """A frame of one flat colour.

    Args:
        tint: ``(red, green, blue)``, each 0.0 to 1.0.
        height: Frame height in pixels.
        width: Frame width in pixels.
        dtype: Element type of the result.
        device: Where the frame is built.

    Returns:
        A ``(height, width, 3)`` tensor.
    """
    return torch.tensor(tuple(tint), dtype=dtype, device=device).view(1, 1, 3).expand(
        int(height), int(width), 3)


def ramp(height: int, width: int, angle: float, dtype=None, device=None) -> torch.Tensor:
    """A 0 to 1 linear ramp across a frame at an angle.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        angle: Degrees, counted counter-clockwise from pointing right. 0 puts 1 at the
            right edge, 90 puts it at the top.
        dtype: Element type of the result.
        device: Where the ramp is built.

    Returns:
        A ``(height, width)`` tensor running 0 at one side to 1 at the other.
    """
    columns = torch.linspace(0.0, 1.0, max(int(width), 1), dtype=dtype, device=device)
    rows = torch.linspace(0.0, 1.0, max(int(height), 1), dtype=dtype, device=device)
    turn = math.radians(float(angle))
    across = columns.view(1, -1) * math.cos(turn) - rows.view(-1, 1) * math.sin(turn)
    low = across.min()
    span = torch.clamp(across.max() - low, min=EPSILON)
    return (across - low) / span


def padded(plane: torch.Tensor, margin: int) -> torch.Tensor:
    """A plane with a band of zeros added on every side.

    Args:
        plane: ``(height, width)`` or ``(height, width, channels)`` tensor.
        margin: Pixels added to each side.

    Returns:
        A tensor twice ``margin`` larger in each direction.
    """
    margin = int(margin)
    if margin <= 0:
        return plane
    if plane.ndim == 2:
        stack_of = plane.reshape(1, 1, plane.shape[0], plane.shape[1])
        return F.pad(stack_of, (margin,) * 4)[0, 0]
    stack_of = plane.permute(2, 0, 1).unsqueeze(0)
    return F.pad(stack_of, (margin,) * 4)[0].permute(1, 2, 0)


def trimmed(plane: torch.Tensor, margin: int) -> torch.Tensor:
    """A plane with a band taken off every side.

    Args:
        plane: ``(height, width)`` or ``(height, width, channels)`` tensor.
        margin: Pixels taken off each side.

    Returns:
        A tensor twice ``margin`` smaller in each direction.
    """
    margin = int(margin)
    if margin <= 0:
        return plane
    return plane[margin:plane.shape[0] - margin, margin:plane.shape[1] - margin]


def blur(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Blur a plane with a separable Gaussian, reading everything outside it as empty.

    Args:
        plane: ``(height, width)`` or ``(height, width, channels)`` tensor.
        radius: Reach of the blur in pixels. At or below 0 the plane comes back as it is.

    Returns:
        A tensor of the same shape.
    """
    reach = int(round(float(radius)))
    if reach <= 0:
        return plane
    flat = plane.unsqueeze(-1) if plane.ndim == 2 else plane
    height, width = int(flat.shape[0]), int(flat.shape[1])
    stack_of = flat.permute(2, 0, 1).unsqueeze(0)
    stack_of = F.pad(stack_of, (reach + 1,) * 4)
    stack_of = convolve.gaussian_blur(stack_of, size=reach * 2 + 1)
    stack_of = stack_of[:, :, reach + 1:reach + 1 + height, reach + 1:reach + 1 + width]
    answer = stack_of[0].permute(1, 2, 0)
    return answer.squeeze(-1) if plane.ndim == 2 else answer


def _morphology(plane: torch.Tensor, radius: float, spread: bool) -> torch.Tensor:
    """A plane taken to the highest or lowest value within a disc."""
    reach = int(round(float(radius)))
    if reach <= 0:
        return plane
    height, width = int(plane.shape[0]), int(plane.shape[1])
    stack_of = F.pad(plane.reshape(1, 1, height, width), (reach,) * 4)
    disc = convolve.ellipse_kernel(reach, plane.dtype, plane.device)
    worked = convolve.dilate(stack_of, disc) if spread else convolve.erode(stack_of, disc)
    return worked[0, 0, reach:reach + height, reach:reach + width]


def grow(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Spread a plane out to the highest value within a disc.

    Args:
        plane: ``(height, width)`` tensor.
        radius: Disc radius in pixels. At or below 0 the plane comes back as it is.

    Returns:
        A tensor of the same shape.
    """
    return _morphology(plane, radius, True)


def shrink(plane: torch.Tensor, radius: float) -> torch.Tensor:
    """Pull a plane back to the lowest value within a disc.

    Args:
        plane: ``(height, width)`` tensor.
        radius: Disc radius in pixels. At or below 0 the plane comes back as it is.

    Returns:
        A tensor of the same shape.
    """
    return _morphology(plane, radius, False)


def slopes(plane: torch.Tensor):
    """How fast a plane rises across and down, one pixel apart.

    Args:
        plane: ``(height, width)`` tensor.

    Returns:
        ``(across, down)``, each the shape of the plane.
    """
    stack_of = plane.reshape(1, 1, plane.shape[0], plane.shape[1])
    edged = F.pad(stack_of, (1, 1, 1, 1), mode="replicate")
    across = (edged[:, :, 1:-1, 2:] - edged[:, :, 1:-1, :-2]) * 0.5
    down = (edged[:, :, 2:, 1:-1] - edged[:, :, :-2, 1:-1]) * 0.5
    return across[0, 0], down[0, 0]


def shifted(plane: torch.Tensor, across: int, down: int) -> torch.Tensor:
    """Move a plane by whole pixels, leaving zeros behind it.

    Args:
        plane: ``(height, width)`` tensor.
        across: Pixels to move right. A negative number moves left.
        down: Pixels to move down. A negative number moves up.

    Returns:
        A tensor of the same shape.
    """
    height, width = int(plane.shape[0]), int(plane.shape[1])
    across, down = int(across), int(down)
    moved = torch.zeros_like(plane)
    if abs(across) >= width or abs(down) >= height:
        return moved
    top, left = max(down, 0), max(across, 0)
    bottom, right = height + min(down, 0), width + min(across, 0)
    moved[top:bottom, left:right] = plane[top - down:bottom - down,
                                          left - across:right - across]
    return moved


def offset(angle: float, distance: float):
    """Where an effect lands for a direction and a distance.

    Args:
        angle: Degrees, counted counter-clockwise from pointing right.
        distance: Pixels from the layer.

    Returns:
        ``(across, down)`` in whole pixels, down counted on the screen.
    """
    turn = math.radians(float(angle))
    return (int(round(math.cos(turn) * float(distance))),
            int(round(-math.sin(turn) * float(distance))))


def coverage(picture: torch.Tensor, hidden=None) -> torch.Tensor:
    """Where a layer paints, read off its picture and its transparency mask.

    Args:
        picture: ``(height, width, channels)`` picture. A fourth channel is its alpha.
        hidden: ``(height, width)`` transparency, 1 where the layer is hidden, or None.

    Returns:
        A ``(height, width)`` tensor, 1 where the layer paints.
    """
    if int(picture.shape[-1]) >= 4:
        found = picture[:, :, 3]
    else:
        found = torch.ones(picture.shape[:2], dtype=picture.dtype, device=picture.device)
    if hidden is not None:
        veil = hidden.to(dtype=found.dtype, device=found.device)
        if veil.shape[0] != found.shape[0] or veil.shape[1] != found.shape[1]:
            veil = F.interpolate(veil.reshape(1, 1, veil.shape[0], veil.shape[1]),
                                 size=(int(found.shape[0]), int(found.shape[1])),
                                 mode="bilinear")[0, 0]
        found = found * torch.clamp(1.0 - veil, 0.0, 1.0)
    return torch.clamp(found, 0.0, 1.0)


def stack(document) -> list:
    """Every layer of a document as its own dictionary, lowest in the stack first.

    Args:
        document: A ``LAYERS`` value, or a bare list of layer dictionaries.

    Returns:
        The entries carrying a picture, in stacking order.
    """
    entries = document.get("layers", []) if isinstance(document, dict) else document
    found = [
        entry for entry in (entries or [])
        if isinstance(entry, dict) and isinstance(entry.get("image"), torch.Tensor)
    ]
    return sorted(found, key=lambda entry: _whole(entry.get("z_index")))


def chosen(entries, index: int, name: str, node: str):
    """The layer an index or a name picks out of a stack.

    Args:
        entries: The stack, lowest first.
        index: Position in the stack. A negative number counts from the top.
        name: Name to match, ignoring case and surrounding space. Empty reads the index.
        node: The name of the node asking, for the message a miss raises.

    Returns:
        ``(position, entry)``, the position counted from the bottom of the stack.

    Raises:
        ValueError: The stack is empty, the name matches nothing, or the index is past the
            end of the stack.
    """
    if not entries:
        raise ValueError(
            f"{node} was handed a stack with no layer in it. Wire in a stack that Add Layer "
            f"or Layers From Bounding Boxes has put at least one layer into."
        )
    wanted = str(name or "").strip().casefold()
    if wanted:
        for position, entry in enumerate(entries):
            if str(entry.get("name") or "").strip().casefold() == wanted:
                return position, entry
        held = ", ".join(repr(str(entry.get("name") or "")) for entry in entries)
        raise ValueError(
            f"{node} found no layer named {name!r} in the stack, which holds {held}. Set the "
            f"name on Add Layer, or clear layer_name to pick the layer by layer_index."
        )
    position = int(index)
    if position < 0:
        position += len(entries)
    if not 0 <= position < len(entries):
        raise ValueError(
            f"{node} was asked for layer {index} of a stack holding {len(entries)}. Use 0 for "
            f"the bottom layer, {len(entries) - 1} for the top, or -1 to count from the top."
        )
    return position, entries[position]


def frames(entry) -> list:
    """Each picture a layer carries, with the transparency that goes with it.

    Args:
        entry: A layer dictionary.

    Returns:
        ``(picture, hidden)`` pairs. A picture is ``(height, width, channels)`` and a
        transparency is ``(height, width)`` or None.
    """
    pictures = entry["image"]
    if pictures.ndim == 3:
        pictures = pictures.unsqueeze(0)
    veil = entry.get("mask")
    if isinstance(veil, torch.Tensor) and veil.ndim == 2:
        veil = veil.unsqueeze(0)
    found = []
    for index in range(int(pictures.shape[0])):
        plane = None
        if isinstance(veil, torch.Tensor) and veil.ndim == 3:
            if int(veil.shape[0]) == 1:
                plane = veil[0]
            elif index < int(veil.shape[0]):
                plane = veil[index]
        found.append((pictures[index], plane))
    return found


def replaced(document, entry, pictures: torch.Tensor, coverages: torch.Tensor,
             margin: int):
    """A document with one layer's picture, transparency and placement replaced.

    Args:
        document: The ``LAYERS`` value the layer came out of.
        entry: The layer dictionary being replaced.
        pictures: ``(frames, height, width, 3)`` pictures the effect was baked into.
        coverages: ``(frames, height, width)`` coverage, 1 where the layer paints.
        margin: Pixels the picture grew by on every side.

    Returns:
        ``(document, entry)``, a new document and the layer standing in it. The document
        handed in is left as it was.
    """
    grown = dict(entry)
    grown["image"] = pictures
    grown["mask"] = torch.clamp(1.0 - coverages, 0.0, 1.0)
    margin = int(margin)
    if margin > 0:
        source_width = int(pictures.shape[2]) - 2 * margin
        source_height = int(pictures.shape[1]) - 2 * margin
        shown_width = _whole(entry.get("w")) or source_width
        shown_height = _whole(entry.get("h")) or source_height
        across = 2 * int(round(margin * shown_width / max(source_width, 1)))
        down = 2 * int(round(margin * shown_height / max(source_height, 1)))
        grown["x"] = _whole(entry.get("x")) - across // 2
        grown["y"] = _whole(entry.get("y")) - down // 2
        if _whole(entry.get("w")) > 0:
            grown["w"] = shown_width + across
        if _whole(entry.get("h")) > 0:
            grown["h"] = shown_height + down
    held = list(document.get("layers") or []) if isinstance(document, dict) else list(document)
    answer = dict(document) if isinstance(document, dict) else {}
    answer["layers"] = [grown if item is entry else item for item in held]
    answer.setdefault("version", 1)
    return answer, grown


def applied(document, entry, render, work: int, keep: int):
    """Render an effect over every frame of a layer and bake it back into the document.

    Args:
        document: The ``LAYERS`` value the layer came out of.
        entry: The layer dictionary to work on.
        render: Callable handed ``(colours, coverage)`` padded by ``work`` and answering
            the pair the effect produced.
        work: Pixels of room the effect is rendered with.
        keep: Pixels of that room the layer keeps.

    Returns:
        ``(document, entry)``, a new document and the layer as it now stands in it.
    """
    work, keep = max(int(work), 0), max(int(keep), 0)
    work = max(work, keep)
    pictures, coverages = [], []
    for picture, veil in frames(entry):
        alpha = coverage(picture, veil)
        colours = picture[:, :, :3]
        if int(colours.shape[2]) < 3:
            colours = colours[:, :, :1].expand(-1, -1, 3)
        colours, alpha = render(padded(colours, work), padded(alpha, work))
        pictures.append(trimmed(colours, work - keep))
        coverages.append(trimmed(alpha, work - keep))
    return replaced(document, entry, torch.stack(pictures, dim=0),
                    torch.stack(coverages, dim=0), keep)


def _stroke_reach(position: str, width: int):
    """``(outside, inside)`` pixels a stroke of one width and position reaches."""
    width = max(int(width), 0)
    if position == "outer":
        return width, 0
    if position == "inner":
        return 0, width
    outside = (width + 1) // 2
    return outside, width - outside


def stroke_margin(position: str, width: int):
    """Room a stroke is rendered with, and how much of it the layer keeps.

    Args:
        position: One of :data:`STROKE_POSITIONS`.
        width: Band width in pixels.

    Returns:
        ``(work, keep)`` in pixels, both counted on every side.
    """
    outside = _stroke_reach(position, width)[0]
    return outside, outside


def stroke(colours: torch.Tensor, alpha: torch.Tensor, position: str, width: int, tint,
           opacity: float, mode: str):
    """Draw a band along the edge of a layer's coverage.

    Args:
        colours: ``(height, width, 3)`` layer colours, already padded.
        alpha: ``(height, width)`` coverage, padded the same way.
        position: One of :data:`STROKE_POSITIONS`.
        width: Band width in pixels.
        tint: ``(red, green, blue)`` the band is drawn in, each 0.0 to 1.0.
        opacity: How strongly the band is laid down, 0.0 to 1.0.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(colours, coverage)``.
    """
    outside, within = _stroke_reach(position, width)
    paint = fill(tint, alpha.shape[0], alpha.shape[1], alpha.dtype, alpha.device)
    answer, answer_alpha = colours, alpha
    if within > 0:
        band = torch.clamp(alpha - shrink(alpha, within), 0.0, 1.0)
        answer = inside(answer, paint, band * opacity, mode)
    if outside > 0:
        band = torch.clamp(grow(alpha, outside) - alpha, 0.0, 1.0)
        answer, answer_alpha = over(answer, answer_alpha, paint, band * opacity, mode)
    return answer, answer_alpha


def shadow_margin(side: str, angle: float, distance: float, size: int):
    """Room a shadow is rendered with, and how much of it the layer keeps.

    Args:
        side: One of :data:`SHADOW_MODES`.
        angle: Degrees the shadow is cast at.
        distance: Pixels the shadow moves.
        size: Blur radius in pixels.

    Returns:
        ``(work, keep)`` in pixels, both counted on every side. An inner shadow keeps none
        of its room.
    """
    across, down = offset(angle, distance)
    reach = max(abs(across), abs(down)) + max(int(size), 0)
    return reach, 0 if side == "inner" else reach


def shadow(colours: torch.Tensor, alpha: torch.Tensor, side: str, angle: float,
           distance: float, spread: float, size: int, tint, opacity: float, mode: str):
    """Cast a shadow behind a layer's coverage or inside it.

    Args:
        colours: ``(height, width, 3)`` layer colours, already padded.
        alpha: ``(height, width)`` coverage, padded the same way.
        side: One of :data:`SHADOW_MODES`.
        angle: Degrees, counted counter-clockwise from pointing right.
        distance: Pixels the shadow moves from the layer.
        spread: Share of ``size`` spent hardening the edge rather than blurring it.
        size: Blur radius in pixels.
        tint: ``(red, green, blue)`` the shadow is drawn in, each 0.0 to 1.0.
        opacity: How strongly the shadow is laid down, 0.0 to 1.0.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(colours, coverage)``.
    """
    across, down = offset(angle, distance)
    size = max(int(size), 0)
    choke = int(round(min(max(float(spread), 0.0), 1.0) * size))
    paint = fill(tint, alpha.shape[0], alpha.shape[1], alpha.dtype, alpha.device)
    if side == "inner":
        shape = blur(grow(1.0 - shifted(alpha, across, down), choke), size - choke)
        band = torch.clamp(shape, 0.0, 1.0) * alpha * opacity
        return inside(colours, paint, band, mode), alpha
    shape = blur(grow(shifted(alpha, across, down), choke), size - choke)
    return over(paint, torch.clamp(shape, 0.0, 1.0) * opacity, colours, alpha, mode,
                colours)


def glow_margin(side: str, size: int):
    """Room a glow is rendered with, and how much of it the layer keeps.

    Args:
        side: One of :data:`GLOW_MODES`.
        size: Blur radius in pixels.

    Returns:
        ``(work, keep)`` in pixels, both counted on every side. An inner glow keeps none of
        its room.
    """
    reach = max(int(size), 0)
    return reach, 0 if side == "inner" else reach


def glow(colours: torch.Tensor, alpha: torch.Tensor, side: str, size: int, spread: float,
         tint, opacity: float, mode: str):
    """Spread light outward from a layer's coverage or inward from its edge.

    Args:
        colours: ``(height, width, 3)`` layer colours, already padded.
        alpha: ``(height, width)`` coverage, padded the same way.
        side: One of :data:`GLOW_MODES`.
        size: How far the light reaches, in pixels. At or below 0 nothing is drawn.
        spread: Share of ``size`` spent hardening the edge rather than blurring it.
        tint: ``(red, green, blue)`` the glow is drawn in, each 0.0 to 1.0.
        opacity: How strongly the glow is laid down, 0.0 to 1.0.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(colours, coverage)``.
    """
    size = max(int(size), 0)
    if size <= 0:
        return colours, alpha
    choke = int(round(min(max(float(spread), 0.0), 1.0) * size))
    paint = fill(tint, alpha.shape[0], alpha.shape[1], alpha.dtype, alpha.device)
    if side == "inner":
        shape = blur(grow(1.0 - alpha, choke), size - choke)
        band = torch.clamp(shape, 0.0, 1.0) * alpha * opacity
        return inside(colours, paint, band, mode), alpha
    shape = blur(grow(alpha, choke), size - choke)
    return over(paint, torch.clamp(shape, 0.0, 1.0) * opacity, colours, alpha, mode,
                colours)


def overlay(colours: torch.Tensor, alpha: torch.Tensor, kind: str, tint, second,
            angle: float, opacity: float, mode: str):
    """Paint a layer's coverage with a flat colour or a two-stop gradient.

    Args:
        colours: ``(height, width, 3)`` layer colours.
        alpha: ``(height, width)`` coverage.
        kind: One of :data:`OVERLAY_FILLS`.
        tint: ``(red, green, blue)`` at the start of the fill, each 0.0 to 1.0.
        second: ``(red, green, blue)`` at the end of a gradient, each 0.0 to 1.0.
        angle: Degrees a gradient runs at, counted counter-clockwise from pointing right.
        opacity: How strongly the fill is laid down, 0.0 to 1.0.
        mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(colours, coverage)``. The coverage is the one handed in.
    """
    height, width = int(alpha.shape[0]), int(alpha.shape[1])
    low = fill(tint, height, width, alpha.dtype, alpha.device)
    if kind == "gradient":
        high = fill(second, height, width, alpha.dtype, alpha.device)
        weights = ramp(height, width, angle, alpha.dtype, alpha.device).unsqueeze(-1)
        paint = low + (high - low) * weights
    else:
        paint = low
    return inside(colours, paint, alpha * opacity, mode), alpha


def bevel_margin(style: str, size: int, soften: int):
    """Room a bevel is rendered with, and how much of it the layer keeps.

    Args:
        style: One of :data:`BEVEL_STYLES`.
        size: How far the slope runs, in pixels.
        soften: Extra blur on the slope, in pixels.

    Returns:
        ``(work, keep)`` in pixels, both counted on every side. An inner bevel keeps none
        of its room.
    """
    span = max(int(size), 0) + max(int(soften), 0)
    return span, 0 if style == "inner" else span


def bevel(colours: torch.Tensor, alpha: torch.Tensor, style: str, depth: float,
          direction: str, size: int, soften: int, angle: float, altitude: float,
          highlight, highlight_opacity: float, highlight_mode: str, shade,
          shade_opacity: float, shade_mode: str):
    """Light a slope built from a layer's coverage.

    Args:
        colours: ``(height, width, 3)`` layer colours, already padded.
        alpha: ``(height, width)`` coverage, padded the same way.
        style: One of :data:`BEVEL_STYLES`.
        depth: How steep the slope reads, 0.0 upward.
        direction: One of :data:`BEVEL_DIRECTIONS`.
        size: How far the slope runs, in pixels.
        soften: Extra blur on the slope, in pixels.
        angle: Degrees the light comes from, counted counter-clockwise from pointing right.
        altitude: Degrees the light sits above the surface, 0 to 90.
        highlight: ``(red, green, blue)`` the lit side is drawn in, each 0.0 to 1.0.
        highlight_opacity: How strongly the lit side is laid down, 0.0 to 1.0.
        highlight_mode: One of :data:`BLEND_MODES`.
        shade: ``(red, green, blue)`` the unlit side is drawn in, each 0.0 to 1.0.
        shade_opacity: How strongly the unlit side is laid down, 0.0 to 1.0.
        shade_mode: One of :data:`BLEND_MODES`.

    Returns:
        ``(colours, coverage)``.
    """
    span = max(max(int(size), 0) + max(int(soften), 0), 1)
    surface = blur(blur(alpha, size), soften)
    across, down = slopes(surface)
    turn = math.radians(float(angle))
    lit = -(across * math.cos(turn) - down * math.sin(turn))
    lit = lit * span * float(depth) * math.cos(math.radians(float(altitude)))
    if direction == "down":
        lit = -lit
    lit = torch.clamp(lit, -1.0, 1.0)

    if style == "inner":
        region = alpha
    elif style == "outer":
        region = torch.clamp(grow(alpha, span) - alpha, 0.0, 1.0)
    else:
        region = torch.clamp(grow(alpha, span), 0.0, 1.0)

    height, width = int(alpha.shape[0]), int(alpha.shape[1])
    warm = fill(highlight, height, width, alpha.dtype, alpha.device)
    cool = fill(shade, height, width, alpha.dtype, alpha.device)
    up = torch.clamp(lit, min=0.0) * region * highlight_opacity
    low = torch.clamp(-lit, min=0.0) * region * shade_opacity
    if style == "inner":
        answer = inside(colours, warm, up, highlight_mode)
        return inside(answer, cool, low, shade_mode), alpha
    answer, answer_alpha = over(colours, alpha, warm, up, highlight_mode)
    return over(answer, answer_alpha, cool, low, shade_mode)


def report(node: str, document, entry, position: int, total: int, margin: int) -> None:
    """Publish what an effect did, for the node's own interface to draw. Never raises.

    Args:
        node: The name of the node reporting.
        document: The ``LAYERS`` value the layer came out of.
        entry: The layer dictionary after the effect was baked in.
        position: Where that layer sits in the stack, counted from the bottom.
        total: How many layers the stack holds.
        margin: Pixels the layer grew by on every side.
    """
    try:
        if not run_result.watching():
            return
        pictures = entry["image"]
        canvas = layer_ops.size_of(document)
        run_result.publish(
            status=run_result.OK,
            summary=(
                f"{node} baked into layer {position} of {total}, which grew {2 * margin}px "
                f"in each direction"
                if margin else f"{node} baked into layer {position} of {total}"
            ),
            counts={"layer": position, "layers": total, "margin px": margin},
            facts={
                "name": str(entry.get("name") or "unnamed"),
                "layer size": f"{int(pictures.shape[2])}x{int(pictures.shape[1])}",
                "placement": f"{_whole(entry.get('x'))}, {_whole(entry.get('y'))}",
                "canvas": f"{canvas[0]}x{canvas[1]}",
            },
        )
    except Exception as error:
        logger.debug("%s published no report (%s)", node, error)
