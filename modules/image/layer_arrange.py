"""Applying a saved arrangement to a ``LAYERS`` document.

An arrangement maps a layer index, lowest first, onto placement fields. Rotation arrives in
degrees and is stored in radians. A size change is resampled into the picture.
"""

from __future__ import annotations

import json
import math

__all__ = [
    "FIELDS",
    "NAME_CHARS",
    "SEPARATOR",
    "THUMBNAIL_EDGE",
    "applied",
    "arranged",
    "arrangement",
    "canvas_size",
    "chunks",
    "drawn_size",
    "entries",
    "placement",
    "rebuilt",
    "rows",
    "thumbnails",
]

#: Placement fields an arrangement may name on one layer.
FIELDS = ("x", "y", "w", "h", "rotation", "opacity", "visible", "z_index")

#: What separates the fields of a published row.
SEPARATOR = "|"

#: Characters of a layer name one published row carries.
NAME_CHARS = 32

#: Longest edge of one published layer thumbnail, in pixels.
THUMBNAIL_EDGE = 96

#: The narrowest and shortest a layer may be arranged to, in pixels.
MIN_SIDE = 1

#: What a decoded JSON value is called in an error, keyed on the Python type it arrives as.
_KINDS = {
    "dict": "an object",
    "list": "an array",
    "str": "a string",
    "int": "a number",
    "float": "a number",
    "bool": "a true or false",
    "NoneType": "a null",
}


def _kind(value) -> str:
    """What one decoded JSON value is called in an error message.

    Args:
        value: The decoded value.

    Returns:
        The phrase naming it, its article included.
    """
    return _KINDS.get(type(value).__name__, "a value")


def _whole(value, fallback: int = 0) -> int:
    """One number as a whole number, or ``fallback`` where it is not one.

    Args:
        value: Whatever the document or the arrangement carried.
        fallback: What to answer for anything that is not a number.

    Returns:
        The rounded number.
    """
    if isinstance(value, bool):
        return fallback
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _real(value, fallback: float = 0.0) -> float:
    """One number as a float, or ``fallback`` where it is not one.

    Args:
        value: Whatever the document or the arrangement carried.
        fallback: What to answer for anything that is not a number.

    Returns:
        The number as a float.
    """
    if isinstance(value, bool):
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number == number else fallback


def arrangement(text) -> dict:
    """Read the arrangement widget into one change per layer index.

    Args:
        text: The widget's JSON, or an empty string.

    Returns:
        Layer index to the fields of :data:`FIELDS` that entry names.

    Raises:
        ValueError: When the text is not a JSON object keyed on whole numbers.
    """
    body = (text or "").strip()
    if not body:
        return {}
    try:
        decoded = json.loads(body)
    except ValueError as error:
        raise ValueError(
            f"arrangement is not readable JSON: {error}. Write an object keyed on the layer "
            f'index, such as {{"0": {{"x": 64, "y": 0}}}}, or leave it as {{}} to pass the '
            f"stack through as it arrived."
        ) from error
    if not isinstance(decoded, dict):
        raise ValueError(
            f"arrangement is {_kind(decoded)}, not an object. Write an object keyed "
            f'on the layer index, such as {{"0": {{"x": 64, "y": 0}}}}.'
        )

    plan = {}
    for key, value in decoded.items():
        try:
            index = int(str(key).strip())
        except ValueError as error:
            raise ValueError(
                f"arrangement is keyed on {key!r}, which is not a layer index. Key each entry "
                f'on the layer number counted from the bottom of the stack, such as "0".'
            ) from error
        if index < 0:
            raise ValueError(
                f"arrangement names layer {index}, and a layer index starts at 0 for the "
                f"bottom of the stack. Count up from there."
            )
        if not isinstance(value, dict):
            raise ValueError(
                f"arrangement entry {key!r} is {_kind(value)}, not an object. Write "
                f'the fields to set, such as {{"x": 64, "y": 0, "w": 512}}.'
            )
        plan[index] = {name: value[name] for name in FIELDS if name in value}
    return plan


def entries(document) -> list:
    """Every layer of a document that carries a picture, lowest in the stack first.

    Args:
        document: A ``LAYERS`` value, or a bare list of layer dictionaries.

    Returns:
        The layer dictionaries themselves, in the order an arrangement indexes them.
    """
    found = document.get("layers", []) if isinstance(document, dict) else document
    kept = [
        entry
        for entry in found or []
        if isinstance(entry, dict) and entry.get("image") is not None
    ]
    return sorted(kept, key=lambda entry: _whole(entry.get("z_index")))


def drawn_size(entry) -> tuple[int, int]:
    """The size one layer is drawn at on the canvas, in pixels.

    Args:
        entry: A layer dictionary carrying a picture.

    Returns:
        ``(width, height)``, from the layer's own ``w`` and ``h`` where it names them and
        from its picture where it does not.
    """
    picture = entry.get("image")
    height, width = int(picture.shape[-3]), int(picture.shape[-2])
    return (
        max(MIN_SIDE, _whole(entry.get("w"), width) or width),
        max(MIN_SIDE, _whole(entry.get("h"), height) or height),
    )


def placement(entry) -> tuple:
    """Everything about one layer an arrangement can change.

    Args:
        entry: A layer dictionary carrying a picture.

    Returns:
        Position, drawn size, rotation, opacity, visibility and stacking, each read the way
        an arrangement writes it, so two layers compare equal where nothing was arranged.
    """
    return (
        _whole(entry.get("x")),
        _whole(entry.get("y")),
        drawn_size(entry),
        round(_real(entry.get("rotation")), 6),
        round(min(1.0, max(0.0, _real(entry.get("opacity", 1.0), 1.0))), 6),
        bool(entry.get("visible", True)),
        _whole(entry.get("z_index")),
    )


def canvas_size(document, layers) -> tuple[int, int]:
    """The canvas a document names, or the one its layers are drawn over.

    Args:
        document: A ``LAYERS`` value.
        layers: The layer dictionaries read out of it.

    Returns:
        ``(width, height)`` in pixels, at least one in each direction.
    """
    named = document.get("canvas") if isinstance(document, dict) else None
    if isinstance(named, (tuple, list)) and len(named) >= 2:
        width, height = _whole(named[0]), _whole(named[1])
        if width > 0 and height > 0:
            return width, height

    width = height = 0
    for entry in layers:
        across, down = drawn_size(entry)
        width = max(width, _whole(entry.get("x")) + across)
        height = max(height, _whole(entry.get("y")) + down)
    return max(1, width), max(1, height)


def _resampled(tensor, width: int, height: int, planes: bool):
    """One picture or coverage tensor at a new size.

    Args:
        tensor: The tensor to resample, in any of the layouts a layer carries.
        width: Width in pixels.
        height: Height in pixels.
        planes: True where the last axis holds channels, False for a bare coverage map.

    Returns:
        The tensor at the new size, in the layout it arrived in.
    """
    import torch
    from torch.nn.functional import interpolate

    body = tensor if planes else tensor.unsqueeze(-1)
    batched = body if body.ndim == 4 else body.unsqueeze(0)
    moved = batched.movedim(-1, 1).to(torch.float32)
    scaled = interpolate(moved, size=(height, width), mode="bilinear", align_corners=False)
    out = scaled.movedim(1, -1).to(tensor.dtype)
    if body.ndim != 4:
        out = out[0]
    return out if planes else out[..., 0]


def _arranged(entry, change) -> dict:
    """One layer with a change applied, resampled where its drawn size moved.

    Args:
        entry: The layer dictionary to copy.
        change: The fields to set on it, or None.

    Returns:
        A new layer dictionary. The one that arrived is left as it was.
    """
    layer = dict(entry)
    if not change:
        return layer

    # The picture and the drawn size are left alone unless the change names one of the sides.
    if "w" in change or "h" in change:
        picture = layer["image"]
        height, width = int(picture.shape[-3]), int(picture.shape[-2])
        held_across, held_down = drawn_size(layer)
        across = _whole(change.get("w"), held_across)
        down = _whole(change.get("h"), held_down)
        # A side under one pixel names the size the layer is already drawn at.
        across = across if across >= MIN_SIDE else held_across
        down = down if down >= MIN_SIDE else held_down
        if (across, down) != (held_across, held_down):
            if (across, down) != (width, height):
                layer["image"] = _resampled(picture, across, down, planes=True)
                cover = layer.get("mask")
                if cover is not None:
                    layer["mask"] = _resampled(cover, across, down, planes=False)
            layer["w"], layer["h"] = across, down

    if "x" in change:
        layer["x"] = _whole(change["x"], _whole(layer.get("x")))
    if "y" in change:
        layer["y"] = _whole(change["y"], _whole(layer.get("y")))
    if "rotation" in change:
        turned = _real(change["rotation"], math.degrees(_real(layer.get("rotation"))))
        layer["rotation"] = math.radians(turned)
    if "opacity" in change:
        held = _real(layer.get("opacity", 1.0), 1.0)
        layer["opacity"] = min(1.0, max(0.0, _real(change["opacity"], held)))
    if "visible" in change:
        layer["visible"] = bool(change["visible"])
    if "z_index" in change:
        layer["z_index"] = _whole(change["z_index"], _whole(layer.get("z_index")))
    return layer


def arranged(layers, changes) -> list:
    """Every layer with its own change applied, in the order it was indexed.

    Args:
        layers: What :func:`entries` answered.
        changes: What :func:`arrangement` answered.

    Returns:
        One new layer dictionary each, so the ones that arrived are left as they were.
    """
    return [_arranged(entry, changes.get(index)) for index, entry in enumerate(layers)]


def rebuilt(document, layers) -> dict:
    """One document carrying arranged layers in place of the ones it held.

    Args:
        document: The ``LAYERS`` value the layers were read out of.
        layers: What :func:`arranged` answered.

    Returns:
        A new document. Entries carrying no picture are passed through untouched.
    """
    held = document.get("layers", []) if isinstance(document, dict) else document
    rest = [
        entry
        for entry in held or []
        if not (isinstance(entry, dict) and entry.get("image") is not None)
    ]

    version = _whole(document.get("version"), 1) if isinstance(document, dict) else 1
    out = {"version": version or 1, "layers": list(layers) + rest}
    named = document.get("canvas") if isinstance(document, dict) else None
    if isinstance(named, (tuple, list)) and len(named) >= 2:
        width, height = _whole(named[0]), _whole(named[1])
        if width > 0 and height > 0:
            out["canvas"] = (width, height)
    return out


def applied(document, changes) -> dict:
    """One document with an arrangement applied to its layers.

    Args:
        document: The ``LAYERS`` value to arrange.
        changes: What :func:`arrangement` answered.

    Returns:
        A new document holding copied layers, so the one that arrived is left as it was.
    """
    return rebuilt(document, arranged(entries(document), changes))


def rows(layers) -> list:
    """One published line per layer, for a panel to draw the stack from.

    Args:
        layers: Layer dictionaries in the order an arrangement indexes them.

    Returns:
        One line each, carrying the index, ``z_index``, ``x``, ``y``, ``w``, ``h``,
        visibility as 1 or 0, opacity and the name, separated by :data:`SEPARATOR` with the
        name last.
    """
    lines = []
    for index, entry in enumerate(layers):
        across, down = drawn_size(entry)
        name = str(entry.get("name") or "").replace("\n", " ").strip()[:NAME_CHARS]
        opacity = min(1.0, max(0.0, _real(entry.get("opacity", 1.0), 1.0)))
        lines.append(
            SEPARATOR.join(
                (
                    str(index),
                    str(_whole(entry.get("z_index"))),
                    str(_whole(entry.get("x"))),
                    str(_whole(entry.get("y"))),
                    str(across),
                    str(down),
                    "1" if bool(entry.get("visible", True)) else "0",
                    f"{opacity:.3f}",
                    name,
                )
            )
        )
    return lines


def chunks(lines, limit: int, most: int) -> list:
    """Published lines grouped so each group fits inside a character limit.

    Args:
        lines: What :func:`rows` answered.
        limit: Characters one group may run to, newlines included.
        most: How many groups to build. Lines past the last are left out.

    Returns:
        Each group as one text, lines joined by newlines.
    """
    groups, held, spent = [], [], 0
    for line in lines:
        cost = len(line) + (1 if held else 0)
        if held and spent + cost > limit:
            groups.append("\n".join(held))
            if len(groups) >= most:
                return groups
            held, spent = [], 0
            cost = len(line)
        held.append(line)
        spent += cost
    if held:
        groups.append("\n".join(held))
    return groups[:most]


def thumbnails(layers, width: int, height: int, edge: int = THUMBNAIL_EDGE):
    """Small pictures of a document, one per layer, each drawn over the whole canvas.

    Args:
        layers: Layer dictionaries in the order an arrangement indexes them.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        edge: Longest edge of one thumbnail, in pixels.

    Returns:
        An ``IMAGE`` tensor, ``(layers, height, width, 3)``, black where a layer does not
        cover the canvas. None where there is nothing to draw. Rotation and blend mode are
        not drawn.
    """
    import torch

    if not layers or width < 1 or height < 1:
        return None
    scale = min(1.0, edge / max(width, height))
    across = max(1, int(round(width * scale)))
    down = max(1, int(round(height * scale)))

    frames = []
    for entry in layers:
        picture = entry["image"]
        plane = picture[0] if picture.ndim == 4 else picture
        plane = plane.detach().to(torch.float32)
        cover = entry.get("mask")
        if cover is not None:
            cover = cover.detach().to(torch.float32)
            cover = cover[0] if cover.ndim == 3 else cover
        elif plane.shape[-1] == 4:
            cover = plane[..., 3]

        wide, tall = drawn_size(entry)
        box_w = max(1, int(round(wide * scale)))
        box_h = max(1, int(round(tall * scale)))
        patch = _resampled(plane[..., :3], box_w, box_h, planes=True)
        if patch.shape[-1] < 3:
            patch = patch.repeat(1, 1, 3)[:, :, :3]
        alpha = (
            _resampled(cover, box_w, box_h, planes=False)
            if cover is not None
            else torch.ones((box_h, box_w), dtype=patch.dtype, device=patch.device)
        )

        frame = torch.zeros((down, across, 3), dtype=patch.dtype, device=patch.device)
        left = int(round(_whole(entry.get("x")) * scale))
        top = int(round(_whole(entry.get("y")) * scale))
        right = min(across, left + box_w)
        bottom = min(down, top + box_h)
        source_left, source_top = max(0, -left), max(0, -top)
        left, top = max(0, left), max(0, top)
        if right > left and bottom > top:
            piece = patch[
                source_top : source_top + (bottom - top),
                source_left : source_left + (right - left),
            ]
            mask = alpha[
                source_top : source_top + (bottom - top),
                source_left : source_left + (right - left),
            ]
            frame[top:bottom, left:right] = piece * mask.clamp(0.0, 1.0).unsqueeze(-1)
        frames.append(frame.clamp(0.0, 1.0))
    return torch.stack(frames, dim=0)
