"""Converting between the pack's ``IMAGE_BOUNDS`` rows and ComfyUI's bounding boxes.

A bounds row is ``(rmin, rmax, cmin, cmax)`` with every edge inclusive. A box is
``{"x", "y", "width", "height"}`` with the origin at its top left corner.
"""

from __future__ import annotations

__all__ = [
    "ORDERS",
    "area",
    "boxes_to_rows",
    "clamped",
    "grown",
    "normalise",
    "ordered",
    "overlap",
    "rows_to_boxes",
    "suppressed",
]


def normalise(boxes) -> list[dict]:
    """Read a bounding box value into a flat list of box dictionaries.

    Args:
        boxes: One box, a list of boxes, a list of per-frame lists of boxes, which is what a
            detector emits, or JSON text holding any of those.

    Returns:
        Every box found, in the order it was given. Empty where the value holds none.
    """
    if isinstance(boxes, str):
        import json

        try:
            boxes = json.loads(boxes)
        except ValueError:
            return []
    if not boxes:
        return []
    if isinstance(boxes, dict):
        return [boxes]
    found: list[dict] = []
    for entry in boxes:
        if isinstance(entry, dict):
            found.append(entry)
        elif isinstance(entry, (list, tuple)):
            found.extend(item for item in entry if isinstance(item, dict))
    return found


def boxes_to_rows(boxes) -> list[tuple[int, int, int, int]]:
    """Convert bounding boxes into inclusive bounds rows.

    Args:
        boxes: A bounding box value, as :func:`normalise` accepts it.

    Returns:
        One ``(rmin, rmax, cmin, cmax)`` row per box. A box with no area is dropped.
    """
    rows = []
    for box in normalise(boxes):
        left = int(round(float(box.get("x", 0))))
        top = int(round(float(box.get("y", 0))))
        width = int(round(float(box.get("width", 0))))
        height = int(round(float(box.get("height", 0))))
        if width < 1 or height < 1:
            continue
        rows.append((top, top + height - 1, left, left + width - 1))
    return rows


def rows_to_boxes(rows, metadata=None) -> list[dict]:
    """Convert inclusive bounds rows into bounding boxes.

    Args:
        rows: ``(rmin, rmax, cmin, cmax)`` rows, as :func:`modules.image.bounds.rows`
            answers them.
        metadata: Attached to every box under ``metadata`` when given.

    Returns:
        One box dictionary per row, each ``{"x", "y", "width", "height"}``.
    """
    boxes = []
    for row in rows:
        rmin, rmax, cmin, cmax = (int(value) for value in row)
        box = {
            "x": cmin,
            "y": rmin,
            "width": max(0, cmax - cmin + 1),
            "height": max(0, rmax - rmin + 1),
        }
        if metadata:
            box["metadata"] = dict(metadata)
        boxes.append(box)
    return boxes


#: How a filtered set is ordered.
ORDERS = (
    "as found",
    "area, largest first",
    "area, smallest first",
    "left to right",
    "top to bottom",
)


def area(box) -> int:
    """The pixels one box covers."""
    return max(0, int(box.get("width", 0))) * max(0, int(box.get("height", 0)))


def overlap(first, second) -> float:
    """How much two boxes share, as the intersection over their union.

    Args:
        first: A box dictionary.
        second: A box dictionary.

    Returns:
        0.0 where they do not touch, 1.0 where they are the same rectangle.
    """
    left = max(int(first.get("x", 0)), int(second.get("x", 0)))
    top = max(int(first.get("y", 0)), int(second.get("y", 0)))
    right = min(
        int(first.get("x", 0)) + int(first.get("width", 0)),
        int(second.get("x", 0)) + int(second.get("width", 0)),
    )
    bottom = min(
        int(first.get("y", 0)) + int(first.get("height", 0)),
        int(second.get("y", 0)) + int(second.get("height", 0)),
    )
    shared = max(0, right - left) * max(0, bottom - top)
    union = area(first) + area(second) - shared
    return shared / union if union else 0.0


def grown(box, pixels: int) -> dict:
    """One box with every edge moved out by ``pixels``, or in where it is negative."""
    grown_box = dict(box)
    grown_box["x"] = int(box.get("x", 0)) - pixels
    grown_box["y"] = int(box.get("y", 0)) - pixels
    grown_box["width"] = max(0, int(box.get("width", 0)) + pixels * 2)
    grown_box["height"] = max(0, int(box.get("height", 0)) + pixels * 2)
    return grown_box


def clamped(box, width: int, height: int) -> dict:
    """One box held inside a frame, losing whatever fell outside it."""
    left = max(0, min(width, int(box.get("x", 0))))
    top = max(0, min(height, int(box.get("y", 0))))
    right = max(0, min(width, int(box.get("x", 0)) + int(box.get("width", 0))))
    bottom = max(0, min(height, int(box.get("y", 0)) + int(box.get("height", 0))))
    held = dict(box)
    held.update({"x": left, "y": top, "width": right - left, "height": bottom - top})
    return held


def suppressed(found, threshold: float) -> list[dict]:
    """Drop every box that overlaps a larger one by more than ``threshold``.

    Args:
        found: Boxes to thin, in any order.
        threshold: Share of the union two boxes may share before the smaller is dropped.
            1.0 or more keeps everything.

    Returns:
        The boxes kept, largest first.
    """
    if threshold >= 1.0:
        return list(found)
    kept: list[dict] = []
    for box in sorted(found, key=area, reverse=True):
        if all(overlap(box, held) <= threshold for held in kept):
            kept.append(box)
    return kept


def ordered(found, order: str) -> list[dict]:
    """Boxes sorted the way ``order`` names, which is a member of :data:`ORDERS`."""
    if order == ORDERS[1]:
        return sorted(found, key=area, reverse=True)
    if order == ORDERS[2]:
        return sorted(found, key=area)
    if order == ORDERS[3]:
        return sorted(found, key=lambda box: (int(box.get("x", 0)), int(box.get("y", 0))))
    if order == ORDERS[4]:
        return sorted(found, key=lambda box: (int(box.get("y", 0)), int(box.get("x", 0))))
    return list(found)
