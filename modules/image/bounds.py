"""Reading the two rectangle values the image nodes wire.

A crop window is ``(size, (left, top, right, bottom))``, right and bottom exclusive. A
bounds row is ``(rmin, rmax, cmin, cmax)``, every edge inclusive.
"""

from __future__ import annotations

__all__ = ["crop_window", "rows"]


def crop_window(crop_data, node: str) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Read a crop window's size and rectangle out of a ``CROP_DATA`` value.

    Args:
        crop_data: ``(size, (left, top, right, bottom))`` from a crop node.
        node: The reading node's display name, opening the message where the value is
            refused.

    Returns:
        ``((width, height), (left, top, right, bottom))``, every value an integer.

    Raises:
        ValueError: The value is not a size and a four-edge rectangle.
    """
    try:
        size, window = crop_data
        width, height = (int(value) for value in size)
        left, top, right, bottom = (int(value) for value in window)
    except (TypeError, ValueError):
        raise ValueError(
            f"{node} could not read a crop window out of {crop_data!r}. A crop window is a "
            f"size and a rectangle, written ((width, height), (left, top, right, bottom)), "
            f"which is what the crop nodes emit."
        ) from None
    return (width, height), (left, top, right, bottom)


def rows(image_bounds) -> list[tuple[int, ...]]:
    """Normalise an ``IMAGE_BOUNDS`` value into a list of four-number rows.

    Args:
        image_bounds: A list of four-number rows, or one bare row.

    Returns:
        The rows, each as whole numbers. Empty where the value holds no rectangle.
    """
    if not image_bounds:
        return []
    first = image_bounds[0]
    candidates = [image_bounds] if isinstance(first, (int, float)) else list(image_bounds)
    return [tuple(int(value) for value in row) for row in candidates]
