"""Splitting an image into a grid of tiles, and the colour drawn between them.

Two pieces the tile nodes share: choosing a grid shape for a tile count, and reading the
border colour out of a hex string.
"""

from __future__ import annotations

__all__ = ["WHITE", "compute_grid", "parse_hex_color"]

#: Colour used when a hex string cannot be read.
WHITE = (255, 255, 255)


def compute_grid(max_tiles: int) -> tuple[int, int]:
    """Choose the squarest grid that holds exactly ``max_tiles`` tiles.

    Args:
        max_tiles: Tiles the image is cut into.

    Returns:
        ``(rows, columns)``, whose product is ``max_tiles``.
    """
    rows, columns = 1, max_tiles
    for candidate in range(1, int(max_tiles ** 0.5) + 1):
        if max_tiles % candidate == 0:
            rows, columns = candidate, max_tiles // candidate
    return rows, columns


def parse_hex_color(text: str, default: tuple[int, int, int] = WHITE) -> tuple[int, int, int]:
    """Read an RGB triple out of a hex colour string.

    Args:
        text: A colour such as ``"#FFFFFF"`` or ``"ff8800"``. Surrounding space and a
            leading ``#`` are ignored, and anything past the sixth digit is ignored too.
        default: Colour returned when the string holds no readable pair of digits.

    Returns:
        ``(red, green, blue)``, each 0-255.
    """
    digits = text.strip().lstrip("#")
    try:
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    except ValueError:
        return default
