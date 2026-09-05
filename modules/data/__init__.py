"""Assets the pack draws with, and the fonts a user adds.

``font.ttf`` and ``fonts/`` hold fonts, ``cascades/`` holds the OpenCV classifier
cascades. Fonts are also read from ``<config dir>/fonts``.
"""

from __future__ import annotations

from .paths import (
    CASCADES,
    FONTS,
    cascade_file,
    data_directory,
    font_catalog,
    font_file,
    font_names,
    user_font_directory,
)

__all__ = [
    "CASCADES",
    "FONTS",
    "cascade_file",
    "data_directory",
    "font_catalog",
    "font_file",
    "font_names",
    "user_font_directory",
]
