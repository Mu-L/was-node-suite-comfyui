"""Suite-wide constants that belong inside no single module.

Holds ``ALLOWED_EXT``, the image extensions the file-listing nodes and
``VideoWriter.create_video`` accept, and ``MAX_SEQUENCE_FRAMES``, the ceiling on one load.
"""

from __future__ import annotations

__all__ = ["ALLOWED_EXT", "MAX_SEQUENCE_FRAMES"]

#: The image extensions the file-listing nodes accept.
ALLOWED_EXT = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")

#: Most frames one load answers. A load is held in memory all at once, so the ceiling is
#: what keeps a mistyped pattern from reading a whole drive into RAM.
MAX_SEQUENCE_FRAMES = 4096
