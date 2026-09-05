"""A high dynamic range image carried to the browser as a scene's environment."""

from __future__ import annotations

__all__ = ["FORMATS", "MAX_BYTES", "SUFFIXES", "carried"]

from pathlib import Path

from ..interface import three_asset
from ..log import get_logger

logger = get_logger("threejs.environments")

#: Suffixes a loader in the browser can read, against the content type they are served as.
FORMATS = {
    ".hdr": "image/vnd.radiance",
    ".exr": "image/x-exr",
}

#: The suffixes above, for a file menu.
SUFFIXES = tuple(FORMATS)

#: Largest environment accepted, so one file cannot fill the asset store on its own.
MAX_BYTES = 128 * 1024 * 1024


def carried(path: Path) -> tuple[str, str]:
    """Hold one environment file for the browser and answer where to fetch it.

    Args:
        path: The image on disk, already resolved through the containment layer.

    Returns:
        ``(url, format)``, the address the browser fetches and the suffix without its dot.

    Raises:
        ValueError: The suffix has no loader, or the file is larger than :data:`MAX_BYTES`.
        OSError: The file could not be read.
    """
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(
            f"{path.name} is a {suffix or 'suffixless'} file and the environment reads "
            f"{', '.join(SUFFIXES)}. Convert it, or wire an ordinary image into the image "
            f"input instead."
        )

    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(
            f"{path.name} is {size / 1048576:.0f} MB and the limit is "
            f"{MAX_BYTES // 1048576} MB. Resize it to 2k or 4k across; an environment is "
            f"blurred into a reflection and gains nothing from more."
        )

    body = path.read_bytes()
    key = three_asset.keep(body, FORMATS[suffix])
    logger.info("%s is held for the browser as %s, %.1f KB", path.name, key, size / 1024.0)
    return "%s?key=%s" % (three_asset.ROUTE, key), suffix.lstrip(".")
