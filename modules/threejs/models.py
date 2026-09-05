"""A 3D model file carried to the browser for a Three.js scene."""

from __future__ import annotations

__all__ = ["FORMATS", "MAX_BYTES", "MAX_SIDECARS", "SIDECARS", "SUFFIXES", "carried"]

import json
import re
from pathlib import Path
from urllib.parse import unquote

from ..interface import three_asset
from ..log import get_logger

logger = get_logger("threejs.models")

#: Suffixes a loader in the browser can read, against the content type they are served as.
FORMATS = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "text/plain",
    ".stl": "model/stl",
    ".dae": "model/vnd.collada+xml",
    ".fbx": "model/fbx",
    ".ply": "model/ply",
    ".3mf": "model/3mf",
}

#: The suffixes above, for a file menu.
SUFFIXES = tuple(FORMATS)

#: Largest model accepted, so one file cannot fill the asset store on its own.
MAX_BYTES = 256 * 1024 * 1024

#: Files a model names by relative path, against the content type they are served as. A
#: ``.dae``, a ``.gltf`` and an ``.obj`` all keep their pictures beside them rather than
#: inside them.
SIDECARS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".tga": "image/x-tga",
    ".bin": "application/octet-stream",
    ".mtl": "model/mtl",
}

#: Most sidecars held for one model, and the most bytes across them.
MAX_SIDECARS = 24
MAX_SIDECAR_BYTES = 96 * 1024 * 1024


#: Bytes of a model read while looking for the names it references.
MAX_SCAN_BYTES = 64 * 1024 * 1024

#: A Collada image, a Wavefront material library and a Wavefront texture map.
_COLLADA_IMAGE = re.compile(r"<init_from>\s*([^<\s][^<]*?)\s*</init_from>", re.I)
_OBJ_LIBRARY = re.compile(r"^\s*mtllib\s+(.+?)\s*$", re.I | re.M)
_MTL_MAP = re.compile(r"^\s*map_\w+\s+(?:-\S+\s+\S+\s+)*(.+?)\s*$", re.I | re.M)


def _referenced(path: Path) -> list[str]:
    """The file names a model names, read out of the model itself.

    Args:
        path: The model on disk.

    Returns:
        Names as the model spells them, in the order found, without duplicates.
    """
    suffix = path.suffix.lower()
    if suffix not in (".dae", ".gltf", ".obj"):
        return []
    if path.stat().st_size > MAX_SCAN_BYTES:
        logger.warning("%s is too large to scan for the files it names", path.name)
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        logger.warning("%s could not be read to find what it names: %s", path.name, error)
        return []

    names: list[str] = []
    if suffix == ".dae":
        names = _COLLADA_IMAGE.findall(text)
    elif suffix == ".gltf":
        try:
            document = json.loads(text)
        except ValueError:
            return []
        for group in ("images", "buffers"):
            for entry in document.get(group) or []:
                where = entry.get("uri") if isinstance(entry, dict) else None
                if isinstance(where, str) and not where.startswith("data:"):
                    names.append(where)
    else:
        for library in _OBJ_LIBRARY.findall(text):
            names.append(library)
            beside = path.parent / Path(library).name
            if beside.is_file():
                try:
                    names.extend(_MTL_MAP.findall(beside.read_text(encoding="utf-8", errors="replace")))
                except OSError:
                    continue

    seen: list[str] = []
    for name in names:
        cleaned = unquote(name.strip()).replace("\\", "/")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _sidecars(path: Path) -> dict[str, str]:
    """Hold the files a model names and answer where each is fetched.

    Args:
        path: The model on disk.

    Returns:
        ``{name: url}`` keyed on the file name as the model spells it. A name that does not
        resolve beside the model is left out and logged.
    """
    found: dict[str, str] = {}
    total = 0
    for name in _referenced(path):
        if len(found) >= MAX_SIDECARS:
            logger.warning("%s names more than %d files; the rest are not held", path.name, MAX_SIDECARS)
            break
        # A name is taken as a leaf beside the model, so nothing reaches out of the folder.
        item = path.parent / Path(name).name
        if not item.is_file():
            logger.warning("%s names %s, which is not beside it, so it will be missing", path.name, name)
            continue
        content_type = SIDECARS.get(item.suffix.lower())
        if content_type is None:
            continue
        size = item.stat().st_size
        if total + size > MAX_SIDECAR_BYTES:
            logger.warning(
                "the files %s names pass %d MB, so %s and any after it are not held",
                path.name, MAX_SIDECAR_BYTES // 1048576, item.name,
            )
            break
        key = three_asset.keep(item.read_bytes(), content_type)
        found[name] = "%s?key=%s" % (three_asset.ROUTE, key)
        total += size
    return found


def carried(path: Path) -> tuple[str, str, dict[str, str]]:
    """Hold one model file for the browser and answer where to fetch it.

    Args:
        path: The model on disk, already resolved through the containment layer.

    Returns:
        ``(url, format, sidecars)``: the address the browser fetches, the suffix without its
        dot, and ``{name: url}`` for the files beside it that it may name.

    Raises:
        ValueError: The suffix has no loader, or the file is larger than :data:`MAX_BYTES`.
        OSError: The file could not be read.
    """
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(
            f"{path.name} is a {suffix or 'suffixless'} file and the viewer reads "
            f"{', '.join(SUFFIXES)}. Convert it, or load it with Load 3D and save a .glb."
        )

    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(
            f"{path.name} is {size / 1048576:.0f} MB and the limit is "
            f"{MAX_BYTES // 1048576} MB. A model that large will not draw smoothly in a "
            f"browser either; decimate it first."
        )

    body = path.read_bytes()
    key = three_asset.keep(body, FORMATS[suffix])
    sidecars = _sidecars(path) if suffix not in (".glb", ".fbx", ".stl", ".3mf") else {}
    logger.info(
        "%s is held for the browser as %s, %.1f KB, with %d file(s) beside it",
        path.name, key, size / 1024.0, len(sidecars),
    )
    return "%s?key=%s" % (three_asset.ROUTE, key), suffix.lstrip("."), sidecars
