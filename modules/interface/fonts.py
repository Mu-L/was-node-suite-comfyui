"""A font the pack draws with, as the bytes a browser can load.

``GET /was/interface/api/font?name=<font name>`` answers the file
:func:`modules.data.paths.font_file` maps that name to, or 404. Bytes are held in memory,
:data:`MAX_BYTES` for one font and :data:`MAX_CACHED_BYTES` in all.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from .. import log
from ..data import paths
from ..util.sandbox import contains
from .channel import NO_STORE

__all__ = [
    "CACHE",
    "DEFAULT_MEDIA_TYPE",
    "MAX_AGE",
    "MAX_BYTES",
    "MAX_CACHED_BYTES",
    "MEDIA_TYPES",
    "ROUTE",
    "font_payload",
    "font_roots",
    "media_type",
    "register_routes",
]

logger = log.get_logger("interface.fonts")

#: The one route serving a font, keyed by the ``name`` a widget stores. Named for the asset
#: rather than for a node.
ROUTE = "/was/interface/api/font"

#: Bytes of one font that are served. Every bundled face is under a megabyte, a CJK
#: collection can be twenty times this, and half a font is not a font, so one over the budget
#: is refused rather than cut.
MAX_BYTES = 8 * 1024 * 1024

#: Bytes of font held in memory across every name. This process outlives every prompt, so
#: the store is bounded by the memory it costs rather than by a count of files.
MAX_CACHED_BYTES = 16 * 1024 * 1024

#: Seconds the browser is told it may hold a font, which is what stops every node drawing
#: text refetching the same file on every page load. The ``ETag`` travels with it, so a font
#: replaced on disk is picked up on the first request after this rather than never.
MAX_AGE = 3600

#: Sent with a font. The bytes behind a name are the typeface, so a copy the browser holds
#: says nothing untrue. A thumbnail and a file's lines are the opposite case, where a held
#: copy claims to be the current state of a graph or a disk, and those are sent no-store.
CACHE = {"Cache-Control": f"private, max-age={MAX_AGE}"}

#: Answered for a name the catalog does not hold, which is every path.
UNLISTED = "that font is not one this pack lists"

#: Answered for a catalogued font that could not be read or is over :data:`MAX_BYTES`.
UNREADABLE = "that font could not be read"

#: Media type per font suffix, from RFC 8081. A suffix in
#: :data:`modules.data.paths.FONT_SUFFIXES` with no entry here is still served, since
#: ``FontFace`` reads the bytes rather than the header.
MEDIA_TYPES = {".ttf": "font/ttf", ".otf": "font/otf", ".ttc": "font/collection"}

#: Sent for a font suffix :data:`MEDIA_TYPES` does not name.
DEFAULT_MEDIA_TYPE = "application/octet-stream"

#: A font's resolved path to that font's ``(revision, bytes)``, most recently served last.
#: Keyed on the path rather than on the name, so two names for one file cost one copy and a
#: catalog that has moved on cannot serve one font's bytes under another's name.
_fonts: OrderedDict[str, tuple[str, bytes]] = OrderedDict()

#: Serializes the reads and the store, so several requests arriving together cost one file
#: read at a time rather than one each. The route runs on the server's thread while a prompt
#: may be running.
_lock = threading.Lock()

_registered = False


def font_roots() -> list[Path]:
    """The directories a served font may sit in.

    Returns:
        ``modules/data/fonts`` and ``<config dir>/fonts``, resolved, and only the ones that
        resolved. Neither is required to exist.
    """
    found = []
    for root in (paths.data_directory() / paths.FONT_DIR, paths.user_font_directory()):
        if root is None:
            continue
        try:
            found.append(root.resolve())
        except (OSError, ValueError):
            logger.debug("a font directory could not be resolved", exc_info=True)
    return found


def media_type(path: Path) -> str:
    """The media type a font file is served as.

    Args:
        path: The font file, whose suffix is read.

    Returns:
        The :data:`MEDIA_TYPES` entry for that suffix, or :data:`DEFAULT_MEDIA_TYPE`.
    """
    return MEDIA_TYPES.get(path.suffix.lower(), DEFAULT_MEDIA_TYPE)


def font_payload(name) -> tuple[bytes, str, str] | None:
    """One catalogued font's bytes, its media type, and the revision it was read at.

    Args:
        name: The exact name a ``font`` widget stores, as :func:`modules.data.paths.font_names`
            offers it. Anything else, a path included, answers None.

    Returns:
        ``(bytes, media type, revision)``, or None when the name is not in the catalog, when
        what it maps to is not a font file inside a :func:`font_roots` directory, when the
        file is empty or gone, or when it is larger than :data:`MAX_BYTES`. The revision
        changes whenever the file does and is what the ``ETag`` is built from.
    """
    path = _catalogued(name)
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError as error:
        logger.debug("%s could not read the font %r (%s)", ROUTE, name, error)
        return None
    if not stat.st_size:
        logger.debug("%s refused the font %r: the file is empty", ROUTE, name)
        return None
    if stat.st_size > MAX_BYTES:
        logger.debug(
            "%s refused the font %r: %d bytes, over the budget of %d",
            ROUTE, name, stat.st_size, MAX_BYTES,
        )
        return None
    revision = f"{stat.st_size}:{stat.st_mtime_ns}"
    key = str(path)
    with _lock:
        held = _fonts.get(key)
        if held is not None and held[0] == revision:
            _fonts.move_to_end(key)
            return held[1], media_type(path), revision
        try:
            data = path.read_bytes()
        except OSError as error:
            logger.debug("%s could not read the font %r (%s)", ROUTE, name, error)
            return None
        # The file can grow between the stat and the read, so the budget is applied to what
        # was actually read as well as to what was measured.
        if len(data) > MAX_BYTES:
            logger.debug(
                "%s refused the font %r: it grew to %d bytes while it was being read",
                ROUTE, name, len(data),
            )
            return None
        _hold(key, revision, data)
    return data, media_type(path), revision


def register_routes() -> bool:
    """Register the route serving a catalogued font.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case an interface asking for a font gets a
        failed request.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_font(request):
            name = request.query.get("name")
            payload = font_payload(name)
            if payload is None:
                # A name nobody listed and a catalogued file that could not be read both
                # answer 404. The two messages separate a menu that has moved on from a font
                # that has, and neither carries a path.
                return web.Response(
                    status=404,
                    text=UNREADABLE if _listed(name) else UNLISTED,
                    headers=NO_STORE,
                )
            data, media, revision = payload
            tag = f'"{revision}"'
            headers = {**CACHE, "ETag": tag}
            if _unchanged(request.headers.get("If-None-Match"), tag):
                return web.Response(status=304, headers=headers)
            return web.Response(body=data, content_type=media, headers=headers)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking for the font its node "
            "renders with gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving fonts", ROUTE)
    return True


def _catalogued(name) -> Path | None:
    """The file a font name maps to, or None when this route may not serve it.

    Args:
        name: The raw query value.

    Returns:
        The resolved path, or None when the name is not a key of the catalog or resolves
        somewhere other than a font file inside a :func:`font_roots` directory.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        # A mapping lookup, not a join: the key is compared whole, so a separator, a drive, a
        # leading root, a '..' segment and a NUL byte are all names the catalog does not hold
        # rather than paths this reads.
        path = paths.font_file(name).resolve()
    except ValueError:
        logger.debug("%s was asked for a font the catalog does not hold", ROUTE)
        return None
    except OSError:
        logger.debug("%s could not resolve a catalogued font", ROUTE, exc_info=True)
        return None
    if not _permitted(path):
        logger.debug("%s refused a font outside every font directory", ROUTE)
        return None
    return path


def _permitted(path: Path) -> bool:
    """Whether a resolved path is a font file this route may serve.

    Args:
        path: A resolved path the catalog answered with.

    Returns:
        True for a font suffix inside a :func:`font_roots` directory, and for the legacy font
        that sits beside those directories rather than in one.
    """
    if path.suffix.lower() not in paths.FONT_SUFFIXES:
        return False
    if any(contains(root, path) for root in font_roots()):
        return True
    legacy = _legacy_file()
    return legacy is not None and contains(legacy, path)


def _legacy_file() -> Path | None:
    """The v2 font, which sits in the data directory rather than under ``fonts/``."""
    try:
        return paths.font_file(paths.LEGACY_FONT).resolve()
    except (OSError, ValueError):
        logger.debug("the legacy font could not be resolved", exc_info=True)
        return None


def _listed(name) -> bool:
    """Whether a name is a key of the catalog, which separates the two refusals."""
    if not isinstance(name, str):
        return False
    try:
        paths.font_file(name)
    except ValueError:
        return False
    return True


def _hold(key: str, revision: str, data: bytes) -> None:
    """Store one font's bytes, with :data:`_lock` held by the caller.

    Args:
        key: The resolved path, as the store's key.
        revision: The size and modification time the bytes were read at.
        data: The bytes.
    """
    # Reinserted rather than assigned, so serving a font again makes it the most recent and
    # the oldest is the one evicted.
    _fonts.pop(key, None)
    _fonts[key] = (revision, data)
    total = sum(len(held) for _, held in _fonts.values())
    while len(_fonts) > 1 and total > MAX_CACHED_BYTES:
        _, evicted = _fonts.popitem(last=False)
        total -= len(evicted[1])


def _unchanged(header, tag: str) -> bool:
    """Whether an ``If-None-Match`` header already carries the tag of what would be sent.

    Args:
        header: The raw header value, or None when the request carries none.
        tag: The ``ETag`` this answer would be sent with.

    Returns:
        True when the browser holds this revision, which is answered 304 and no bytes. A weak
        validator counts: the bytes behind one revision do not change.
    """
    if not header:
        return False
    for candidate in str(header).split(","):
        entry = candidate.strip()
        if entry.startswith("W/"):
            entry = entry[2:].strip()
        if entry in ("*", tag):
            return True
    return False
