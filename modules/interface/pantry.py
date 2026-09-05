"""The Noodle Soup Prompts pantry as JSON a node interface draws.

``GET /was/interface/api/nsp_pantry`` answers the terminology names with their counts, one
terminology's words on ``term``, or the words matching ``search``. One answer holds at most
:data:`MAX_ROWS` rows of :data:`MAX_ROW_CHARS` characters.
"""

from __future__ import annotations

from .. import log
from ..prompt import nsp
from .channel import NO_STORE

__all__ = [
    "MATCH_CEILING",
    "MAX_ROWS",
    "MAX_ROW_CHARS",
    "MAX_SEARCH_CHARS",
    "ROUTE",
    "listing_payload",
    "register_routes",
    "search_payload",
    "term_payload",
]

logger = log.get_logger("interface.pantry")

#: The one route serving the pantry.
ROUTE = "/was/interface/api/nsp_pantry"

#: Rows in one answer, and the largest ``limit`` a request can ask for.
MAX_ROWS = 500

#: Characters of one word that are returned. The longest published word is 163, so this cuts
#: nothing the pantry holds and bounds a word somebody pasted a paragraph into.
MAX_ROW_CHARS = 400

#: Characters of a search term that are read. Longer than any published word's own words.
MAX_SEARCH_CHARS = 64

#: Matches counted at all. Past this the answer says it was truncated and counts this many.
MATCH_CEILING = 10_000

_registered = False


def listing_payload() -> dict:
    """Every terminology name with its counts, as the object the route answers with.

    Never raises.

    Returns:
        ``{"terms", "entries", "generation"}``. ``terms`` carries ``name``, ``entries`` and
        ``own`` per terminology in pantry order, ``entries`` is how many words the whole
        pantry holds, and ``generation`` is the stamp that moves on every pantry write.
    """
    try:
        counts = nsp.terms()
        local = nsp.local_counts()
        stamp = nsp.generation()
    except Exception as error:
        # A pantry nobody can read is an empty panel, never a failed request: the picked
        # box beside it still holds whatever was typed into it.
        logger.debug("%s could not read the pantry (%s)", ROUTE, error)
        return {"terms": [], "entries": 0, "generation": ""}
    return {
        "terms": [
            {"name": name, "entries": int(total), "own": int(local.get(name, 0))}
            for name, total in counts.items()
        ],
        "entries": sum(counts.values()),
        "generation": stamp,
    }


def term_payload(term, start=0, limit=MAX_ROWS) -> dict:
    """One terminology's words, as the object the route answers with.

    Never raises.

    Args:
        term: The terminology name. One the pantry does not hold answers no words.
        start: First word of the window, counting from the start of the terminology.
        limit: How many words the window holds, clamped to :data:`MAX_ROWS`.

    Returns:
        ``{"name", "start", "entries", "total", "truncated", "generation"}``. Each entry
        carries ``text`` and ``own``, ``total`` is how many words the terminology holds,
        and ``truncated`` says whether the window stops short of the end.
    """
    name = str(term or "")
    window = _bounded(limit, MAX_ROWS)
    first = max(0, _whole(start, 0))
    try:
        total = int(nsp.terms().get(name, 0))
        rows = nsp.term_page(name, first, window) if total else []
        stamp = nsp.generation()
    except Exception as error:
        logger.debug("%s could not read the %s terminology (%s)", ROUTE, name, error)
        return _empty_term(name, first)
    return {
        "name": name,
        "start": first,
        "entries": [{"text": text[:MAX_ROW_CHARS], "own": bool(mine)} for text, mine in rows],
        "total": total,
        "truncated": first + len(rows) < total,
        "generation": stamp,
    }


def search_payload(search, start=0, limit=MAX_ROWS) -> dict:
    """The words matching a search, as the object the route answers with.

    Never raises.

    Args:
        search: Text matched anywhere in a word, cut to :data:`MAX_SEARCH_CHARS`.
        start: How many matches to pass over first.
        limit: How many matches the window holds, clamped to :data:`MAX_ROWS`.

    Returns:
        ``{"search", "start", "matches", "total", "truncated", "generation"}``. Each match
        carries ``term``, ``text`` and ``own``, ``total`` is the match count and stops at
        :data:`MATCH_CEILING`, and ``truncated`` says whether the count stopped there.
    """
    needle = str(search or "")[:MAX_SEARCH_CHARS]
    window = _bounded(limit, MAX_ROWS)
    first = max(0, _whole(start, 0))
    try:
        rows, total = nsp.search_entries(needle, first, window, MATCH_CEILING)
        stamp = nsp.generation()
    except Exception as error:
        logger.debug("%s could not search the pantry (%s)", ROUTE, error)
        return {
            "search": needle,
            "start": first,
            "matches": [],
            "total": 0,
            "truncated": False,
            "generation": "",
        }
    return {
        "search": needle,
        "start": first,
        "matches": [
            {"term": term, "text": text[:MAX_ROW_CHARS], "own": bool(mine)}
            for term, text, mine in rows
        ],
        "total": total,
        "truncated": total >= MATCH_CEILING,
        "generation": stamp,
    }


def register_routes() -> bool:
    """Register the route serving the pantry.

    Returns:
        True when the route was registered. False when it was registered already, or when
        the server could not be reached, in which case a panel asking for the pantry gets a
        failed request and draws what the widget holds.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_nsp_pantry(request):
            search = request.query.get("search")
            term = request.query.get("term")
            start = _int(request.query.get("start"), 0)
            limit = _int(request.query.get("limit"), MAX_ROWS)
            if search:
                payload = search_payload(search, start=start, limit=limit)
            elif term:
                payload = term_payload(term, start=start, limit=limit)
            else:
                payload = listing_payload()
            return web.json_response(payload, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking for the terminology "
            "pantry gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the Noodle Soup Prompts pantry", ROUTE)
    return True


def _empty_term(name: str, start: int) -> dict:
    """The answer for a terminology whose words could not be read."""
    return {
        "name": name,
        "start": start,
        "entries": [],
        "total": 0,
        "truncated": False,
        "generation": "",
    }


def _bounded(value, ceiling: int) -> int:
    """A requested window size held between one row and ``ceiling``."""
    return max(1, min(_whole(value, ceiling), ceiling))


def _whole(value, default: int) -> int:
    """A value as an integer, or ``default`` when it is not one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int(value, default):
    """A query value as an integer, or ``default`` when it is missing or not one."""
    if value is None:
        return default
    return _whole(str(value).strip(), default)
