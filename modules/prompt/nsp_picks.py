"""A chosen set of Noodle Soup Prompts words, one pick per line.

``artist: greg rutkowski`` is one word, ``artist: *`` or a bare ``artist`` a whole
terminology. Blank lines and ``#`` comments are dropped, order is kept.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import NamedTuple

from . import nsp
from .. import log
from ..util.text_files import is_comment

__all__ = ["MAX_LINES", "Pick", "Resolved", "WHOLE", "parse", "resolve", "term_names"]

logger = log.get_logger("prompt.nsp_picks")

#: How many lines of a pick list are read. Above this the rest is reported and ignored, so a
#: widget somebody pasted a whole pantry into is bounded before anything is looked up.
MAX_LINES = 8192

#: The word standing for every word of a terminology.
WHOLE = "*"

#: What divides a terminology name from one of its words.
SEPARATOR = ":"


class Pick(NamedTuple):
    """One line of a pick list.

    Attributes:
        term: The terminology named.
        entry: One word of it, or an empty string for the whole terminology.
        line: The line as written.
    """

    term: str
    entry: str
    line: str


class Resolved(NamedTuple):
    """What a pick list came to, read against the pantry.

    Attributes:
        entries: The words, in pick order, repeats dropped.
        own: How many of them were added from a node or a file.
        missing: Picked words the pantry no longer holds, in pick order.
        unknown: Terminologies taken whole that the pantry does not hold.
        empty: Terminologies taken whole that hold no word.
        repeats: Picks dropped for naming a word already taken.
    """

    entries: list[str]
    own: int
    missing: list[str]
    unknown: list[str]
    empty: list[str]
    repeats: int


def parse(text: str, known: Iterable[str] = ()) -> tuple[list[Pick], int]:
    """The picks a widget holds, in the order they are written.

    Args:
        text: The widget's value.
        known: The terminology names the pantry holds, which is what resolves a colon
            inside a name.

    Returns:
        ``(picks, overflow)``. Blank lines and comment lines are dropped, and ``overflow``
        counts the lines past :data:`MAX_LINES` that were not read.
    """
    names = {str(name) for name in known}
    picks: list[Pick] = []
    lines = str(text or "").splitlines()
    for line in lines[:MAX_LINES]:
        entry = line.strip()
        if not entry or is_comment(entry):
            continue
        picks.append(_read(entry, names))
    overflow = max(0, len(lines) - MAX_LINES)
    if overflow:
        logger.warning(
            "the pick list holds more than %d lines; the last %d were not read",
            MAX_LINES,
            overflow,
        )
    return picks, overflow


def _read(line: str, names: set[str]) -> Pick:
    """One trimmed line as a pick.

    Args:
        line: The line, already trimmed and known to carry something.
        names: The terminology names the pantry holds.

    Returns:
        The pick. A line whose whole text names a terminology takes it whole, otherwise the
        split falls at the first colon whose prefix names one, or at the first colon.
    """
    if line in names:
        return Pick(line, "", line)
    at = -1
    while True:
        at = line.find(SEPARATOR, at + 1)
        if at < 0:
            break
        if line[:at].strip() in names:
            return _split(line, at)
    first = line.find(SEPARATOR)
    if first < 0:
        return Pick(line, "", line)
    return _split(line, first)


def _split(line: str, at: int) -> Pick:
    """One line divided at a colon.

    Args:
        line: The line, already trimmed.
        at: Where the dividing colon sits.

    Returns:
        The pick, with ``entry`` empty where the word is :data:`WHOLE`.
    """
    term = line[:at].strip()
    entry = line[at + 1 :].strip()
    return Pick(term, "" if entry == WHOLE else entry, line)


def resolve(picks: Iterable[Pick], counts: Mapping[str, int]) -> Resolved:
    """The words a pick list comes to, read against the pantry.

    Args:
        picks: The picks, in order.
        counts: ``{term: entries}`` as the pantry holds it.

    Returns:
        The reading :class:`Resolved` describes. A word the pantry no longer holds is still
        emitted, carrying the text of the pick.
    """
    entries: list[str] = []
    seen: set[str] = set()
    whole: dict[str, dict[str, bool]] = {}
    single: dict[tuple[str, str], bool | None] = {}
    own = 0
    missing: list[str] = []
    unknown: list[str] = []
    empty: list[str] = []
    repeats = 0

    def take(word: str, mine: bool) -> None:
        nonlocal own, repeats
        if word in seen:
            repeats += 1
            return
        seen.add(word)
        entries.append(word)
        own += mine

    for pick in picks:
        term = pick.term
        if pick.entry:
            key = (term, pick.entry)
            if key not in single:
                single[key] = nsp.entry_mark(term, pick.entry)
            mark = single[key]
            if mark is None:
                missing.append(pick.entry)
            take(pick.entry, mark is True)
            continue
        if term not in counts:
            unknown.append(term)
            continue
        if term not in whole:
            whole[term] = dict(nsp.term_page(term, 0, int(counts[term])))
        held = whole[term]
        if not held:
            empty.append(term)
            continue
        for word, mine in held.items():
            take(word, mine)

    return Resolved(entries, own, missing, unknown, empty, repeats)


def term_names(counts: Mapping[str, int], local: Mapping[str, int]) -> tuple[list[str], int]:
    """The terminology names themselves, as the answer to a pick list with nothing in it.

    Args:
        counts: ``{term: entries}`` as the pantry holds it.
        local: ``{term: entries}`` counting only what was added from a node or a file.

    Returns:
        ``(names, own)``, in pantry order, ``own`` counting the terminologies holding a
        word that the published pantry did not supply.
    """
    names = list(counts)
    return names, sum(1 for name in names if local.get(name))
