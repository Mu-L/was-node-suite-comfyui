"""Splitting a prompt into tags, and tidying the result.

:func:`strip_emphasis` reduces ``(tag:1.4)``, ``((tag))`` and ``[tag]`` to ``tag``, so two
spellings of one tag compare equal. Which of a set of duplicates survives is
:func:`clean`'s ``keep`` argument.
"""

from __future__ import annotations

import re

__all__ = ["SORTS", "clean", "sort_tags", "split_tags", "strip_emphasis", "tag_key"]

#: ``(tag:1.4)``, the A1111 weighted form, which is also what ComfyUI's own encoder reads.
_WEIGHTED = re.compile(r"^\(\s*(?P<body>.+?)\s*:\s*-?\d+(?:\.\d+)?\s*\)$", re.DOTALL)

#: ``(tag)`` and ``[tag]``, the repeat-to-emphasise forms, one nesting level at a time.
_BRACKETED = re.compile(r"^\(\s*(?P<body>.+?)\s*\)$|^\[\s*(?P<alt>.+?)\s*\]$", re.DOTALL)

#: Any run of whitespace, including the line breaks a multi-line prompt box produces.
_WHITESPACE = re.compile(r"\s+")

#: How many times emphasis is unwrapped before the tag is taken as it is. Deep nesting is
#: not a thing a real prompt does, and the bound keeps a pathological string from looping.
_MAX_UNWRAP = 8

#: Sort name -> ``(key function, reverse)``. ``none`` is handled before this is consulted.
SORTS = {
    "a-z": (lambda tag: strip_emphasis(tag).casefold(), False),
    "z-a": (lambda tag: strip_emphasis(tag).casefold(), True),
    "shortest first": (len, False),
    "longest first": (len, True),
}


def strip_emphasis(tag: str) -> str:
    """Reduce a tag to its words, with emphasis markup removed.

    Args:
        tag: One tag, already split out of the prompt.

    Returns:
        The tag with whole-tag emphasis removed and surrounding space trimmed.
    """
    body = tag.strip()
    for _ in range(_MAX_UNWRAP):
        weighted = _WEIGHTED.match(body)
        if weighted:
            body = weighted.group("body").strip()
            continue
        bracketed = _BRACKETED.match(body)
        if bracketed:
            body = (bracketed.group("body") or bracketed.group("alt")).strip()
            continue
        break
    return body


def tag_key(tag: str, ignore_case: bool = True, ignore_emphasis: bool = True) -> str:
    """The value two tags are compared on when duplicates are removed.

    Args:
        tag: One tag.
        ignore_case: Whether ``Neon Glow`` and ``neon glow`` count as the same tag.
        ignore_emphasis: Whether ``(neon glow:1.4)`` counts as the same tag as
            ``neon glow``.

    Returns:
        The comparison key. Internal whitespace is collapsed either way, since ``neon
        glow`` and ``neon  glow`` are the same tag by any reading.
    """
    body = strip_emphasis(tag) if ignore_emphasis else tag.strip()
    body = _WHITESPACE.sub(" ", body)
    return body.casefold() if ignore_case else body


def split_tags(text: str, delimiter: str = ",") -> list[str]:
    """Split a prompt into tags.

    Args:
        text: The prompt.
        delimiter: What separates one tag from the next. An empty delimiter splits on
            whitespace, which turns the prompt into its words.

    Returns:
        The tags, in order, each trimmed of surrounding whitespace. Empty entries are
        kept; :func:`clean` decides whether an empty tag is dropped.
    """
    if not delimiter:
        return [part for part in _WHITESPACE.split(text.strip()) if part]
    return [part.strip() for part in text.split(delimiter)]


def sort_tags(tags: list[str], order: str) -> list[str]:
    """Order a list of tags.

    Args:
        tags: The tags.
        order: ``none`` to keep the order they arrived in, or a key of :data:`SORTS`. An
            unknown name keeps the order rather than raising, since it reaches here from a
            combo whose options may have grown since the workflow was saved.

    Returns:
        The tags in the requested order. Sorting is by the tag's words, so emphasis markup
        does not push a weighted tag away from its neighbours.
    """
    entry = SORTS.get(order)
    if entry is None:
        return list(tags)
    key, reverse = entry
    return sorted(tags, key=key, reverse=reverse)


def clean(
    text: str,
    delimiter: str = ",",
    dedupe: bool = True,
    ignore_case: bool = True,
    ignore_emphasis: bool = True,
    keep: str = "first",
    remove_empty: bool = True,
    collapse_whitespace: bool = True,
    order: str = "none",
    limit: int = 0,
) -> tuple[list[str], int]:
    """Split a prompt into tags and tidy them.

    Args:
        text: The prompt.
        delimiter: What separates one tag from the next. Empty splits on whitespace.
        dedupe: Whether repeated tags are reduced to one.
        ignore_case: Whether a case difference still counts as a duplicate.
        ignore_emphasis: Whether an emphasis difference still counts as a duplicate.
        keep: ``first`` or ``last``, which of a set of duplicates survives. Position is
            the earliest occurrence either way.
        remove_empty: Whether tags holding nothing are dropped.
        collapse_whitespace: Whether runs of whitespace inside a tag become one space.
        order: ``none`` or a key of :data:`SORTS`.
        limit: Keep at most this many tags, counted last. 0 keeps all.

    Returns:
        ``(tags, removed)``: the surviving tags in final order, and how many entries the
        tidy-up took out.
    """
    tags = split_tags(text, delimiter)
    original = len(tags)

    if collapse_whitespace:
        tags = [_WHITESPACE.sub(" ", tag).strip() for tag in tags]
    if remove_empty:
        tags = [tag for tag in tags if tag and strip_emphasis(tag)]

    if dedupe:
        position: dict[str, int] = {}
        chosen: dict[str, str] = {}
        for tag in tags:
            key = tag_key(tag, ignore_case, ignore_emphasis)
            if key not in position:
                position[key] = len(position)
                chosen[key] = tag
            elif keep == "last":
                chosen[key] = tag
        tags = [chosen[key] for key in sorted(position, key=position.get)]

    tags = sort_tags(tags, order)
    if limit > 0:
        tags = tags[:limit]
    return tags, original - len(tags)
