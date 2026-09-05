"""JSON state files into the sqlite store.

A store imports once, its marker committing with its rows. The JSON is read, never
written or removed. A truncated file is salvaged to its last complete entry.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .. import log
from . import store as store_module

__all__ = [
    "JSON_SOURCES",
    "KIND_FIELDS",
    "KIND_KV",
    "KIND_LISTS",
    "KIND_TERMS",
    "ImportReport",
    "import_pending",
    "import_store",
    "salvage_json",
]

logger = log.get_logger("state.migration")

#: A file of ``{category: {key: value}}``.
KIND_KV = "kv"

#: A file of ``{name: {field: value}}``, one flat list of records with bodies.
KIND_FIELDS = "fields"

#: A file of ``{category: [name, ...]}``, ordered records that are only names.
KIND_TERMS = "terms"

#: A file of ``{group: {list name: [item, ...]}}``, one ordered record list per key.
KIND_LISTS = "lists"

#: Each store, the file it is imported from, and the shape of that file.
JSON_SOURCES = (
    (store_module.SETTINGS, "was_suite_settings.json", KIND_KV),
    (store_module.HISTORY, "was_history.json", KIND_LISTS),
    (store_module.STYLES, "styles.json", KIND_FIELDS),
    (store_module.NSP, "nsp_pantry.json", KIND_TERMS),
)

#: Meta table key holding one store's import record.
MARKER = "import:{}"

#: Whitespace JSON allows between tokens.
SPACE = " \t\r\n"

#: What ends a bare number, ``true``, ``false`` or ``null``.
LITERAL_END = " \t\r\n,}]"


class ImportReport(NamedTuple):
    """What one store's import did.

    Attributes:
        store: The store name.
        source: The JSON file that was read.
        imported: Whether rows were written.
        categories: Categories written.
        entries: Keys or records written.
        note: What happened, for a log line. ``None`` after a clean full import.
    """

    store: str
    source: Path
    imported: bool
    categories: int
    entries: int
    note: str | None


def _string_end(text: str, start: int) -> int | None:
    """The offset just past a JSON string, or ``None`` when it is unterminated.

    Args:
        text: The document.
        start: Offset of the opening quote.

    Returns:
        The offset just past the closing quote.
    """
    index = start + 1
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return None


def _literal_end(text: str, start: int) -> int | None:
    """The offset just past a number or keyword, or ``None`` when it runs to the end.

    Args:
        text: The document.
        start: Offset of the first character of the literal.

    Returns:
        The offset of the character that ends it.
    """
    index = start
    length = len(text)
    while index < length and text[index] not in LITERAL_END:
        index += 1
    return None if index >= length else index


def _repair(text: str) -> tuple[str, int] | None:
    """A truncated document cut back to its last complete entry and closed off.

    Args:
        text: The document.

    Returns:
        ``(repaired document, offset kept)``, or ``None`` when there is nothing to close.
    """
    frames: list[list] = []
    index = 0
    length = len(text)
    root_end = None
    while index < length:
        char = text[index]
        if char in SPACE:
            index += 1
            continue
        if char in "{[":
            frames.append([char, index + 1, "key" if char == "{" else "element"])
            index += 1
            continue
        if char in "}]":
            if not frames:
                break
            frames.pop()
            index += 1
            if not frames:
                root_end = index
                break
            frames[-1][1] = index
            continue
        if char == ":":
            if frames and frames[-1][0] == "{":
                frames[-1][2] = "value"
            index += 1
            continue
        if char == ",":
            if frames and frames[-1][0] == "{":
                frames[-1][2] = "key"
            index += 1
            continue
        if char == '"':
            end = _string_end(text, index)
            if end is None:
                break
            # A string in key position is not an entry on its own.
            if frames and not (frames[-1][0] == "{" and frames[-1][2] == "key"):
                frames[-1][1] = end
            index = end
            continue
        end = _literal_end(text, index)
        if end is None:
            break
        if frames:
            frames[-1][1] = end
        index = end

    if root_end is not None:
        return text[:root_end], root_end
    if not frames:
        return None
    kept = frames[-1][1]
    closers = "".join("}" if frame[0] == "{" else "]" for frame in reversed(frames))
    return text[:kept] + closers, kept


def salvage_json(text: str) -> tuple[Any, str | None]:
    """Parse JSON, recovering as much of a truncated document as still parses.

    Args:
        text: The file's contents.

    Returns:
        ``(data, note)``. ``note`` is ``None`` when the whole document parsed, and
        otherwise says what was dropped. ``data`` is ``None`` when nothing parsed at all.
    """
    body = text.lstrip("\ufeff").rstrip("\x00")
    try:
        return json.loads(body), None
    except ValueError as error:
        first = error
    repaired = _repair(body)
    if repaired is None:
        return None, f"it is not JSON at all ({first})"
    document, kept = repaired
    try:
        data = json.loads(document)
    except ValueError as error:
        return None, f"nothing in it could be read ({error})"
    where = getattr(first, "pos", None)
    dropped = len(body) - kept
    return data, (
        f"it stops being valid JSON at byte {where} of {len(body)}, so the last "
        f"{dropped} byte(s) were dropped and everything before them was kept"
    )


def _shape_kv(data: Mapping) -> tuple[dict, list[str]]:
    """``{category: {key: value}}`` from a settings or history file.

    Args:
        data: The parsed document.

    Returns:
        ``(shaped, skipped)``, ``skipped`` naming every category that was not an object.
    """
    shaped: dict[str, dict] = {}
    skipped: list[str] = []
    for category, entries in data.items():
        if isinstance(entries, Mapping):
            shaped[str(category)] = dict(entries)
        else:
            skipped.append(str(category))
    return shaped, skipped


def _shape_fields(data: Mapping) -> tuple[dict, list[str]]:
    """One flat list of records with bodies, from a style library.

    Args:
        data: The parsed document.

    Returns:
        ``(shaped, skipped)``, ``skipped`` naming every entry that was not an object.
    """
    rows: list[tuple[str, dict | None]] = []
    skipped: list[str] = []
    for name, body in data.items():
        if isinstance(body, Mapping):
            rows.append((str(name), dict(body)))
        else:
            skipped.append(str(name))
    return {store_module.DEFAULT_CATEGORY: rows}, skipped


def _shape_terms(data: Mapping) -> tuple[dict, list[str]]:
    """``{category: [(term, None), ...]}`` from a terminology pantry.

    Args:
        data: The parsed document.

    Returns:
        ``(shaped, skipped)``, ``skipped`` naming every category or entry that was not a
        list of strings.
    """
    shaped: dict[str, list[tuple[str, dict | None]]] = {}
    skipped: list[str] = []
    for category, terms in data.items():
        if not isinstance(terms, list):
            skipped.append(str(category))
            continue
        rows: list[tuple[str, dict | None]] = []
        for index, term in enumerate(terms):
            if isinstance(term, str):
                rows.append((term, None))
            else:
                skipped.append(f"{category}[{index}]")
        shaped[str(category)] = rows
    return shaped, skipped


def _shape_lists(data: Mapping) -> tuple[dict, list[str]]:
    """``{list name: [(item, None), ...]}`` from a file of grouped lists.

    Args:
        data: The parsed document.

    Returns:
        ``(shaped, skipped)``, ``skipped`` naming every group or list that was not shaped
        like the rest. A list name repeated across groups keeps both groups' items.
    """
    shaped: dict[str, list[tuple[str, dict | None]]] = {}
    skipped: list[str] = []
    for group, lists in data.items():
        if not isinstance(lists, Mapping):
            skipped.append(str(group))
            continue
        for key, items in lists.items():
            if not isinstance(items, list):
                skipped.append(f"{group}/{key}")
                continue
            rows = shaped.setdefault(str(key), [])
            for index, item in enumerate(items):
                if isinstance(item, str):
                    rows.append((item, None))
                else:
                    skipped.append(f"{group}/{key}[{index}]")
    return shaped, skipped


SHAPERS = {
    KIND_KV: _shape_kv,
    KIND_FIELDS: _shape_fields,
    KIND_TERMS: _shape_terms,
    KIND_LISTS: _shape_lists,
}


def import_pending(database, directory) -> list[ImportReport]:
    """Import every store that has a JSON file and has not been imported yet.

    Args:
        database: The :class:`~modules.state.store.StateStore` to import into.
        directory: Where the JSON files live.

    Returns:
        One :class:`ImportReport` per store, in :data:`JSON_SOURCES` order.
    """
    root = Path(directory)
    reports = [
        import_store(database, name, root / filename, kind)
        for name, filename, kind in JSON_SOURCES
    ]
    if any(report.imported for report in reports):
        database.optimize()
    return reports


def import_store(database, name: str, source, kind: str) -> ImportReport:
    """Import one JSON file into one store, at most once.

    Args:
        database: The :class:`~modules.state.store.StateStore` to import into.
        name: The store name, one of :data:`~modules.state.store.STORES`.
        source: The JSON file to read. It is never written to.
        kind: :data:`KIND_KV`, :data:`KIND_FIELDS` or :data:`KIND_TERMS`.

    Returns:
        What was done, as an :class:`ImportReport`.
    """
    source = Path(source)
    key = MARKER.format(name)
    nothing = ImportReport(name, source, False, 0, 0, None)

    if database.meta(key) is not None:
        return nothing._replace(note="already imported")
    if not source.is_file():
        return nothing._replace(note="no file to import")
    empty = database.is_empty(name)
    if empty is None:
        return nothing._replace(note="the store could not be read")
    if not empty:
        database.set_meta(key, _marker(source, 0, 0, "the store already held data"))
        return nothing._replace(note="the store already held data")

    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        logger.warning(
            "%s could not be read (%s), so the %s store starts empty. The file is left "
            "where it is.",
            source,
            error,
            name,
        )
        return nothing._replace(note=f"unreadable ({error})")

    data, salvaged = salvage_json(text)
    if data is None:
        logger.warning(
            "%s could not be parsed: %s. The %s store starts empty and the file is left "
            "where it is, so nothing has been lost.",
            source,
            salvaged,
            name,
        )
        return nothing._replace(note=salvaged)
    if not isinstance(data, Mapping):
        logger.warning(
            "%s holds a %s where an object was expected, so the %s store starts empty. "
            "The file is left where it is.",
            source,
            type(data).__name__,
            name,
        )
        return nothing._replace(note="not a JSON object")

    shaped, skipped = SHAPERS[kind](data)
    counts: dict[str, int] = {"categories": 0, "entries": 0}
    raced = {"value": False}

    def work(connection):
        row = connection.execute("SELECT 1 FROM meta WHERE key = ?", (key,)).fetchone()
        if row is not None:
            raced["value"] = True
            return
        writer = store_module.write_kv if kind == KIND_KV else store_module.write_records
        categories, entries = writer(connection, name, shaped)
        counts["categories"] = categories
        counts["entries"] = entries
        connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            (key, _marker(source, categories, entries, salvaged)),
        )

    if not database.write(work, f"importing {source.name}"):
        return nothing._replace(note="the import could not be written")
    if raced["value"]:
        return nothing._replace(note="already imported")

    if salvaged:
        logger.warning(
            "%s was imported into the %s store, but %s. %s category(s) and %s entry(s) "
            "were recovered and the file is left exactly where it is.",
            source.name,
            name,
            salvaged,
            counts["categories"],
            counts["entries"],
        )
    else:
        logger.info(
            "imported %s category(s) and %s entry(s) from %s into the %s store; the file "
            "is left where it is",
            counts["categories"],
            counts["entries"],
            source.name,
            name,
        )
    if skipped:
        logger.warning(
            "%s entry(s) in %s were not shaped like the rest and were not imported: %s",
            len(skipped),
            source.name,
            ", ".join(skipped[:10]) + (", ..." if len(skipped) > 10 else ""),
        )

    return ImportReport(name, source, True, counts["categories"], counts["entries"], salvaged)


def _marker(source: Path, categories: int, entries: int, note: str | None) -> str:
    """The meta table value recording one import.

    Args:
        source: The file that was read.
        categories: Categories written.
        entries: Keys or records written.
        note: What was salvaged or why nothing was written.

    Returns:
        JSON text.
    """
    try:
        size = source.stat().st_size
        modified = source.stat().st_mtime
    except OSError:
        size, modified = -1, 0.0
    return json.dumps(
        {
            "source": str(source),
            "size": size,
            "modified": modified,
            "categories": categories,
            "entries": entries,
            "note": note,
            "at": time.time(),
            "schema_version": store_module.SCHEMA_VERSION,
        }
    )
