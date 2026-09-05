"""Which files inside an archive this pack has a loader for.

An entry's extension decides its kind: :data:`IMAGE`, :data:`TEXT`, :data:`DOCUMENT`, or no
kind at all.
"""

from __future__ import annotations

from ..constants import ALLOWED_EXT
from ..document.container import SUFFIX as DOCUMENT_SUFFIX
from ..util.text_files import TEXT_EXTENSIONS

__all__ = [
    "DOCUMENT",
    "EXTENSIONS",
    "IMAGE",
    "KINDS",
    "SUPPORTED",
    "TEXT",
    "extension_list",
    "kind_of",
    "supported",
]

#: The three kinds of file the pack can load out of an archive.
IMAGE = "image"
TEXT = "text"
DOCUMENT = "document"

#: The kinds, in the order a report lists them.
KINDS: tuple[str, ...] = (IMAGE, TEXT, DOCUMENT)

#: The extensions each kind covers, lowercased and each with its leading dot.
EXTENSIONS: dict[str, tuple[str, ...]] = {
    IMAGE: tuple(sorted(ALLOWED_EXT)),
    TEXT: tuple(sorted(TEXT_EXTENSIONS)),
    DOCUMENT: (DOCUMENT_SUFFIX,),
}

#: Every extension that has a kind, so ``.zip`` is absent: an archive inside an archive is
#: skipped like any other unsupported file, which is what stops a nested archive being
#: opened at any depth.
SUPPORTED: frozenset[str] = frozenset(
    extension for group in EXTENSIONS.values() for extension in group
)


def kind_of(name: str) -> str | None:
    """Which kind of file a name is.

    Args:
        name: An entry name or a file name. Only its extension is read, so a path is fine.

    Returns:
        :data:`IMAGE`, :data:`TEXT`, :data:`DOCUMENT`, or ``None`` where the pack has no
        loader for it, which covers a file with no extension at all.
    """
    lowered = str(name).lower()
    for kind in KINDS:
        if lowered.endswith(EXTENSIONS[kind]):
            return kind
    return None


def supported(name: str) -> bool:
    """Whether a name is one of the kinds this pack loads."""
    return kind_of(name) is not None


def extension_list(kind: str | None = None) -> str:
    """The extensions of one kind, or of all of them, for a message.

    Args:
        kind: One of :data:`KINDS`, or ``None`` for every supported extension.

    Returns:
        The extensions in one comma-separated string, sorted, such as ``.md, .txt``.
    """
    group = EXTENSIONS.get(kind) if kind is not None else tuple(sorted(SUPPORTED))
    return ", ".join(group or ())
