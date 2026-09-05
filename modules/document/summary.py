"""One document's metadata, read out for typed sockets and for reading.

:func:`fields` returns every value a metadata view emits, in the order ``meta.json`` writes
them. :func:`summary` renders the same values as text, one field to a line.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from .container import Document
from .metadata import Metadata

__all__ = [
    "AUTHORED_FIELDS",
    "Fields",
    "KEYWORD_SEPARATOR",
    "NONE_AT_ALL",
    "NOT_SET",
    "fields",
    "has_metadata",
    "keywords_text",
    "summary",
]

#: Written where a text field holds nothing. A document's metadata is read to find out what
#: is missing before it is exported, so an empty field is named rather than left blank.
NOT_SET = "(not set)"

#: Written where a collection holds nothing, such as the custom pairs of a document nobody
#: has added any to.
NONE_AT_ALL = "(none)"

#: What joins the keywords into the single string ODF, OOXML and PDF all spell them as.
KEYWORD_SEPARATOR = ", "

#: The fields an author fills in, which :func:`has_metadata` answers for. The timestamps and
#: the generator are left out: all three are written on every document this pack saves, so
#: counting them would answer true for every file it has ever produced.
AUTHORED_FIELDS = ("title", "description", "author", "copyright", "language", "keywords")

#: The label on each single-valued line of :func:`summary`, in the order ``meta.json`` writes
#: them, with the derived counts after the authored fields. Each is the name of the socket
#: carrying the same value, so the text says which output to reach for.
_LINE_LABELS = (
    "title",
    "description",
    "author",
    "copyright",
    "language",
    "keywords",
    "created",
    "modified",
    "generator",
    "word_count",
    "character_count",
    "asset_count",
)

#: Column the values start in: the longest label and two spaces, so every value lines up and
#: no label ever touches the value beside it.
_WIDTH = max(len(label) for label in _LINE_LABELS) + 2


class Fields(NamedTuple):
    """Every value a metadata view emits, in socket order.

    Attributes:
        title: What the document is called, empty when it carries no title.
        description: The sentence or two saying what the document is.
        author: Who wrote it.
        copyright: The rights statement, as free text.
        language: BCP 47 tag such as ``"en"`` or ``"pt-BR"``.
        keywords: The keywords joined by :data:`KEYWORD_SEPARATOR`.
        keywords_list: The same keywords as a list, in the order they were given.
        created: When the document was first made, in the container's stamp format.
        modified: When its content last changed, in the same format.
        generator: What produced it.
        custom: A copy of the author's own pairs, in the order the document holds them.
        word_count: Words in the content, counted now.
        character_count: Characters in the content, counted now.
        asset_count: How many files are embedded in the container.
        assets: Their names, sorted, each relative to ``assets/`` and spelled with ``/``.
        has_metadata: Whether any of :data:`AUTHORED_FIELDS` holds something.
        summary: The whole reading as text, from :func:`summary`.
    """

    title: str
    description: str
    author: str
    copyright: str
    language: str
    keywords: str
    keywords_list: list[str]
    created: str
    modified: str
    generator: str
    custom: dict[str, str]
    word_count: int
    character_count: int
    asset_count: int
    assets: list[str]
    has_metadata: bool
    summary: str


def fields(document: Document) -> Fields:
    """Read every value a metadata view emits.

    Args:
        document: The document to read.

    Returns:
        The values, in socket order. ``keywords_list``, ``custom`` and ``assets`` are new
        collections rather than views onto the document, so a node downstream may sort or
        edit them without reaching into a value on a wire.
    """
    metadata = document.metadata
    return Fields(
        title=metadata.title,
        description=metadata.description,
        author=metadata.author,
        copyright=metadata.copyright,
        language=metadata.language,
        keywords=keywords_text(metadata.keywords),
        keywords_list=list(metadata.keywords),
        created=metadata.created,
        modified=metadata.modified,
        generator=metadata.generator,
        custom=dict(metadata.custom),
        word_count=document.word_count,
        character_count=document.character_count,
        asset_count=len(document.assets),
        assets=sorted(document.assets),
        has_metadata=has_metadata(metadata),
        summary=summary(document),
    )


def summary(document: Document) -> str:
    """Render a document's metadata as aligned text.

    Args:
        document: The document to read.

    Returns:
        One line per field, labelled with the name of the socket carrying it, then the
        embedded file names under ``asset_count`` and the custom pairs under ``custom``.
        There is no trailing line break.
    """
    metadata = document.metadata
    values = (
        metadata.title,
        metadata.description,
        metadata.author,
        metadata.copyright,
        metadata.language,
        keywords_text(metadata.keywords),
        metadata.created,
        metadata.modified,
        metadata.generator,
        str(document.word_count),
        str(document.character_count),
        str(len(document.assets)),
    )
    lines = [_row(label, value) for label, value in zip(_LINE_LABELS, values, strict=True)]
    lines.extend(f"  {_one_line(name)}" for name in sorted(document.assets))
    lines.extend(_custom_lines(metadata.custom))
    return "\n".join(lines)


def keywords_text(keywords) -> str:
    """The keywords as one string.

    Args:
        keywords: The keywords in the order they were given.

    Returns:
        Them joined by :data:`KEYWORD_SEPARATOR`, which is how every export format spells
        the field, and an empty string where there are none.
    """
    return KEYWORD_SEPARATOR.join(keywords)


def has_metadata(metadata: Metadata) -> bool:
    """Whether a document says anything about itself.

    Args:
        metadata: The record to test.

    Returns:
        True when any of :data:`AUTHORED_FIELDS` holds something. A document read from a
        container with no ``meta.json``, and one whose fields are all empty, both answer
        False, so a switch can route to a node that fills them in.
    """
    return any(getattr(metadata, name) for name in AUTHORED_FIELDS)


def _custom_lines(custom: Mapping[str, str]) -> list[str]:
    """The custom pairs as summary lines, in the order the document holds them.

    Args:
        custom: The author's own pairs.

    Returns:
        One line naming the field and one line per pair, or a single line saying there are
        none. The order is the document's own.
    """
    if not custom:
        return [_row("custom", NONE_AT_ALL)]
    return [
        "custom",
        *(_row(f"  {_one_line(name)}", value) for name, value in custom.items()),
    ]


def _row(label: str, value: str) -> str:
    """One labelled line of the summary.

    Args:
        label: The name written at the start of the line, on one line.
        value: What it holds. An empty value is written as :data:`NOT_SET`.

    Returns:
        The label padded to :data:`_WIDTH` and the value after it. A value holding line
        breaks keeps every one of its lines, each continued in the column the value started
        in, so nothing is dropped and no line of a value can be read as a field of its own.
    """
    padded = label.ljust(_WIDTH - 1) + " "
    first, *rest = (value or NOT_SET).splitlines() or [""]
    indent = " " * len(padded)
    return "\n".join([padded + first, *(indent + line for line in rest)])


def _one_line(name: str) -> str:
    """One name on a single line.

    Args:
        name: An embedded file's name or a custom field's name, as the document spells it.

    Returns:
        The name with every line break written as a space. A name is a label, and a label
        that spanned lines would read as another field of the document, which a container
        assembled anywhere else could otherwise arrange. The socket carries the name as it
        is spelled.
    """
    return " ".join(name.splitlines())
