"""The save formats a document needs no third-party library for.

:data:`FORMATS` is what a save node offers, each spelled as the extension the file gets, and
:func:`payload` answers the bytes one writes. Text is UTF-8.
"""

from __future__ import annotations

import html
import re
from typing import Any

from . import text
from .compose import markup_from_text
from .container import Document, NotADocument, is_document
from .metadata import Metadata

__all__ = [
    "CONTAINER",
    "ENCODING",
    "FORMATS",
    "MARKUP",
    "TEXT_FORMATS",
    "as_document",
    "html_page",
    "is_whole_page",
    "normalized",
    "payload",
    "require_source",
    "text_of",
]

#: The document container: a zip holding ``content.html``, ``meta.json`` and ``assets/``.
#: The only format that keeps an embedded file, and the only one that keeps the metadata as
#: metadata rather than as part of the writing.
CONTAINER = ".wasdoc"

#: A whole HTML page: the document's markup with a head carrying the character set and the
#: metadata. The one format here that writes markup.
MARKUP = ".html"

#: Formats whose file is the text and nothing else. They differ from one another only in
#: the extension, which is what decides how an editor colours the file and what opens it on
#: a double click; the bytes written are identical.
TEXT_FORMATS = (".txt", ".md", ".css", ".js", ".py", ".json")

#: Every format, container first, in the order a widget offers them.
FORMATS = (CONTAINER, MARKUP) + TEXT_FORMATS

#: The codec every text format is written in. The container's own entries are UTF-8 by the
#: format definition, JSON is UTF-8 by RFC 8259, and an HTML page written in anything else
#: needs a matching declaration in its head to be readable at all.
ENCODING = "utf-8"

#: Metadata field to the ``name`` its ``<meta>`` element carries in an HTML page.
PAGE_META = (
    ("description", "description"),
    ("author", "author"),
    ("copyright", "copyright"),
    ("generator", "generator"),
)

#: Where the two timestamps are written in a page, and the vocabulary that defines them.
DCTERMS = "http://purl.org/dc/terms/"
CREATED_META = "dcterms.created"
MODIFIED_META = "dcterms.modified"

#: ``<meta>`` names the page writes itself. A custom pair spelled as one of them is left
#: out, since a second element of the same name is read by nobody and contradicts the first.
RESERVED_META = frozenset(
    {name for _, name in PAGE_META} | {"keywords", CREATED_META, MODIFIED_META}
)

#: What joins the keywords in a page, the spelling every export format uses for the field.
KEYWORD_SEPARATOR = ", "

#: Markup that opens a whole page rather than a fragment. A fragment mentioning HTML in its
#: writing carries ``&lt;html``, so only a real element matches.
_WHOLE_PAGE = re.compile(r"<\s*html[\s/>]|<!\s*doctype\s+html", re.IGNORECASE)

#: How much of the markup is examined for that opening. A page declares itself at the top.
_PAGE_WINDOW = 2048


def require_source(value: Any, label: str = "doc") -> Document | str:
    """Read a socket that takes either a document or a string.

    Args:
        value: Whatever arrived on the socket.
        label: The input's name, used in the message.

    Returns:
        The document, or the string, whichever arrived.

    Raises:
        NotADocument: The value is neither.
    """
    if isinstance(value, Document):
        return value
    if isinstance(value, str):
        return value
    raise NotADocument(
        f"the {label} input takes a document or a string of text, and was given "
        f"{_described(value)}.\n"
        f"  Connect a node with a DOC output, such as Text to DOC, or any node with a "
        f"string output, such as Text Multiline."
    )


def normalized(extension: Any) -> str:
    """One of :data:`FORMATS`, read from a widget value.

    Args:
        extension: The widget value. A leading dot is added where it is missing and the
            case is ignored, so ``PY`` and ``.py`` both name the same format.

    Returns:
        The format, spelled as :data:`FORMATS` spells it.

    Raises:
        ValueError: The value names no format this module writes.
    """
    text = str(extension or "").strip().lower()
    if text and not text.startswith("."):
        text = "." + text
    if text not in FORMATS:
        raise ValueError(
            f"{extension!r} is not a format that can be written without an extra library.\n"
            f"  The formats are: {', '.join(FORMATS)}."
        )
    return text


def payload(value: Any, extension: Any) -> bytes:
    """The bytes one format writes for a document or a string.

    Args:
        value: A :class:`~modules.document.container.Document` or a string.
        extension: One of :data:`FORMATS`.

    Returns:
        The whole file. The container's bytes for :data:`CONTAINER`, a UTF-8 HTML page for
        :data:`MARKUP`, and the UTF-8 text for a format in :data:`TEXT_FORMATS`.

    Raises:
        NotADocument: ``value`` is neither a document nor a string.
        ValueError: ``extension`` names no format.
        DocumentError: The container could not be built.
    """
    source = require_source(value)
    chosen = normalized(extension)
    if chosen == CONTAINER:
        return as_document(source).data
    if chosen == MARKUP:
        content = source.content if is_document(source) else source
        record = source.metadata if is_document(source) else None
        return html_page(content, record).encode(ENCODING)
    return text_of(source).encode(ENCODING)


def as_document(value: Any) -> Document:
    """The document a container is written from.

    Args:
        value: A document, or a string to wrap as one.

    Returns:
        The document itself, or a new one whose content is the string wrapped into
        paragraphs and whose timestamps are stamped now. A string carries no metadata, so
        the document made from one has none beyond its stamps.

    Raises:
        NotADocument: ``value`` is neither a document nor a string.
        DocumentError: The content cannot go in a container.
    """
    source = require_source(value)
    if is_document(source):
        return source
    return Document.build(markup_from_text(source))


def text_of(value: Any) -> str:
    """The characters a text format writes.

    Args:
        value: A document, or a string.

    Returns:
        The string exactly as it arrived, or the document's text: markup removed and
        entities decoded, so no tag ever reaches a file that is not HTML, with a blank
        line between one block and the next so the paragraphs it was written in survive.
        That is a wider separator than the one the container's counts are taken over, and
        :mod:`modules.document.text` states both.

    Raises:
        NotADocument: ``value`` is neither a document nor a string.
    """
    source = require_source(value)
    if not is_document(source):
        return source
    return text.plain_text(source.content, block_breaks=text.BLOCK_BREAK)


def html_page(content: str, metadata: Metadata | None = None) -> str:
    """A whole HTML file for a document's markup.

    Args:
        content: The markup. A document's ``content.html`` is a fragment, so a page is
            built around it; markup that already opens a page is returned as it is.
        metadata: The record whose fields become the page's head. ``None`` writes a head
            carrying the character set alone.

    Returns:
        The page, with ``<meta charset>`` first so a reader knows the encoding before it
        reads a character of the text.
    """
    body = content or ""
    if is_whole_page(body):
        return body
    record = metadata if isinstance(metadata, Metadata) else Metadata()
    lang = f' lang="{_attribute(record.language)}"' if record.language else ""
    head = "".join(f"  {line}\n" for line in _head(record))
    written = body if not body or body.endswith("\n") else body + "\n"
    return (
        f"<!DOCTYPE html>\n"
        f"<html{lang}>\n"
        f"<head>\n{head}</head>\n"
        f"<body>\n{written}</body>\n"
        f"</html>\n"
    )


def is_whole_page(markup: str) -> bool:
    """Whether markup opens a whole HTML page rather than a fragment.

    Args:
        markup: The markup to look at. Only its opening is examined, since a page declares
            itself at the top.

    Returns:
        True for markup carrying a doctype or an ``<html>`` element, which is written to a
        file as it stands rather than wrapped in a second page.
    """
    return bool(_WHOLE_PAGE.search((markup or "")[:_PAGE_WINDOW]))


def _head(record: Metadata) -> list[str]:
    """The elements of a page's head, character set first.

    Args:
        record: The metadata to write.

    Returns:
        One line per element, in the order they are written. A field holding nothing writes
        no element at all, so an empty title never replaces the file name a viewer would
        otherwise show.
    """
    lines = ['<meta charset="utf-8">']
    if record.title:
        lines.append(f"<title>{html.escape(record.title)}</title>")
    for field, name in PAGE_META:
        value = getattr(record, field, "")
        if value:
            lines.append(_meta(name, value))
    if record.keywords:
        lines.append(_meta("keywords", KEYWORD_SEPARATOR.join(record.keywords)))
    if record.created or record.modified:
        lines.append(f'<link rel="schema.dcterms" href="{DCTERMS}">')
        if record.created:
            lines.append(_meta(CREATED_META, record.created))
        if record.modified:
            lines.append(_meta(MODIFIED_META, record.modified))
    for name, value in record.custom.items():
        label = str(name).strip()
        if label and value and label.lower() not in RESERVED_META:
            lines.append(_meta(label, value))
    return lines


def _meta(name: str, value: str) -> str:
    """One ``<meta>`` element, both of its attribute values escaped."""
    return f'<meta name="{_attribute(name)}" content="{_attribute(value)}">'


def _attribute(value: str) -> str:
    """One attribute value, with both quote characters and the markup ones escaped."""
    return html.escape(str(value), quote=True)


def _described(value: Any) -> str:
    """Name what arrived on a socket, for the message that refuses it."""
    if value is None:
        return "nothing at all, so nothing is connected to it"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return (
            f"{len(bytes(value))} raw byte(s); a DOC carries the document itself rather "
            f"than the bytes of its container"
        )
    name = type(value).__name__
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"
