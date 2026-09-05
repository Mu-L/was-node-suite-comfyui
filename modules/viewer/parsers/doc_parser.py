"""Document parser for the WAS Content Viewer.

Hands a ``DOC`` to the viewer as the document's own markup, with every embedded file inlined
as a ``data:`` URL.
"""

import base64
import html as html_escape
import mimetypes
import re

from .base_parser import BaseParser

#: Where ``content.html`` names an embedded file, relative to the container's ``assets/``.
ASSET_REFERENCE = re.compile(
    r"""(src|href|poster)\s*=\s*(["'])\s*(?:\./)?assets/([^"']+)\2""",
    re.IGNORECASE,
)

#: Separator the viewer splits a list of items on.
LIST_SEPARATOR = "\n---LIST_SEPARATOR---\n"


class DocParser(BaseParser):
    """Parser for the ``DOC`` type, drawing a document as the markup it holds."""

    PARSER_NAME = "doc"
    PARSER_VIEW = "html"
    PARSER_PRIORITY = 20

    @classmethod
    def detect_input(cls, content) -> bool:
        """Whether any item is a document.

        Args:
            content: One value, or a list of them.

        Returns:
            ``True`` when at least one item is a document container.
        """
        items = content if isinstance(content, (list, tuple)) else [content]
        return any(cls._is_document(item) for item in items)

    @classmethod
    def handle_input(cls, content, logger=None) -> dict:
        """Draw every document in the input as its markup.

        Args:
            content: One value, or a list of them.
            logger: Where to report what was drawn.

        Returns:
            ``display_content``, ``output_values`` and ``content_hash``, or ``None`` when no
            item is a document.
        """
        items = list(content) if isinstance(content, (list, tuple)) else [content]
        documents = [item for item in items if cls._is_document(item)]
        if not documents:
            return None

        pages = [cls._markup(document, logger) for document in documents]
        display_content = LIST_SEPARATOR.join(pages)
        words = sum(getattr(document, "word_count", 0) for document in documents)
        content_hash = f"doc_{len(documents)}_{words}_{hash(display_content[:400]) & 0xFFFFFFFF}"

        if logger:
            logger.info(
                "[Doc Parser] Drew %d document(s), %d word(s)", len(documents), words
            )

        return {
            "display_content": display_content,
            "output_values": items,
            "content_hash": content_hash,
        }

    @classmethod
    def _is_document(cls, value) -> bool:
        """Whether one value is a document container."""
        if value is None:
            return False
        try:
            from ...document.container import is_document
        except Exception:
            return False
        return is_document(value)

    @classmethod
    def _markup(cls, document, logger=None) -> str:
        """One document as a whole HTML page, with its embedded files inlined."""
        content = getattr(document, "content", "") or ""
        assets = {}
        try:
            assets = dict(getattr(document, "assets", {}) or {})
        except Exception:
            assets = {}

        body = cls._inline_assets(content, assets, logger) if assets else content
        if not body.strip():
            body = "<p><em>This document has no content.</em></p>"

        title = ""
        try:
            title = getattr(document.metadata, "title", "") or ""
        except Exception:
            title = ""

        language = ""
        try:
            language = getattr(document.metadata, "language", "") or ""
        except Exception:
            language = ""

        if re.search(r"<html[\s>]", body, re.IGNORECASE):
            return body

        opening = f'<html lang="{html_escape.escape(language, quote=True)}">' if language else "<html>"
        head = f"<title>{html_escape.escape(title)}</title>" if title else ""
        return f"<!DOCTYPE html>{opening}<head><meta charset=\"utf-8\">{head}</head><body>{body}</body></html>"

    @classmethod
    def _inline_assets(cls, markup: str, assets: dict, logger=None) -> str:
        """Rewrite every ``assets/`` reference in the markup as a ``data:`` URL."""
        missing = []

        def replace(match: "re.Match") -> str:
            attribute, quote, name = match.group(1), match.group(2), match.group(3)
            payload = assets.get(name)
            if payload is None:
                missing.append(name)
                return match.group(0)
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            encoded = base64.b64encode(payload).decode("ascii")
            return f"{attribute}={quote}data:{kind};base64,{encoded}{quote}"

        rewritten = ASSET_REFERENCE.sub(replace, markup)
        if missing and logger:
            logger.warning(
                "[Doc Parser] %d asset(s) the markup names are not in the container: %s",
                len(missing),
                ", ".join(sorted(set(missing))[:5]),
            )
        return rewritten
