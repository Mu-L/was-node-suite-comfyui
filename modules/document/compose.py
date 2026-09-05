"""Document markup built from plain text.

:func:`markup_from_text` wraps text in paragraphs. A blank line starts a new ``<p>``, a line
break becomes ``<br>``, and ``&``, ``<`` and ``>`` are written as entities.
"""

from __future__ import annotations

import html

__all__ = ["LINE_BREAK", "markup_from_text"]

#: What a line break inside a paragraph is written as. The HTML5 spelling of a void
#: element, and :mod:`.text` reads it as one line break rather than as a word boundary.
LINE_BREAK = "<br>"


def markup_from_text(text: str) -> str:
    """The paragraphs of a piece of plain text, as document markup.

    Args:
        text: The text as typed or piped in. ``\\r\\n`` and ``\\r`` are read as line breaks,
            and a line holding nothing but whitespace ends the paragraph before it.

    Returns:
        One ``<p>`` element per paragraph, one to a line, with :data:`LINE_BREAK` between
        the lines of a paragraph and ``&``, ``<`` and ``>`` written as entities. An empty
        string when the text holds no character other than whitespace, so an empty box
        produces an empty document rather than an empty paragraph.
    """
    return "\n".join(
        "<p>{}</p>".format(LINE_BREAK.join(html.escape(line, quote=False) for line in lines))
        for lines in _paragraphs(text or "")
    )


def _paragraphs(text: str) -> list[list[str]]:
    """The lines of a piece of text, grouped into paragraphs.

    Args:
        text: The text, with line breaks in any of the three spellings.

    Returns:
        One list of lines per paragraph, in order, each line with its trailing whitespace
        removed. A blank line is a separator and belongs to no group, so text of nothing
        but blank lines produces no paragraphs at all.
    """
    found: list[list[str]] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            current.append(line.rstrip())
        elif current:
            found.append(current)
            current = []
    if current:
        found.append(current)
    return found
