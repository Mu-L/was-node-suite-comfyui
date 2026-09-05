"""Plain text from document HTML, and the two counts taken from it.

:func:`plain_text` decodes entities, drops tags, collapses whitespace, keeps ``<pre>`` as
written, and writes ``block_breaks`` line breaks at a block boundary. :func:`word_count` and
:func:`character_count` read that text.
"""

from __future__ import annotations

import unicodedata
from html.parser import HTMLParser

__all__ = [
    "BLOCK_BREAK",
    "COUNTING_BREAK",
    "INLINE_ELEMENTS",
    "LINE_BREAK_ELEMENT",
    "PREFORMATTED_ELEMENTS",
    "SKIPPED_ELEMENTS",
    "character_count",
    "counts",
    "plain_text",
    "word_count",
]

#: Elements that do not break a line. Text either side of one belongs to the same word, so
#: ``<b>fi</b>sh`` is one word and ``<p>one</p><p>two</p>`` is two. ``<br>`` is not among
#: them: it is a line break, and the words either side of it are separate.
INLINE_ELEMENTS = frozenset(
    {
        "a", "abbr", "acronym", "area", "b", "bdi", "bdo", "big", "button", "cite", "code",
        "data", "del", "dfn", "em", "font", "i", "img", "input", "ins", "kbd", "label",
        "map", "mark", "meter", "nobr", "output", "param", "picture", "progress", "q",
        "rb", "rp", "rt", "ruby", "s", "samp", "select", "small", "source", "span",
        "strike", "strong", "sub", "sup", "time", "track", "tt", "u", "var", "wbr",
    }
)

#: Elements whose whitespace is part of the writing and is kept exactly as it is written,
#: since a browser shows it that way and an editor counts it that way.
PREFORMATTED_ELEMENTS = frozenset({"pre"})

#: The one element that writes a single line break whatever a block boundary writes. It
#: marks a break inside a paragraph, so the lines either side of it are one paragraph.
LINE_BREAK_ELEMENT = "br"

#: Line breaks a block boundary writes for the two counts. One, so
#: ``<p>one</p><p>two</p>`` and ``one<br>two`` read alike.
COUNTING_BREAK = 1

#: Line breaks a block boundary writes where the text is going into a file a person reads.
#: A blank line between blocks is what keeps the paragraphs a document was written in.
BLOCK_BREAK = 2

#: Elements whose content is not writing. A style sheet and a program are not read by
#: anybody, a document's own title is metadata rather than body text, and a template holds
#: markup that has not been placed in the document.
SKIPPED_ELEMENTS = frozenset({"head", "script", "style", "template", "title"})

#: First letters of the Unicode categories a word may not be made only of: punctuation,
#: symbols, separators and control characters.
_NOT_WORD_CATEGORIES = "PSZC"


def plain_text(markup: str, *, block_breaks: int = COUNTING_BREAK) -> str:
    """The readable text of an HTML fragment.

    Args:
        markup: HTML, as ``content.html`` holds it. A fragment and a whole document are
            both accepted, and neither has to be well formed.
        block_breaks: How many ``\\n`` a block element writes as it opens or closes.
            :data:`COUNTING_BREAK` is the rule the two counts are defined over;
            :data:`BLOCK_BREAK` leaves a blank line between blocks, which is what a text
            file wants. A ``<br>`` writes one under either.

    Returns:
        The text with entities decoded, markup removed, whitespace collapsed to single
        spaces, and the line broken wherever a block element or a ``<br>`` breaks it.
        A no-break space is read as a space. The body of ``<pre>`` keeps its own
        whitespace, that being what it is for, so text opening or closing with one keeps
        the space and the line breaks inside it; anywhere else there is no leading or
        trailing whitespace.
    """
    reader = _Text(block_breaks)
    reader.feed(markup or "")
    reader.close()
    return reader.result()


def word_count(text: str) -> int:
    """How many words a piece of plain text holds.

    Args:
        text: Text from :func:`plain_text`, not markup.

    Returns:
        The number of whitespace-separated runs holding at least one character that is
        not punctuation, a symbol, a separator or a control character. A hyphenated word
        is one word, an apostrophe does not split one, a number with a decimal point or a
        thousands separator is one word, and a run of punctuation on its own, such as a
        lone dash or a bare emoji, is not a word at all.
    """
    return sum(1 for run in text.split() if _is_word(run))


def character_count(text: str) -> int:
    """How many characters a piece of plain text holds.

    Args:
        text: Text from :func:`plain_text`, not markup.

    Returns:
        The number of characters in it, counting each space between words and each line
        break as one character. An entity counts as the character it stands for, so
        ``&amp;`` is one character and not five.
    """
    return len(text)


def counts(markup: str) -> tuple[int, int]:
    """The word and character counts of an HTML fragment, in one pass.

    Args:
        markup: HTML, as ``content.html`` holds it.

    Returns:
        ``(word count, character count)``, both taken from :func:`plain_text`.
    """
    text = plain_text(markup)
    return word_count(text), character_count(text)


def _is_word(run: str) -> bool:
    """Whether one whitespace-separated run counts as a word."""
    return any(unicodedata.category(char)[0] not in _NOT_WORD_CATEGORIES for char in run)


class _Text(HTMLParser):
    """One pass over a fragment, collecting the text a reader would see.

    Args:
        block_breaks: Line breaks a block element writes as it opens or closes, at least
            one.

    Attributes:
        parts: The text collected so far, in order.
    """

    def __init__(self, block_breaks: int = COUNTING_BREAK) -> None:
        # convert_charrefs hands over decoded text, which is what both counts are of: an
        # entity stands for one character and is counted as that character.
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._blocks = max(int(block_breaks), COUNTING_BREAK)
        self._skipped: str | None = None
        self._depth = 0
        self._pre = 0
        self._space = False
        self._break = 0

    def result(self) -> str:
        """Everything collected, joined."""
        return "".join(self.parts)

    def handle_starttag(self, tag, attrs):
        if self._skipped is not None:
            # Only the element being skipped can nest inside itself here: everything else
            # between its tags is on its way out with it.
            if tag == self._skipped:
                self._depth += 1
            return
        if tag in SKIPPED_ELEMENTS:
            self._skipped = tag
            self._depth = 1
            return
        if tag in PREFORMATTED_ELEMENTS:
            self._pre += 1
        self._separate(tag)

    def handle_endtag(self, tag):
        if self._skipped is not None:
            if tag == self._skipped:
                self._depth -= 1
                if self._depth < 1:
                    self._skipped = None
            return
        if tag in PREFORMATTED_ELEMENTS and self._pre:
            self._pre -= 1
        self._separate(tag)

    def handle_data(self, data):
        if self._skipped is not None:
            return
        if self._pre:
            self._flush()
            self.parts.append(data)
            return
        words = data.split()
        if not words:
            self._space = self._space or bool(self.parts)
            return
        if data[:1].isspace():
            self._space = bool(self.parts)
        self._flush()
        self.parts.append(" ".join(words))
        self._space = data[-1:].isspace()
        self._break = 0

    def _separate(self, tag: str) -> None:
        """Note that a tag has been read, and how many line breaks it writes."""
        if tag in INLINE_ELEMENTS or not self.parts:
            return
        written = COUNTING_BREAK if tag == LINE_BREAK_ELEMENT else self._blocks
        # A block boundary and a <br> can both fall between two pieces of text, and the
        # widest of them is the separator: a paragraph ending beside a line break inside it
        # is still the end of the paragraph.
        self._break = max(self._break, written)

    def _flush(self) -> None:
        """Write whatever separator is pending, before the text that follows it."""
        if not self.parts:
            self._break = 0
            self._space = False
            return
        if self._break:
            wanted = self._break - self._written_breaks()
            if wanted > 0:
                self.parts.append("\n" * wanted)
        elif self._space and not self.parts[-1].endswith((" ", "\n")):
            self.parts.append(" ")
        self._break = 0
        self._space = False

    def _written_breaks(self) -> int:
        """How many line breaks the text collected so far already ends with."""
        found = 0
        for piece in reversed(self.parts):
            for char in reversed(piece):
                if char != "\n":
                    return found
                found += 1
        return found
