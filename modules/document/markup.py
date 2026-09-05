"""HTML clean-up for the document nodes.

:func:`clean` removes ``<script>`` and ``<iframe>`` elements with their content, the
``<object>`` and ``<embed>`` tags, attribute names beginning ``on``, and ``javascript:`` URLs.
"""

from __future__ import annotations

import collections
import html
import re
from html.parser import HTMLParser

__all__ = [
    "DENIED_SCHEMES",
    "DROPPED_ELEMENTS",
    "HANDLER_PREFIX",
    "REMOVED_ELEMENTS",
    "URL_ATTRIBUTES",
    "clean",
    "describe",
]

#: Elements removed together with everything between their tags. HTML reads what sits
#: inside either one as raw text rather than as markup: a browser runs the one and ignores
#: the other, and neither is anything a reader sees, so keeping the text would put program
#: code or a dead frame's source into the document as writing.
REMOVED_ELEMENTS = frozenset({"script", "iframe"})

#: Elements whose tags are removed while what sits between them stays. That content is the
#: fallback a reader is shown when the embed does not load, so it is the document's own
#: writing, and anything dangerous in it is removed by this same pass.
DROPPED_ELEMENTS = frozenset({"object", "embed"})

#: Attributes whose value is read as a URL, and the only ones a scheme is refused in. An
#: attribute outside this set keeps whatever it holds, so a paragraph about ``javascript:``
#: in a ``title`` is left alone.
URL_ATTRIBUTES = frozenset(
    {
        "action",
        "background",
        "cite",
        "data",
        "formaction",
        "href",
        "longdesc",
        "ping",
        "poster",
        "src",
        "srcset",
        "xlink:href",
    }
)

#: URL schemes refused in one of :data:`URL_ATTRIBUTES`. The attribute is dropped and the
#: element and its text stay, so a scripted link becomes plain words.
DENIED_SCHEMES = ("javascript:",)

#: An attribute whose name starts with this is an event handler and is dropped whatever
#: element it sits on.
HANDLER_PREFIX = "on"

#: Characters a browser drops from a URL before reading its scheme. Removed from a copy of
#: the value, so ``java&Tab;script:`` and a value opening with a newline are still seen.
_IGNORED_IN_URL = re.compile(r"[\x00-\x20\x7f]")

#: What closes a comment, matching the parser's own reading of it, so a comment can be put
#: back with the spacing it was written with.
_COMMENT_CLOSE = re.compile(r"--\s*>")

#: An end tag in the form the parser recognises. Matched against the source so a tag goes
#: back with its own spelling, which is what keeps the camel case of an SVG element.
_END_TAG = re.compile(r"</\s*[a-zA-Z][-.a-zA-Z0-9:_]*\s*>")


def clean(markup: str) -> tuple[str, collections.Counter]:
    """Remove script, frame and event-handler markup from an HTML fragment.

    Args:
        markup: HTML, as a document node's widget holds it.

    Returns:
        ``(cleaned markup, what was removed)``. The counter is keyed on a readable label,
        ``"<script>"``, ``"onclick"`` or ``"javascript: href"``. It is empty when there
        was nothing to remove, and the markup is then the string that was passed in.
    """
    cleaner = _Cleaner(markup)
    cleaner.feed(markup)
    cleaner.close()
    return cleaner.result(), cleaner.removed


def describe(removed: collections.Counter) -> str:
    """Name what :func:`clean` removed, for a log line.

    Args:
        removed: The counter :func:`clean` answered.

    Returns:
        ``"2 <script>, 1 onclick"``, commonest first and then by label. An empty string
        when nothing was removed.
    """
    entries = sorted(removed.items(), key=lambda entry: (-entry[1], entry[0]))
    return ", ".join(f"{count} {label}" for label, count in entries)


def _denied_url(value: str | None) -> bool:
    """Does this attribute value open with one of :data:`DENIED_SCHEMES`?"""
    if not value:
        return False
    return _IGNORED_IN_URL.sub("", value).lower().startswith(DENIED_SCHEMES)


class _Cleaner(HTMLParser):
    """One pass over a fragment, writing the parts that are kept.

    Args:
        source: The markup being read, which the handlers slice to put a construct back
            exactly as it was written.

    Attributes:
        parts: The cleaned markup, in the order it was read.
        removed: Label to the number of times it was removed.
    """

    def __init__(self, source: str):
        # convert_charrefs would hand over decoded text, which would then have to be
        # escaped again on the way out, rewriting every entity in the document. With it
        # off each entity arrives as itself and is written back as it was.
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.removed: collections.Counter = collections.Counter()
        self._source = source
        self._line_starts = [0] + [at + 1 for at, char in enumerate(source) if char == "\n"]
        self._skipped: str | None = None
        self._depth = 0

    def result(self) -> str:
        """The cleaned markup.

        Returns:
            Everything that was kept, joined. A tag left unfinished at the end of the
            document is not in it: the parser drops one exactly as a browser does, so
            what comes out is what a browser would have made of what went in.
        """
        return "".join(self.parts)

    def handle_starttag(self, tag, attrs):
        if self._skipped is not None:
            # Only the element being skipped can nest inside itself here: everything else
            # between its tags is on its way out with it.
            if tag == self._skipped:
                self._depth += 1
            return
        if tag in REMOVED_ELEMENTS:
            self._skipped = tag
            self._depth = 1
            self.removed[f"<{tag}>"] += 1
            return
        if tag in DROPPED_ELEMENTS:
            self.removed[f"<{tag}>"] += 1
            return
        self.parts.append(self._start_tag(tag, attrs, ">"))

    def handle_startendtag(self, tag, attrs):
        if self._skipped is not None:
            return
        if tag in REMOVED_ELEMENTS or tag in DROPPED_ELEMENTS:
            self.removed[f"<{tag}>"] += 1
            return
        self.parts.append(self._start_tag(tag, attrs, "/>"))

    def handle_endtag(self, tag):
        if self._skipped is not None:
            if tag == self._skipped:
                self._depth -= 1
                if self._depth < 1:
                    self._skipped = None
            return
        if tag in REMOVED_ELEMENTS or tag in DROPPED_ELEMENTS:
            return
        # The parser lowercases a tag name, which would rewrite the camel case an SVG
        # element is spelled with, so the tag goes back as the source wrote it.
        written = _END_TAG.match(self._source, self._offset())
        self.parts.append(written.group(0) if written else f"</{tag}>")

    def handle_data(self, data):
        if self._skipped is not None:
            return
        if self.cdata_elem:
            # Raw text, such as the body of a style block, where an entity is not read as
            # one and writing one in would corrupt what it is the body of.
            self.parts.append(data)
            return
        # A bare < that opens nothing is written as an entity, and stays text.
        self.parts.append(data.replace("<", "&lt;"))

    def handle_entityref(self, name):
        if self._skipped is None:
            self.parts.append(f"&{name}{self._semicolon(len(name) + 1)}")

    def handle_charref(self, name):
        if self._skipped is None:
            self.parts.append(f"&#{name}{self._semicolon(len(name) + 2)}")

    def handle_comment(self, data):
        if self._skipped is None:
            self.parts.append(self._comment(data))

    def handle_decl(self, decl):
        if self._skipped is None:
            self.parts.append(f"<!{decl}>")

    def handle_pi(self, data):
        if self._skipped is None:
            self.parts.append(f"<?{data}>")

    def unknown_decl(self, data):
        if self._skipped is None:
            self.parts.append(f"<![{data}]]>")

    def _comment(self, data: str) -> str:
        """One comment as it was written.

        Args:
            data: What the parser read between the fences.

        Returns:
            The source text of the comment. HTML reads ``<!`` or ``</`` followed by
            anything that is not a tag or a declaration as a comment running to the first
            ``>``, and a conditional section such as ``<![if !IE]>`` arrives that way with
            none of the text it was written with, so the two are told apart here.
        """
        start = self._offset()
        if self._source.startswith("<!--", start):
            # Anchored where the parser says the text ends, since a search would run on
            # to a later comment and copy everything between the two.
            closed = _COMMENT_CLOSE.match(self._source, start + 4 + len(data))
            return self._source[start : closed.end()] if closed else f"<!--{data}-->"
        written = self._source[start : start + len(data) + 3]
        return written if written.endswith(">") else f"<!--{data}-->"

    def _semicolon(self, length: int) -> str:
        """The semicolon closing the reference being handled, where it was written.

        Args:
            length: Characters between the start of the reference and the semicolon.

        Returns:
            ``";"`` where the source has one, and an empty string where it does not. The
            parser reports ``&D`` in ``R&D and`` as a reference, so adding a semicolon
            that was never written would rewrite ordinary prose.
        """
        at = self._offset() + length
        return ";" if self._source[at : at + 1] == ";" else ""

    def _offset(self) -> int:
        """Where in the source the construct being handled starts, in characters."""
        line, column = self.getpos()
        if not 0 < line <= len(self._line_starts):
            return len(self._source)
        return self._line_starts[line - 1] + column

    def _start_tag(self, tag: str, attrs: list, close: str) -> str:
        """The text of one kept start tag.

        Args:
            tag: The element name, lowercased by the parser.
            attrs: ``(name, value)`` pairs, with ``None`` for an attribute written bare.
            close: ``">"`` for a start tag, ``"/>"`` for one that closes itself.

        Returns:
            The tag as it was written when every attribute survives, so its spelling,
            spacing and quoting are untouched. Otherwise a tag rebuilt from the
            attributes that are kept, each value escaped and double quoted.
        """
        kept = []
        dropped = False
        for name, value in attrs:
            label = self._refuse(name, value)
            if label is None:
                kept.append((name, value))
                continue
            self.removed[label] += 1
            dropped = True
        if not dropped:
            return self.get_starttag_text()
        written = "".join(
            f" {name}" if value is None else f' {name}="{html.escape(value, quote=True)}"'
            for name, value in kept
        )
        return f"<{tag}{written}{close}"

    def _refuse(self, name: str, value: str | None) -> str | None:
        """The label an attribute is counted under, or ``None`` where it is kept.

        Args:
            name: The attribute name, lowercased by the parser.
            value: Its value with entities already decoded, or ``None``.

        Returns:
            ``"onclick"`` for an event handler, ``"javascript: href"`` for a refused URL,
            ``None`` for everything else.
        """
        if name.startswith(HANDLER_PREFIX) and len(name) > len(HANDLER_PREFIX):
            return name
        if name in URL_ATTRIBUTES and _denied_url(value):
            return f"{DENIED_SCHEMES[0]} {name}"
        return None
