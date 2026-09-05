"""The element tree every reader of document markup is built on.

:func:`parse` runs one pass of :mod:`html.parser` over a fragment and returns the
:class:`Element` every top-level node hangs from. Nesting is bounded by :data:`MAX_DEPTH`.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

from .text import INLINE_ELEMENTS, SKIPPED_ELEMENTS

__all__ = [
    "CLOSED_BY",
    "Element",
    "LIST_ELEMENTS",
    "MAX_DEPTH",
    "ROW_GROUPS",
    "Reader",
    "TABLE_PARTS",
    "VOID_ELEMENTS",
    "breaks",
    "parse",
]

#: How deeply elements may nest. A tree is walked recursively, so a document nesting
#: thousands of elements would run the interpreter out of stack. Past this depth an element
#: keeps its text and holds nothing of its own. The limit sits far beyond anything a
#: document written by hand or by an editor reaches.
MAX_DEPTH = 64

#: Elements written without an end tag. One holds nothing, so its content never nests.
VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "param", "source", "track", "wbr",
    }
)

#: Elements laid out as a list of items, whichever way the items are marked.
LIST_ELEMENTS = frozenset({"dir", "menu", "ol", "ul"})

#: Elements that group a table's rows, and are read through to reach them.
ROW_GROUPS = frozenset({"colgroup", "tbody", "tfoot", "thead"})

#: What a table is built of. Anything else inside one is read after it, so text a browser
#: would lift out of the table is kept rather than dropped.
TABLE_PARTS = ROW_GROUPS | {"caption", "td", "th", "tr"}

#: An open element to the start tags that close it, for markup where the end tag was left
#: out. HTML closes an open ``<li>`` when the next one begins and :mod:`html.parser` does
#: not, so without this a list written that way nests every item inside the one before it.
CLOSED_BY = {
    "dd": frozenset({"dd", "dt"}),
    "dt": frozenset({"dd", "dt"}),
    "li": frozenset({"li"}),
    "option": frozenset({"option"}),
    "td": frozenset({"tbody", "td", "tfoot", "th", "thead", "tr"}),
    "th": frozenset({"tbody", "td", "tfoot", "th", "thead", "tr"}),
    "thead": frozenset({"tbody", "tfoot"}),
    "tbody": frozenset({"tbody", "tfoot"}),
    "tr": frozenset({"tbody", "tfoot", "thead", "tr"}),
}


def parse(markup: str) -> "Element":
    """Read an HTML fragment into a tree.

    Args:
        markup: HTML, as ``content.html`` holds it. A fragment and a whole document are
            both accepted, and neither has to be well formed.

    Returns:
        The root :class:`Element`, whose tag is an empty string and whose children are
        the top-level nodes of the fragment.
    """
    reader = Reader()
    reader.feed(markup or "")
    reader.close()
    return reader.root


def breaks(tag: str) -> bool:
    """Whether an element stands on its own rather than running inside a line of text."""
    return tag != "br" and tag not in INLINE_ELEMENTS


class Element:
    """One element of the document, with its attributes and what sits inside it.

    Attributes:
        tag: The element name, lowercased by the parser.
        attrs: ``{name: value}``, an attribute written bare holding an empty string.
        children: Strings of text and further elements, in the order they were read.
    """

    __slots__ = ("attrs", "children", "tag")

    def __init__(self, tag: str, attrs: Any = None) -> None:
        """Hold one element.

        Args:
            tag: The element name.
            attrs: The ``(name, value)`` pairs the parser read, or ``None``.
        """
        self.tag = tag
        self.attrs = {name: value or "" for name, value in attrs} if attrs else {}
        self.children: list[Any] = []

    def attr(self, name: str) -> str:
        """One attribute's value, or an empty string where it was not written."""
        return self.attrs.get(name, "")

    def elements(self, tags: Any) -> list["Element"]:
        """Every immediate child whose tag is in ``tags``, in document order."""
        return [
            child
            for child in self.children
            if isinstance(child, Element) and child.tag in tags
        ]


class Reader(HTMLParser):
    """One pass over a fragment, building the tree a reader is written from.

    Attributes:
        root: The element every top-level node hangs from. It has no tag of its own.
    """

    def __init__(self) -> None:
        # convert_charrefs hands over decoded text, so an entity is already the character it
        # stands for and a reader never sees one.
        super().__init__(convert_charrefs=True)
        self.root = Element("")
        self._open: list[Element] = [self.root]
        self._skipped: str | None = None
        self._depth = 0

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
        element = self._placed(tag, attrs)
        if tag not in VOID_ELEMENTS and len(self._open) < MAX_DEPTH:
            self._open.append(element)

    def handle_startendtag(self, tag, attrs):
        # A tag closing itself holds nothing, whatever element it names, so the rest of the
        # document is not put inside it.
        if self._skipped is None and tag not in SKIPPED_ELEMENTS:
            self._placed(tag, attrs)

    def handle_endtag(self, tag):
        if self._skipped is not None:
            if tag == self._skipped:
                self._depth -= 1
                if self._depth < 1:
                    self._skipped = None
            return
        if tag in VOID_ELEMENTS:
            return
        for at in range(len(self._open) - 1, 0, -1):
            if self._open[at].tag == tag:
                # An end tag closes everything still open inside it, as a browser does.
                del self._open[at:]
                return

    def handle_data(self, data):
        if self._skipped is None and data:
            self._open[-1].children.append(data)

    def _placed(self, tag: str, attrs: Any) -> Element:
        """Add one element where it belongs, closing whatever its start tag closes.

        Args:
            tag: The element name.
            attrs: The attribute pairs the parser read.

        Returns:
            The element, already in the tree.
        """
        while len(self._open) > 1:
            top = self._open[-1].tag
            if tag in CLOSED_BY.get(top, ()) or (top == "p" and breaks(tag)):
                self._open.pop()
                continue
            break
        element = Element(tag, attrs)
        self._open[-1].children.append(element)
        return element
