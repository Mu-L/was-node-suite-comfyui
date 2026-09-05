"""Document markup laid out as plain text, with the shape of the document kept.

:data:`LINK_MODES`, :data:`IMAGE_MODES` and :data:`TABLE_MODES` are the choices a node offers
for a link's target, a picture and a table's columns, default first.
"""

from __future__ import annotations

import textwrap
import unicodedata
from typing import Any

from .text import PREFORMATTED_ELEMENTS
from .tree import LIST_ELEMENTS, ROW_GROUPS, TABLE_PARTS, Element, breaks, parse

__all__ = [
    "BULLET",
    "COLUMN_GAP",
    "HEADING_LEVELS",
    "HEADING_UNDERLINES",
    "IMAGE_MODES",
    "IMAGE_PLACEHOLDER",
    "INDENT",
    "LINK_MODES",
    "MAX_SPAN",
    "MIN_WIDTH",
    "QUOTE_PREFIX",
    "RULE_WIDTH",
    "TABLE_MODES",
    "to_plaintext",
]

#: What becomes of a link's target, default first. ``text and url`` writes the address in
#: brackets after the words, ``text only`` drops it, and ``footnotes`` numbers each address
#: and lists them all at the end.
LINK_MODES = ("text and url", "text only", "footnotes")

#: What becomes of a picture, default first. A picture cannot be drawn in text, so what is
#: left is its alt text, optionally with the file it came from, or nothing.
IMAGE_MODES = ("alt text", "alt text and source", "skip")

#: What becomes of a table, default first. ``aligned columns`` pads every cell so the
#: columns read down the page; ``tab separated`` writes one tab between cells, which is what
#: a spreadsheet reads.
TABLE_MODES = ("aligned columns", "tab separated")

#: Heading level to the character its text is underlined with. One per level, so a level 3
#: heading does not read like a level 2 one.
HEADING_UNDERLINES = ("=", "-", "~", "^", "+", ".")

#: The six heading elements and the level each carries.
HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

#: What marks an unordered list item, what a quoted line opens with, what a definition is
#: indented by, and what separates two table columns. An item's later lines are indented to
#: the width of its own marker, so the text of a wrapped item stays in one column.
BULLET = "- "
QUOTE_PREFIX = "> "
INDENT = "  "
COLUMN_GAP = "  "

#: Stands in for a picture carrying no alt text, inside the brackets an image is written in.
IMAGE_PLACEHOLDER = "image"

#: How many characters wide a horizontal rule is drawn, or the wrapping width where that is
#: narrower. Wide enough to read as a rule, and narrower than the 72 to 80 columns text is
#: usually read at, so it does not read as a line of the document.
RULE_WIDTH = 40

#: Narrowest text column indenting may leave. A list nested six deep inside a quotation
#: still has words to fit, so the indent stops eating the width rather than reaching zero.
MIN_WIDTH = 8

#: Most columns one cell may be spread across. A cell is padded to the span it declares so
#: the columns after it still line up, and a span of millions would otherwise turn one cell
#: into a row of empty ones large enough to exhaust memory.
MAX_SPAN = 64

#: URL schemes whose value carries the file itself. Only the media type is written out: the
#: rest is a base64 payload that would put a picture into the text a character at a time.
_INLINE_DATA = "data:"

#: Prefixes ignored when a link's words are compared with its address, so a link written as
#: its own address is not written out twice.
_IGNORED_PREFIXES = ("https://", "http://", "mailto:")


def to_plaintext(
    markup: str,
    *,
    links: str = LINK_MODES[0],
    images: str = IMAGE_MODES[0],
    tables: str = TABLE_MODES[0],
    width: int = 0,
) -> str:
    """Lay out an HTML fragment as plain text.

    Args:
        markup: HTML, as ``content.html`` holds it. A fragment and a whole document are
            both accepted, and neither has to be well formed.
        links: One of :data:`LINK_MODES`.
        images: One of :data:`IMAGE_MODES`.
        tables: One of :data:`TABLE_MODES`.
        width: Column to wrap paragraphs at, or 0 to leave each on one line. A table and a
            preformatted block are never wrapped, since wrapping either destroys it.

    Returns:
        The text. Blocks are separated by a blank line, there is no leading or trailing
        blank line, and no line carries trailing whitespace. An unreadable value for one of
        the three modes is read as its default rather than refused.
    """
    return _Renderer(links, images, tables, width).document(parse(markup))


# ---------------------------------------------------------------------- one line


class _Inline:
    """A run of inline content, collected as lines of collapsed text.

    Attributes:
        lines: The parts of each line so far, one list per line.
    """

    def __init__(self) -> None:
        self.lines: list[list[str]] = [[]]
        self._space = False

    def text(self, data: str) -> None:
        """Add text, collapsing every run of whitespace in it to one space."""
        words = data.split()
        if not words:
            self._space = True
            return
        if data[:1].isspace():
            self._space = True
        self._write(" ".join(words))
        self._space = bool(data[-1:].isspace())

    def verbatim(self, piece: str) -> None:
        """Add a piece of text that is already written the way it goes out."""
        self._write(piece)

    def attached(self, piece: str) -> None:
        """Add a piece with nothing between it and the text before it.

        Args:
            piece: The text to add.
        """
        self._keep(piece, False)

    def spaced(self, piece: str) -> None:
        """Add a piece of text with one space in front of it.

        Args:
            piece: The text to add.
        """
        self._keep(piece, True)

    def hard_break(self) -> None:
        """Start a new line, as ``<br>`` does."""
        self.lines.append([])
        self._space = False

    def mark(self) -> tuple[int, int]:
        """Where the next piece will land, for reading back what follows it."""
        return len(self.lines) - 1, len(self.lines[-1])

    def since(self, mark: tuple[int, int]) -> str:
        """Everything written since ``mark``, as one line."""
        line, part = mark
        pieces = ["".join(self.lines[line][part:])]
        pieces.extend("".join(rest) for rest in self.lines[line + 1 :])
        return " ".join(piece for piece in pieces if piece)

    def result(self) -> list[str]:
        """The lines, each stripped, with the blank ones at either end dropped."""
        lines = ["".join(parts).strip() for parts in self.lines]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return lines

    def _keep(self, piece: str, space: bool) -> None:
        """Write one piece, keeping a space that was already pending for what follows.

        Args:
            piece: The text to add.
            space: Whether one space goes in front of it.
        """
        # A marker written after a link's words belongs against them, and the space that
        # was pending belongs after it, in front of whatever the sentence goes on to say.
        pending = self._space
        self._space = space
        self._write(piece)
        self._space = pending

    def _write(self, piece: str) -> None:
        """Put one piece on the current line, with the pending space where one is due."""
        line = self.lines[-1]
        if self._space and line and not line[-1].endswith(" "):
            line.append(" ")
        self._space = False
        line.append(piece)


# ---------------------------------------------------------------------- layout


class _Renderer:
    """The layout of one document, written a block at a time.

    Args:
        links: One of :data:`LINK_MODES`.
        images: One of :data:`IMAGE_MODES`.
        tables: One of :data:`TABLE_MODES`.
        width: Column to wrap at, 0 for none.
    """

    def __init__(self, links: str, images: str, tables: str, width: Any) -> None:
        self._links = links if links in LINK_MODES else LINK_MODES[0]
        self._images = images if images in IMAGE_MODES else IMAGE_MODES[0]
        self._tables = tables if tables in TABLE_MODES else TABLE_MODES[0]
        self._width = max(int(width), 0) if isinstance(width, (int, float)) else 0
        self._footnotes: list[str] = []
        self._numbered: dict[str, int] = {}

    def document(self, root: Element) -> str:
        """The whole document as text, footnotes included where they are collected."""
        blocks = self._flow(root.children, self._width)
        text = "\n\n".join("\n".join(block) for block in blocks)
        if not self._footnotes:
            return text
        listed = "\n".join(
            f"[{number}] {url}" for number, url in enumerate(self._footnotes, 1)
        )
        return f"{text}\n\n{listed}" if text else listed

    # ------------------------------------------------------------------ blocks

    def _flow(self, children: list, budget: int) -> list[list[str]]:
        """Every block a container's children make, each as its own lines.

        Args:
            children: The container's children.
            budget: Columns the text may use, 0 for no wrapping.

        Returns:
            One entry per block, in order, each a list of lines and none of them empty.
            Runs of inline content between two blocks each make a block of their own.
        """
        blocks: list[list[str]] = []
        run: list = []
        for child in children:
            if isinstance(child, str) or not breaks(child.tag):
                run.append(child)
                continue
            blocks.extend(self._run(run, budget))
            run = []
            blocks.extend(self._block(child, budget))
        blocks.extend(self._run(run, budget))
        return [block for block in blocks if block]

    def _run(self, nodes: list, budget: int) -> list[list[str]]:
        """One block from a run of inline content, or nothing where it is only spaces."""
        if not nodes:
            return []
        lines = self._lines(nodes, budget)
        return [lines] if lines else []

    def _block(self, node: Element, budget: int) -> list[list[str]]:
        """The blocks one element on its own makes.

        Args:
            node: The element.
            budget: Columns the text may use, 0 for no wrapping.

        Returns:
            One entry per block. An element with no layout of its own, ``div`` and every
            unknown element among them, contributes whatever its children do.
        """
        tag = node.tag
        if tag in HEADING_LEVELS:
            return self._heading(node, HEADING_LEVELS[tag], budget)
        if tag == "hr":
            return [[self._rule(budget)]]
        if tag in PREFORMATTED_ELEMENTS:
            return self._preformatted(node)
        if tag == "blockquote":
            return self._quotation(node, budget)
        if tag in LIST_ELEMENTS:
            return self._list(node, budget)
        if tag == "dl":
            return self._definitions(node, budget)
        if tag == "table":
            return self._table(node, budget)
        if tag == "li":
            # An item outside any list still reads as one, and is the only way markup that
            # lost its <ul> keeps its marker.
            return self._items([(BULLET, node)], budget)
        return self._flow(node.children, budget)

    def _heading(self, node: Element, level: int, budget: int) -> list[list[str]]:
        """One heading: its text, underlined by the character its level carries."""
        lines = self._lines(node.children, budget)
        if not lines:
            return []
        underline = HEADING_UNDERLINES[min(level, len(HEADING_UNDERLINES)) - 1]
        return [lines + [underline * max(_display_width(line) for line in lines)]]

    def _rule(self, budget: int) -> str:
        """One horizontal rule, as wide as :data:`RULE_WIDTH` or the text column."""
        return "-" * (min(RULE_WIDTH, budget) if budget else RULE_WIDTH)

    def _preformatted(self, node: Element) -> list[list[str]]:
        """One preformatted block, its own spacing and line breaks kept as written."""
        lines = _verbatim(node).split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return [lines] if lines else []

    def _quotation(self, node: Element, budget: int) -> list[list[str]]:
        """One quotation, every line of it opening with :data:`QUOTE_PREFIX`."""
        marker = QUOTE_PREFIX.rstrip()
        quoted: list[str] = []
        for block in self._flow(node.children, _reduced(budget, len(QUOTE_PREFIX))):
            if quoted:
                quoted.append("")
            quoted.extend(block)
        return [[QUOTE_PREFIX + line if line else marker for line in quoted]] if quoted else []

    def _list(self, node: Element, budget: int) -> list[list[str]]:
        """One list: its items marked, and anything else in it laid out after them.

        Args:
            node: The ``ul``, ``ol``, ``menu`` or ``dir`` element.
            budget: Columns the text may use, 0 for no wrapping.

        Returns:
            The item block first, then a block for each stray child, since text sitting
            directly inside a list is not an item and has no marker to carry.
        """
        ordered = node.tag == "ol"
        number = _number(node.attr("start"), 1) if ordered else 0
        marked: list[tuple[str, Element]] = []
        stray: list = []
        for child in node.children:
            if isinstance(child, Element) and child.tag == "li":
                if ordered:
                    number = _number(child.attr("value"), number)
                    marked.append((f"{number}. ", child))
                    number += 1
                else:
                    marked.append((BULLET, child))
                continue
            stray.append(child)
        blocks = self._items(marked, budget)
        blocks.extend(self._flow(stray, budget))
        return blocks

    def _items(self, marked: list[tuple[str, Element]], budget: int) -> list[list[str]]:
        """One block holding every item of a list, one item to a line or more.

        Args:
            marked: ``(marker, item)`` pairs, in order.
            budget: Columns the text may use, 0 for no wrapping.

        Returns:
            A single block, or nothing where every item is empty. Markers are padded to one
            width, so ``9.`` and ``10.`` leave their text in the same column.
        """
        width = max((len(marker) for marker, _ in marked), default=0)
        lines: list[str] = []
        for marker, item in marked:
            inner = self._tight(item.children, _reduced(budget, width))
            if not inner:
                lines.append(marker.rstrip())
                continue
            lines.append(marker.ljust(width) + inner[0])
            lines.extend(" " * width + line if line else "" for line in inner[1:])
        return [lines] if lines else []

    def _definitions(self, node: Element, budget: int) -> list[list[str]]:
        """One definition list: each term on its own line, its definitions indented.

        Args:
            node: The ``dl`` element.
            budget: Columns the text may use, 0 for no wrapping.

        Returns:
            The block of terms and definitions, then a block for each stray child, since
            text sitting directly inside a definition list is neither of the two.
        """
        lines: list[str] = []
        stray: list = []
        for child in node.children:
            if not isinstance(child, Element) or child.tag not in ("dd", "dt"):
                stray.append(child)
                continue
            if child.tag == "dt":
                lines.extend(self._tight(child.children, budget))
                continue
            inner = self._tight(child.children, _reduced(budget, len(INDENT)))
            lines.extend(INDENT + line if line else "" for line in inner)
        blocks = [lines] if lines else []
        blocks.extend(self._flow(stray, budget))
        return blocks

    def _table(self, node: Element, budget: int) -> list[list[str]]:
        """One table, as its caption and then a line per row.

        Args:
            node: The ``table`` element.
            budget: Columns the caption may use. The rows are never wrapped.

        Returns:
            The table as one block, then a block for each stray child. Every cell is
            flattened to one line, since a grid holds one line per row: a ``<br>`` or a
            paragraph inside a cell becomes a space. A cell spanning columns is followed by
            the empty cells it covers, so the columns after it still line up. A cell
            spanning rows appears in its own row only.
        """
        caption: list[str] = []
        rows: list[tuple[bool, list[str]]] = []
        stray: list = []
        for child in node.children:
            if not isinstance(child, Element) or child.tag not in TABLE_PARTS:
                stray.append(child)
            elif child.tag == "caption":
                caption.extend(self._lines(child.children, budget))
            elif child.tag == "tr":
                rows.append(self._row(child, False))
            elif child.tag in ROW_GROUPS:
                header = child.tag == "thead"
                rows.extend(self._row(row, header) for row in child.elements(("tr",)))
            else:
                rows.append((child.tag == "th", [self._cell(child)]))
        blocks = [caption + self._laid_out(rows)] if caption or rows else []
        blocks.extend(self._flow(stray, budget))
        return blocks

    def _laid_out(self, rows: list[tuple[bool, list[str]]]) -> list[str]:
        """Every row of a table as one line, each row padded to the same cell count."""
        rows = [(header, cells) for header, cells in rows if cells]
        if not rows:
            return []
        columns = max(len(cells) for _, cells in rows)
        grid = [(header, cells + [""] * (columns - len(cells))) for header, cells in rows]
        return self._grid(grid, columns)

    def _row(self, node: Element, header: bool) -> tuple[bool, list[str]]:
        """One row: whether it is a header row, and the text of each of its cells."""
        cells: list[str] = []
        marked = header
        for cell in node.elements(("td", "th")):
            marked = marked or cell.tag == "th"
            cells.append(self._cell(cell))
            cells.extend([""] * (_span(cell.attr("colspan")) - 1))
        return bool(cells) and marked, cells

    def _cell(self, node: Element) -> str:
        """One cell's text, on one line, with every run of whitespace collapsed."""
        return " ".join(" ".join(self._tight(node.children, 0)).split())

    def _grid(self, grid: list[tuple[bool, list[str]]], columns: int) -> list[str]:
        """Every row as one line, in the shape :data:`TABLE_MODES` names.

        Args:
            grid: ``(is a header row, cells)`` per row, every row the same length.
            columns: How many cells each row holds.

        Returns:
            One line per row. Under ``aligned columns`` a rule sits below the first row
            where that row is a header row, drawn to the same widths.
        """
        if self._tables == "tab separated":
            return ["\t".join(cells) for _, cells in grid]
        widths = [
            max(_display_width(cells[at]) for _, cells in grid) for at in range(columns)
        ]
        lines = []
        for at, (header, cells) in enumerate(grid):
            lines.append(_padded(cells, widths))
            if at == 0 and header:
                lines.append(COLUMN_GAP.join("-" * width for width in widths).rstrip())
        return lines

    # ------------------------------------------------------------------ inline

    def _tight(self, children: list, budget: int) -> list[str]:
        """Every block a container's children make, with no blank line between them."""
        lines: list[str] = []
        for block in self._flow(children, budget):
            lines.extend(block)
        return lines

    def _lines(self, nodes: list, budget: int) -> list[str]:
        """A run of inline content as stripped, wrapped lines."""
        buffer = _Inline()
        self._inline(nodes, buffer)
        return _wrapped(buffer.result(), budget)

    def _inline(self, nodes: list, buffer: _Inline) -> None:
        """Write a run of inline content into ``buffer``.

        Args:
            nodes: The nodes to read, text and elements alike.
            buffer: Where the text is collected.
        """
        for node in nodes:
            if isinstance(node, str):
                buffer.text(node)
            elif node.tag == "br":
                buffer.hard_break()
            elif node.tag == "img":
                self._image(node, buffer)
            elif node.tag == "a":
                self._link(node, buffer)
            elif breaks(node.tag):
                # A block element where only a line of text fits, such as a heading holding
                # a div, still ends the line it is on.
                buffer.hard_break()
                self._inline(node.children, buffer)
                buffer.hard_break()
            else:
                self._inline(node.children, buffer)

    def _image(self, node: Element, buffer: _Inline) -> None:
        """Write what is left of a picture: its alt text, its source, or nothing."""
        if self._images == "skip":
            return
        alt = " ".join(node.attr("alt").split()) or IMAGE_PLACEHOLDER
        source = _source(node.attr("src"))
        if self._images == "alt text and source" and source:
            buffer.verbatim(f"[{alt}: {source}]")
            return
        buffer.verbatim(f"[{alt}]")

    def _link(self, node: Element, buffer: _Inline) -> None:
        """Write a link's words, and its address in the way :data:`LINK_MODES` names."""
        mark = buffer.mark()
        self._inline(node.children, buffer)
        url = _url(node.attr("href"))
        if not url or self._links == "text only":
            return
        shown = buffer.since(mark).strip()
        if not shown:
            # A link with nothing to show, an empty one or a picture that was skipped: the
            # address is the only thing left that says where it goes.
            buffer.verbatim(url)
            return
        if self._links == "footnotes":
            buffer.attached(f"[{self._footnote(url)}]")
            return
        if _bare(shown) != _bare(url):
            buffer.spaced(f"({url})")

    def _footnote(self, url: str) -> int:
        """The number a footnote carries, the same one each time an address repeats."""
        if url not in self._numbered:
            self._footnotes.append(url)
            self._numbered[url] = len(self._footnotes)
        return self._numbered[url]


# ---------------------------------------------------------------------- helpers


def _verbatim(node: Element) -> str:
    """Every character of text inside an element, with a line break for each ``<br>``.

    Args:
        node: The element to read.

    Returns:
        The text as written, whitespace and all. Tags contribute nothing themselves, so
        ``<code>`` inside ``<pre>`` leaves the code alone.
    """
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag == "br":
            parts.append("\n")
        else:
            parts.append(_verbatim(child))
    return "".join(parts)


def _wrapped(lines: list[str], budget: int) -> list[str]:
    """Every line held inside ``budget`` columns.

    Args:
        lines: The lines, already stripped.
        budget: Columns a line may use, 0 for no wrapping.

    Returns:
        The lines, each one longer than the budget replaced by the lines it wraps to. A
        long word is not broken and a hyphenated one is not split, since either rewrites
        the word itself: a line holding one address stays over the budget.
    """
    if not budget:
        return lines
    wrapped: list[str] = []
    for line in lines:
        if not line or len(line) <= budget:
            wrapped.append(line)
            continue
        wrapped.extend(
            textwrap.wrap(line, budget, break_long_words=False, break_on_hyphens=False)
            or [line]
        )
    return wrapped


def _reduced(budget: int, indent: int) -> int:
    """The text column left once ``indent`` characters are spent on indenting it."""
    return max(budget - indent, MIN_WIDTH) if budget else 0


def _number(value: str, fallback: int) -> int:
    """One attribute read as a whole number, or ``fallback`` where it is not one."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _span(value: str) -> int:
    """How many columns one cell covers, at least one and at most :data:`MAX_SPAN`."""
    return min(max(_number(value, 1), 1), MAX_SPAN)


def _padded(cells: list[str], widths: list[int]) -> str:
    """One row of cells padded to ``widths`` and joined, with no trailing spaces."""
    padded = [
        cell + " " * max(width - _display_width(cell), 0)
        for cell, width in zip(cells, widths)
    ]
    return COLUMN_GAP.join(padded).rstrip()


def _display_width(text: str) -> int:
    """How many columns a line takes in a monospaced font.

    Args:
        text: One line.

    Returns:
        Its length, counting an East Asian wide or fullwidth character as two columns and a
        combining mark as none, which is how a terminal and a monospaced font lay both out.
    """
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def _source(src: str) -> str:
    """The source of a picture, short enough to sit in a line of text.

    Args:
        src: The ``src`` attribute.

    Returns:
        The value with its whitespace removed, or just the media type where the attribute
        carries the file itself, since a base64 payload is not a name and can be
        megabytes long. An empty string where there is nothing to show.
    """
    value = "".join(src.split())
    if not value.lower().startswith(_INLINE_DATA):
        return value
    at = min((found for found in (value.find(";"), value.find(",")) if found > 0), default=-1)
    return value[:at] if at > 0 else _INLINE_DATA.rstrip(":")


def _url(href: str) -> str:
    """A link's address, or an empty string where there is nothing worth writing.

    Args:
        href: The ``href`` attribute.

    Returns:
        The address with its whitespace removed. An empty attribute and one naming a place
        inside the document, ``#`` or ``#notes``, come back empty: neither says anything to
        somebody reading the text, since the text has no anchors in it.
    """
    value = "".join(href.split())
    return "" if value.startswith("#") else value


def _bare(value: str) -> str:
    """One address or one piece of link text, reduced to what the two are compared on."""
    stripped = value.strip().rstrip("/")
    for prefix in _IGNORED_PREFIXES:
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].lower()
    return stripped.lower()
