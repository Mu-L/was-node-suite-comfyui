"""Document markup read as blocks and runs, which is the shape a word processor wants.

:func:`to_blocks` walks the element tree and returns a flat list of :class:`Block`. Lengths
are points and colours ``RRGGBB``.
"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .. import tree
from ..text import PREFORMATTED_ELEMENTS
from . import css

__all__ = [
    "Block",
    "Cell",
    "MAX_COLUMNS",
    "MAX_IMAGE_BYTES",
    "MAX_SPAN",
    "Picture",
    "Row",
    "Run",
    "Slot",
    "Style",
    "Table",
    "grid",
    "resolve",
    "to_blocks",
]

#: Most columns one cell may be spread across, and most columns a row may hold. A span of
#: millions in a document's markup would otherwise become a row of empty cells large enough
#: to exhaust memory, and no real table is this wide.
MAX_SPAN = 64
MAX_COLUMNS = 256

#: Largest picture read out of a ``data:`` URL. A container's own files are bounded when
#: the container is read; a data URL is bounded here.
MAX_IMAGE_BYTES = 32 * 1024 * 1024

#: Heading elements and the level each one carries.
HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

#: Elements that turn on one piece of character formatting each.
_BOLD = frozenset({"b", "strong", "th"})
_ITALIC = frozenset({"i", "em", "cite", "dfn", "var"})
_UNDERLINE = frozenset({"u", "ins"})
_STRIKE = frozenset({"s", "strike", "del"})
_MONOSPACE = frozenset({"code", "kbd", "samp", "tt"})

#: Where an element's own attributes carry formatting HTML defined before CSS did.
_FONT_ATTRIBUTES = frozenset({"font", "basefont"})

#: Background a ``<mark>`` is drawn with, which HTML defines as a highlight rather than a
#: colour of its own.
_MARK_BACKGROUND = "FFFF00"

#: Media type of each picture format, by the bytes it starts with. A file whose bytes match
#: none of these keeps the type its name suggests.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)

#: Media type of each picture extension, for a file whose bytes name no format.
_EXTENSION_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".svg": "image/svg+xml",
}

#: Schemes a picture may name that are fetched over the network. Nothing here fetches one.
_REMOTE_SCHEMES = ("http://", "https://", "ftp://", "//")

_DATA_URL = re.compile(r"^data:([^;,]*)(;[^,]*)?,(.*)$", re.DOTALL)

_ASSET_PREFIX = "assets/"


@dataclass(frozen=True)
class Style:
    """The character formatting in force where a piece of text was written.

    Attributes:
        bold: Whether the text is bold.
        italic: Whether the text is italic.
        underline: Whether the text is underlined.
        strike: Whether the text is struck through.
        monospace: Whether the text was written in a code element.
        superscript: Whether the text is raised.
        subscript: Whether the text is lowered.
        color: Text colour as ``RRGGBB``, or ``None``.
        background: Colour behind the text as ``RRGGBB``, or ``None``.
        font: Font family name, or ``None``.
        size: Font size in points, or ``None``.
        link: Address the text links to, or ``None``.
    """

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    monospace: bool = False
    superscript: bool = False
    subscript: bool = False
    color: str | None = None
    background: str | None = None
    font: str | None = None
    size: float | None = None
    link: str | None = None


@dataclass
class Picture:
    """A picture the document draws, with its bytes where they were found.

    Attributes:
        source: The ``src`` as it was written.
        alt: The description written for a reader who cannot see it.
        data: The file's bytes, or ``None`` where they were not found.
        media_type: The picture's media type, such as ``"image/png"``.
        width: Drawn width in points, or ``None`` for the picture's own size.
        height: Drawn height in points, or ``None``.
        remote: Whether the source names somewhere on the network.
    """

    source: str = ""
    alt: str = ""
    data: bytes | None = None
    media_type: str = ""
    width: float | None = None
    height: float | None = None
    remote: bool = False


@dataclass
class Run:
    """One piece of a block: text, a line break, or a picture.

    Attributes:
        text: The characters, empty for a break and for a picture.
        style: The formatting in force.
        picture: The picture drawn here, or ``None``.
        line_break: Whether this is a ``<br>``.
    """

    text: str = ""
    style: Style = field(default_factory=Style)
    picture: Picture | None = None
    line_break: bool = False


@dataclass
class Cell:
    """One cell of a table.

    Attributes:
        blocks: What the cell holds, as blocks of its own.
        header: Whether it was written as ``<th>``.
        colspan: Columns it is spread across, at least 1.
        rowspan: Rows it is spread down, at least 1.
        align: Alignment of its text, or ``None``.
    """

    blocks: list["Block"] = field(default_factory=list)
    header: bool = False
    colspan: int = 1
    rowspan: int = 1
    align: str | None = None


@dataclass
class Row:
    """One row of a table.

    Attributes:
        cells: The cells, in the order they were written.
        header: Whether every cell in it is a header cell.
    """

    cells: list[Cell] = field(default_factory=list)
    header: bool = False


@dataclass
class Table:
    """A table, as rows of cells.

    Attributes:
        rows: The rows, in the order they were written.
        columns: How many columns the widest row occupies, spans counted.
        caption: The caption's runs, empty where there is none.
    """

    rows: list[Row] = field(default_factory=list)
    columns: int = 0
    caption: list[Run] = field(default_factory=list)


@dataclass
class Slot:
    """One position in a table's rectangle of cells.

    Attributes:
        cell: The cell occupying the position.
        origin: Whether the cell was written here, rather than reaching here by a span.
        colspan: Columns the cell covers from here, on an origin slot.
        rowspan: Rows the cell covers from here, on an origin slot.
    """

    cell: Cell
    origin: bool = True
    colspan: int = 1
    rowspan: int = 1


@dataclass
class Block:
    """One block of the document.

    Attributes:
        kind: ``"paragraph"``, ``"heading"``, ``"item"``, ``"preformatted"``, ``"rule"``
            or ``"table"``.
        runs: The block's content, for every kind but a rule, a table and a preformatted
            block.
        level: Heading level 1 to 6, or the depth of a list item starting at 1.
        ordered: Whether a list item is numbered rather than bulleted.
        align: One of :data:`.css.ALIGNMENTS`, or ``None`` for the reader's default.
        quote: How many quotations the block sits inside.
        indent: Further levels of indenting, as a definition's description carries.
        text: A preformatted block's text, exactly as it was written.
        table: The table, for a table block.
    """

    kind: str = "paragraph"
    runs: list[Run] = field(default_factory=list)
    level: int = 0
    ordered: bool = False
    align: str | None = None
    quote: int = 0
    indent: int = 0
    text: str = ""
    table: Table | None = None

    @property
    def empty(self) -> bool:
        """Whether the block would draw nothing at all."""
        if self.kind == "rule":
            return False
        if self.kind == "table":
            return not (self.table and self.table.rows)
        if self.kind == "preformatted":
            return not self.text
        return not any(run.text.strip() or run.picture or run.line_break for run in self.runs)


def to_blocks(markup: str, assets: Mapping[str, bytes] | None = None) -> list[Block]:
    """Read document markup as a list of blocks.

    Args:
        markup: HTML, as ``content.html`` holds it. A fragment and a whole document are
            both accepted, and neither has to be well formed.
        assets: The document's embedded files, keyed as :class:`..container.Document`
            keys them, which is relative to ``assets/``.

    Returns:
        The blocks in document order. Text outside any element becomes a paragraph, an
        empty block is dropped, and a document holding nothing readable gives an empty
        list.
    """
    walker = _Walker(assets or {})
    walker.walk(tree.parse(markup), Style(), _Context())
    walker.flush()
    return [block for block in walker.blocks if not block.empty]


def resolve(source: str, assets: Mapping[str, bytes] | None = None) -> Picture:
    """Find the bytes one ``src`` names, without reaching outside the document.

    Args:
        source: The ``src`` as it was written: a name under ``assets/``, a name relative to
            it, a ``data:`` URL, or a web address.
        assets: The document's embedded files, keyed relative to ``assets/``.

    Returns:
        A :class:`Picture` carrying the bytes and the media type where they were found. A
        web address sets ``remote`` and carries no bytes: nothing here fetches one, so a
        picture from the network is never drawn.
    """
    text = (source or "").strip()
    picture = Picture(source=text)
    if any(text.lower().startswith(scheme) for scheme in _REMOTE_SCHEMES):
        picture.remote = True
    elif text.lower().startswith("data:"):
        picture.data, picture.media_type = _data_url(text)
    else:
        picture.data = _embedded(text, assets or {})
    if not picture.media_type:
        picture.media_type = _media_type(picture.data, text)
    return picture


def _embedded(source: str, assets: Mapping[str, bytes]) -> bytes | None:
    """The document's own file one ``src`` names, or ``None``."""
    name = urllib.parse.unquote(source.split("?")[0].split("#")[0]).replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    candidates = [name]
    if name.startswith(_ASSET_PREFIX):
        candidates.append(name[len(_ASSET_PREFIX):])
    for key in candidates:
        found = assets.get(key)
        if found is not None:
            return found
    return None


def grid(table: Table) -> list[list[Slot]]:
    """Lay a table out as a rectangle, resolving every span.

    Args:
        table: The table read out of the markup, whose rows may hold cells spread across
            columns and down rows.

    Returns:
        One list of :class:`Slot` per row, every row the same length. A span is clamped to
        what the table actually holds, so nothing reaches past the last column or the last
        row, and a position no cell reaches carries an empty cell of its own.
    """
    columns = max(1, min(table.columns, MAX_COLUMNS))
    height = len(table.rows)
    placed: list[list[Slot | None]] = [[None] * columns for _ in range(height)]
    for index, row in enumerate(table.rows):
        column = 0
        for cell in row.cells:
            while column < columns and placed[index][column] is not None:
                column += 1
            if column >= columns:
                break
            across = min(cell.colspan, columns - column)
            down = min(cell.rowspan, height - index)
            placed[index][column] = Slot(cell, True, across, down)
            for step_down in range(down):
                for step_across in range(across):
                    if step_down or step_across:
                        placed[index + step_down][column + step_across] = Slot(cell, False)
            column += across
    return [[slot if slot else Slot(Cell()) for slot in row] for row in placed]


@dataclass
class _Context:
    """Where in the document the walk currently is.

    Attributes:
        kind: What a block closed here is: a paragraph, a heading or a list item.
        level: Heading level, or the depth of the list item being read.
        ordered: Whether the list item being read is numbered.
        align: Alignment inherited from an enclosing block.
        quote: How many quotations enclose it.
        indent: Further levels of indenting.
        depth: How many lists enclose it.
    """

    kind: str = "paragraph"
    level: int = 0
    ordered: bool = False
    align: str | None = None
    quote: int = 0
    indent: int = 0
    depth: int = 0


class _Walker:
    """Turns one element tree into blocks.

    Attributes:
        blocks: What has been read so far, in document order.
    """

    def __init__(self, assets: Mapping[str, bytes]) -> None:
        """Read a document against the files embedded in it.

        Args:
            assets: The document's embedded files, keyed relative to ``assets/``.
        """
        self.blocks: list[Block] = []
        self._assets = assets
        self._runs: list[Run] = []
        self._space = False
        self._open = Block()

    # -------------------------------------------------------------- the walk

    def walk(self, node: tree.Element, style: Style, ctx: _Context) -> None:
        """Read one element's children into blocks.

        Args:
            node: The element whose children are read.
            style: The character formatting in force.
            ctx: Where in the document the walk is.
        """
        for child in node.children:
            if isinstance(child, str):
                self._text(child, style, ctx)
                continue
            tag = child.tag
            if tag == "br":
                self._line_break(style, ctx)
            elif tag == "img":
                self._picture(child, style, ctx)
            elif not tree.breaks(tag):
                self.walk(child, _inline_style(child, style), ctx)
            else:
                self._block(child, style, ctx)

    def flush(self) -> None:
        """Close the block being collected and start a new one."""
        if self._runs:
            self.blocks.append(replace(self._open, runs=self._runs))
        self._runs = []
        self._space = False
        self._open = Block()

    def _block(self, node: tree.Element, style: Style, ctx: _Context) -> None:
        """Read one block-level element."""
        tag = node.tag
        inner = _inline_style(node, style)
        # A block written inside a list item keeps the item's indenting, so a paragraph
        # under a bullet stays in the same column as the item's own text.
        indent = ctx.indent
        if ctx.depth and tag != "li" and tag not in tree.LIST_ELEMENTS:
            indent = min(ctx.indent + ctx.depth, tree.MAX_DEPTH)
        ctx = replace(
            ctx,
            align=css.alignment(_declared_align(node)) or ctx.align,
            kind="paragraph",
            level=0,
            ordered=False,
            indent=indent,
        )
        self.flush()
        if tag == "hr":
            self.blocks.append(Block(kind="rule", quote=ctx.quote, indent=ctx.indent))
        elif tag in HEADING_LEVELS:
            self.walk(node, inner, replace(ctx, kind="heading", level=HEADING_LEVELS[tag]))
        elif tag in PREFORMATTED_ELEMENTS:
            self._preformatted(node, ctx)
        elif tag == "blockquote":
            self.walk(node, inner, replace(ctx, quote=min(ctx.quote + 1, tree.MAX_DEPTH)))
        elif tag in tree.LIST_ELEMENTS:
            self._list(node, inner, ctx)
        elif tag == "li":
            self._item(node, inner, ctx, max(ctx.depth, 1), ctx.ordered)
        elif tag == "table":
            self._table(node, inner, ctx)
        elif tag == "dd":
            self.walk(node, inner, replace(ctx, indent=min(ctx.indent + 1, tree.MAX_DEPTH)))
        else:
            # Anything else, a row or a cell outside a table included. A browser lifts the
            # text of a stray table part out, so it is read as an ordinary block rather
            # than dropped.
            self.walk(node, inner, ctx)
        self.flush()

    def _list(self, node: tree.Element, style: Style, ctx: _Context) -> None:
        """Read a list, one block per item, deeper lists carrying a deeper level."""
        ordered = node.tag == "ol"
        depth = min(ctx.depth + 1, tree.MAX_DEPTH)
        inner = replace(ctx, depth=depth, ordered=ordered)
        for child in node.children:
            if isinstance(child, str):
                self._text(child, style, inner)
                continue
            if child.tag == "li":
                self._item(child, _inline_style(child, style), inner, depth, ordered)
            else:
                self._block(child, style, inner)
        self.flush()

    def _item(
        self, node: tree.Element, style: Style, ctx: _Context, depth: int, ordered: bool
    ) -> None:
        """Read one list item, closing it before any list nested inside it."""
        self.flush()
        self.walk(node, style, replace(ctx, kind="item", level=depth, ordered=ordered))
        self.flush()

    def _preformatted(self, node: tree.Element, ctx: _Context) -> None:
        """Read a preformatted block, keeping its whitespace exactly as written."""
        text = _verbatim(node).strip("\n")
        self.blocks.append(
            Block(kind="preformatted", text=text, quote=ctx.quote, indent=ctx.indent)
        )

    def _table(self, node: tree.Element, style: Style, ctx: _Context) -> None:
        """Read a table into rows of cells, each cell holding blocks of its own."""
        table = Table()
        caption = node.elements({"caption"})
        if caption:
            inner = _Walker(self._assets)
            inner.walk(caption[0], style, _Context())
            inner.flush()
            for block in inner.blocks:
                table.caption.extend(block.runs)
        for row_node in _rows(node):
            row = Row()
            for cell_node in row_node.elements({"td", "th"}):
                row.cells.append(self._cell(cell_node, style, ctx))
            if row.cells:
                row.header = all(cell.header for cell in row.cells)
                table.rows.append(row)
        table.columns = min(
            max((sum(cell.colspan for cell in row.cells) for row in table.rows), default=0),
            MAX_COLUMNS,
        )
        if table.rows:
            self.blocks.append(
                Block(kind="table", table=table, quote=ctx.quote, indent=ctx.indent)
            )
        # Anything in the table that is not part of one is read after it, as a browser
        # lifts such text out rather than dropping it.
        for child in node.children:
            if isinstance(child, tree.Element) and child.tag not in tree.TABLE_PARTS:
                self._block(child, style, ctx)

    def _cell(self, node: tree.Element, style: Style, ctx: _Context) -> Cell:
        """Read one cell, with its span and its own blocks."""
        inner = _Walker(self._assets)
        cell_style = _inline_style(node, style)
        cell_ctx = _Context(align=css.alignment(_declared_align(node)))
        inner.walk(node, cell_style, cell_ctx)
        inner.flush()
        return Cell(
            blocks=[block for block in inner.blocks if not block.empty],
            header=node.tag == "th",
            colspan=_span(node.attr("colspan")),
            rowspan=_span(node.attr("rowspan")),
            align=cell_ctx.align,
        )

    # ------------------------------------------------------------- the pieces

    def _text(self, data: str, style: Style, ctx: _Context) -> None:
        """Add text to the block being collected, collapsing its whitespace."""
        words = data.split()
        if not words:
            self._space = self._space or bool(self._runs)
            return
        prefix = " " if (self._space or data[:1].isspace()) and self._written() else ""
        self._add(Run(text=prefix + " ".join(words), style=style), ctx)
        self._space = bool(data[-1:].isspace())

    def _line_break(self, style: Style, ctx: _Context) -> None:
        """Add a line break inside the block being collected."""
        self._add(Run(style=style, line_break=True), ctx)
        self._space = False

    def _picture(self, node: tree.Element, style: Style, ctx: _Context) -> None:
        """Add a picture to the block being collected."""
        picture = resolve(node.attr("src"), self._assets)
        picture.alt = node.attr("alt").strip()
        picture.width = _dimension(node, "width")
        picture.height = _dimension(node, "height")
        self._add(Run(style=style, picture=picture), ctx)
        self._space = False

    def _add(self, run: Run, ctx: _Context) -> None:
        """Append a run, joining it to the one before where both read the same."""
        if not self._runs:
            self._open = Block(
                kind=ctx.kind,
                level=ctx.level,
                ordered=ctx.ordered,
                align=ctx.align,
                quote=ctx.quote,
                indent=ctx.indent,
            )
        if (
            self._runs
            and run.picture is None
            and not run.line_break
            and self._runs[-1].picture is None
            and not self._runs[-1].line_break
            and self._runs[-1].style == run.style
        ):
            self._runs[-1].text += run.text
            return
        self._runs.append(run)

    def _written(self) -> bool:
        """Whether the block being collected already holds something a space follows."""
        if not self._runs:
            return False
        last = self._runs[-1]
        if last.line_break:
            return False
        return bool(last.picture) or bool(last.text and not last.text.endswith(" "))


def _inline_style(node: tree.Element, style: Style) -> Style:
    """The formatting inside one element, from its tag, its attributes and its style.

    Args:
        node: The element being entered.
        style: The formatting in force outside it.

    Returns:
        A new :class:`Style`. A declaration in the ``style`` attribute wins over the tag
        and over a presentational attribute, as CSS does.
    """
    tag = node.tag
    changes: dict[str, Any] = {}
    if tag in _BOLD:
        changes["bold"] = True
    if tag in _ITALIC:
        changes["italic"] = True
    if tag in _UNDERLINE:
        changes["underline"] = True
    if tag in _STRIKE:
        changes["strike"] = True
    if tag in _MONOSPACE:
        changes["monospace"] = True
    if tag == "sup":
        changes["superscript"] = True
    if tag == "sub":
        changes["subscript"] = True
    if tag == "mark":
        changes["background"] = _MARK_BACKGROUND
    if tag == "a":
        href = node.attr("href").strip()
        if href:
            changes["link"] = href
    if tag in _FONT_ATTRIBUTES:
        changes.update(_font_attributes(node, style))
    changes.update(_style_attribute(node, style))
    return replace(style, **changes) if changes else style


def _style_attribute(node: tree.Element, style: Style) -> dict[str, Any]:
    """The formatting one element's ``style`` attribute declares."""
    declared = css.declarations(node.attr("style"))
    if not declared:
        return {}
    changes: dict[str, Any] = {}
    weight = declared.get("font-weight", "").strip().lower()
    if weight:
        changes["bold"] = weight in ("bold", "bolder") or _weight(weight) >= 600
    slant = declared.get("font-style", "").strip().lower()
    if slant:
        changes["italic"] = slant in ("italic", "oblique")
    decoration = " ".join(
        declared.get(name, "") for name in ("text-decoration", "text-decoration-line")
    ).lower()
    if decoration:
        changes["underline"] = "underline" in decoration
        changes["strike"] = "line-through" in decoration
    vertical = declared.get("vertical-align", "").strip().lower()
    if vertical in ("super", "sub"):
        changes["superscript"] = vertical == "super"
        changes["subscript"] = vertical == "sub"
    found = css.color(declared.get("color", ""))
    if found:
        changes["color"] = found
    background = css.color(declared.get("background-color", "")) or css.color(
        declared.get("background", "")
    )
    if background:
        changes["background"] = background
    family = css.font_family(declared.get("font-family", ""))
    if family:
        changes["font"] = family
    size = css.length(declared.get("font-size", ""), style.size or css.BASE_FONT_POINTS)
    if size:
        changes["size"] = size
    return changes


def _font_attributes(node: tree.Element, style: Style) -> dict[str, Any]:
    """The formatting a ``<font>`` element's own attributes declare."""
    changes: dict[str, Any] = {}
    found = css.color(node.attr("color"))
    if found:
        changes["color"] = found
    face = css.font_family(node.attr("face"))
    if face:
        changes["font"] = face
    size = _font_size(node.attr("size"), style.size or css.BASE_FONT_POINTS)
    if size:
        changes["size"] = size
    return changes


#: Point size of each ``<font size>`` step, which HTML numbers 1 to 7.
_FONT_STEPS = (7.5, 10.0, 12.0, 13.5, 18.0, 24.0, 36.0)


def _font_size(value: str, base: float) -> float | None:
    """A ``<font size>`` value as points, absolute or relative to the size in force."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    if text[0] in "+-":
        step = 2 + number
    else:
        step = number - 1
    if 0 <= step < len(_FONT_STEPS):
        return _FONT_STEPS[step]
    return base


def _weight(value: str) -> int:
    """A numeric ``font-weight``, or 400 where it is not a number."""
    try:
        return int(float(value))
    except ValueError:
        return 400


def _declared_align(node: tree.Element) -> str:
    """The alignment one element declares, from its ``style`` or its ``align``."""
    declared = css.declarations(node.attr("style")).get("text-align", "")
    return declared or node.attr("align")


def _dimension(node: tree.Element, name: str) -> float | None:
    """A picture's drawn width or height in points, from its style or its attribute."""
    declared = css.declarations(node.attr("style")).get(name, "")
    return css.length(declared) if declared else css.length(node.attr(name))


def _span(value: str) -> int:
    """A ``colspan`` or ``rowspan`` value, at least 1 and at most :data:`MAX_SPAN`."""
    try:
        number = int(str(value).strip() or 1)
    except ValueError:
        return 1
    return max(1, min(number, MAX_SPAN))


def _rows(node: tree.Element) -> list[tree.Element]:
    """Every row of a table, read through whatever groups them."""
    found: list[tree.Element] = []
    for child in node.children:
        if not isinstance(child, tree.Element):
            continue
        if child.tag == "tr":
            found.append(child)
        elif child.tag in tree.ROW_GROUPS:
            found.extend(_rows(child))
    return found


def _verbatim(node: tree.Element) -> str:
    """Every character inside one element, with its whitespace kept as written."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag == "br":
            parts.append("\n")
        else:
            parts.append(_verbatim(child))
            if tree.breaks(child.tag):
                parts.append("\n")
    return "".join(parts)


def _data_url(source: str) -> tuple[bytes | None, str]:
    """The bytes and media type behind a ``data:`` URL, or ``(None, "")``."""
    match = _DATA_URL.match(source)
    if not match:
        return None, ""
    media_type = match.group(1).strip() or "text/plain"
    payload = match.group(3)
    if len(payload) > MAX_IMAGE_BYTES:
        return None, media_type
    if ";base64" in (match.group(2) or "").lower():
        try:
            return base64.b64decode(payload, validate=False), media_type
        except (binascii.Error, ValueError):
            return None, media_type
    try:
        return urllib.parse.unquote_to_bytes(payload), media_type
    except (UnicodeError, ValueError):
        return None, media_type


def _media_type(data: bytes | None, source: str) -> str:
    """The media type of a picture, from its first bytes or from its name."""
    if data:
        for signature, media_type in _SIGNATURES:
            if data.startswith(signature):
                return media_type
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
    name = source.split("?")[0].split("#")[0].lower()
    for extension, media_type in _EXTENSION_TYPES.items():
        if name.endswith(extension):
            return media_type
    return "application/octet-stream"
