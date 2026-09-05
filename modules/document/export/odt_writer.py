"""A document written as an ``.odt`` file through odfdo.

Every block and run is mapped onto ODF elements, with character and paragraph formatting in
automatic styles. Metadata goes into ``meta.xml``, with the author's own pairs as
user-defined fields.
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import deps, log
from ..summary import keywords_text
from . import FEATURE, Page, blocks

if TYPE_CHECKING:  # pragma: no cover - imported for the annotation alone
    from ..container import Document

__all__ = ["BULLETS", "MONOSPACE_FONT", "write"]

logger = log.get_logger("document.export.odt")

#: Font a code element and a preformatted block are written in.
MONOSPACE_FONT = "Liberation Mono"

#: Characters the three list levels are bulleted with, repeating past the third.
BULLETS = ("•", "◦", "▪")

#: Levels a list style is written for. ODF numbers a level past this from the last one
#: declared, so ten covers any list a document holds.
LIST_LEVELS = 10

#: Centimetres of indenting one list level adds, and the width left for its label.
LIST_INDENT_CM = 1.0
LIST_LABEL_CM = 0.6

#: Points one level of quotation or definition indenting adds.
INDENT_STEP = 18.0

#: Name of the list styles, and the stem every automatic style name is built from.
_NUMBER_STYLE = "WASNumberList"
_BULLET_STYLE = "WASBulletList"
_TEXT_STEM = "WAST"
_PARAGRAPH_STEM = "WASP"

#: Base style each kind of paragraph inherits from, as ODF spells the names.
_STANDARD = "Standard"
_QUOTATION = "Quotations"
_PREFORMATTED = "Preformatted_20_Text"
_HEADING = "Heading_20_{level}"
_CAPTION = "Illustration"

#: How the four alignments are spelled in ``fo:text-align``. ``start`` and ``end`` follow
#: the writing direction, which is what a word processor writes for left and right.
_ALIGNMENTS = {"left": "start", "right": "end", "center": "center", "justify": "justify"}

#: Where a picture is stored inside the package.
_PICTURES = "Pictures"


def write(document: "Document", page: Page) -> bytes:
    """Write a document as an ``.odt`` file.

    Args:
        document: The document to convert.
        page: The page to lay it out on.

    Returns:
        The file's bytes.

    Raises:
        DependencyError: odfdo is missing or unusable.
    """
    deps.require("odfdo", feature=FEATURE)
    import odfdo

    text = odfdo.Document("text")
    body = text.body
    body.clear()
    _page(text, page)
    _metadata(text, document)
    _Writer(text, page).document(blocks.to_blocks(document.content, document.assets))
    return _serialized(text)


class _Writer:
    """Writes blocks into one odfdo document.

    Attributes:
        text: The document being built.
        page: The page it is laid out on.
    """

    def __init__(self, text, page: Page) -> None:
        """Write into one document.

        Args:
            text: The odfdo ``Document`` to write into.
            page: The page it is laid out on.
        """
        self.text = text
        self.page = page
        self._styles: dict[tuple, str] = {}
        self._lists: dict[bool, str | None] = {}
        self._pictures: dict[str, str | None] = {}
        self._reported: set[str] = set()
        self._dropped = 0

    def document(self, found: list[blocks.Block]) -> None:
        """Write every block, rebuilding list nesting as the levels change."""
        body = self.text.body
        stack: list[Any] = []
        for block in found:
            if block.kind == "item":
                self._item(body, stack, block)
                continue
            stack.clear()
            if block.kind == "table" and block.table and block.table.caption:
                self._caption(body, block.table.caption)
            element = self._element(block)
            if element is not None:
                body.append(element)
        if self._dropped:
            logger.warning(
                "%d picture(s) could not be drawn in the exported .odt and its description "
                "was written in its place. A picture is drawn when the document carries the "
                "file itself; one named by a web address is never fetched.",
                self._dropped,
            )

    def _caption(self, body, runs: list[blocks.Run]) -> None:
        """Write a table's caption as a paragraph above it."""
        import odfdo

        paragraph = odfdo.Paragraph(
            style=self._paragraph_style(blocks.Block(), _CAPTION)
        )
        self._runs(paragraph, runs)
        body.append(paragraph)

    # -------------------------------------------------------------- the blocks

    def _element(self, block: blocks.Block):
        """One block as an ODF element, or ``None`` where it draws nothing."""
        import odfdo

        if block.kind == "heading":
            level = max(1, min(block.level, 6))
            heading = odfdo.Header(level, style=self._paragraph_style(block))
            self._runs(heading, block.runs)
            return heading
        if block.kind == "table":
            return self._table(block)
        if block.kind == "preformatted":
            return self._preformatted(block)
        if block.kind == "rule":
            return odfdo.Paragraph(style=self._rule_style(block))
        paragraph = odfdo.Paragraph(style=self._paragraph_style(block))
        self._runs(paragraph, block.runs)
        return paragraph

    def _item(self, body, stack: list, block: blocks.Block) -> None:
        """Write one list item, opening and closing lists as the level changes."""
        import odfdo

        depth = max(1, block.level)
        while len(stack) > depth:
            stack.pop()
        while len(stack) < depth:
            new_list = odfdo.List(style=self._list_style(block.ordered))
            if stack:
                holder = odfdo.ListItem()
                holder.append(new_list)
                stack[-1].append(holder)
            else:
                body.append(new_list)
            stack.append(new_list)
        item = odfdo.ListItem()
        paragraph = odfdo.Paragraph(style=self._paragraph_style(block))
        self._runs(paragraph, block.runs)
        item.append(paragraph)
        stack[-1].append(item)

    def _preformatted(self, block: blocks.Block):
        """Write a preformatted block, keeping its spaces and its line breaks."""
        import odfdo

        paragraph = odfdo.Paragraph(style=self._paragraph_style(block, _PREFORMATTED))
        style = blocks.Style(monospace=True)
        for index, line in enumerate(block.text.split("\n")):
            if index:
                self._append(paragraph, odfdo.LineBreak())
            self._span(paragraph, line, style, formatted=True)
        return paragraph

    def _table(self, block: blocks.Block):
        """Write a table, declaring every span and writing its covered cells."""
        import odfdo

        table = block.table
        matrix = blocks.grid(table)
        if not matrix or not matrix[0]:
            return None
        element = odfdo.Table("Table")
        self._columns(element, len(matrix[0]))
        for slots in matrix:
            row = odfdo.Row()
            for slot in slots:
                row.append(self._cell(slot))
            element.append(row)
        return element

    def _columns(self, element, columns: int) -> None:
        """Declare the table's columns, which ODF expects before its rows."""
        import odfdo

        try:
            element.append(odfdo.Column(repeated=columns))
        except Exception as error:
            self._once("columns", "the exported table declares no columns (%s)", error)

    def _cell(self, slot: blocks.Slot):
        """One table cell, or the covered cell a span reaches."""
        import odfdo

        if not slot.origin:
            covered = self._covered()
            if covered is not None:
                return covered
        cell = odfdo.Cell()
        if slot.origin and (slot.colspan > 1 or slot.rowspan > 1):
            self._span_attributes(cell, slot)
        for block in slot.cell.blocks or [blocks.Block()]:
            element = self._element(block)
            if element is not None:
                cell.append(element)
        return cell

    def _covered(self):
        """A covered cell, which is what ODF puts where a span reaches."""
        import odfdo

        try:
            return odfdo.Element.from_tag("table:covered-table-cell")
        except Exception as error:
            self._once("covered", "a merged cell was written as an empty one (%s)", error)
            return None

    def _span_attributes(self, cell, slot: blocks.Slot) -> None:
        """Declare how far one cell spans."""
        try:
            if slot.colspan > 1:
                cell.set_attribute("table:number-columns-spanned", str(slot.colspan))
            if slot.rowspan > 1:
                cell.set_attribute("table:number-rows-spanned", str(slot.rowspan))
        except Exception as error:
            self._once("spans", "a cell was written unmerged (%s)", error)

    # ---------------------------------------------------------------- the runs

    def _runs(self, paragraph, found: list[blocks.Run]) -> None:
        """Write every run of one block into a paragraph or a heading."""
        import odfdo

        for run in found:
            if run.line_break:
                self._append(paragraph, odfdo.LineBreak())
            elif run.picture is not None:
                self._picture(paragraph, run.picture)
            elif run.text:
                self._span(paragraph, run.text, run.style)

    def _span(self, paragraph, text: str, style: blocks.Style, formatted: bool = False) -> None:
        """Write one run of text, inside a span and a link where it needs them."""
        import odfdo

        name = self._text_style(style)
        content: Any = text
        if formatted:
            content = odfdo.Span(style=name)
            self._formatted(content, text)
        elif name:
            content = odfdo.Span(text, style=name)
        if style.link:
            link = odfdo.Link(style.link)
            self._append(link, content)
            content = link
        self._append(paragraph, content)

    def _formatted(self, target, text: str) -> None:
        """Write text with its runs of spaces and its tabs kept as they are written."""
        appender = getattr(target, "append_plain_text", None)
        if callable(appender):
            try:
                appender(text)
                return
            except Exception as error:
                self._once("plain", "preformatted spacing may be lost (%s)", error)
        self._append(target, text)

    def _append(self, target, value) -> None:
        """Add text or an element to a paragraph, reporting a refusal once."""
        try:
            target.append(value)
        except Exception as error:
            self._once("append", "part of a paragraph was not written (%s)", error)

    def _picture(self, paragraph, picture: blocks.Picture) -> None:
        """Draw a picture in the line, or write its description in its place."""
        import odfdo

        stored = self._stored(picture)
        if stored:
            try:
                width, height = self._size(picture)
                frame = odfdo.Frame.image_frame(
                    stored, size=(width, height), anchor_type="as-char"
                )
                paragraph.append(frame)
                return
            except Exception as error:
                self._once("frame", "a picture was written as its description (%s)", error)
        self._dropped += 1
        self._append(paragraph, f"[{picture.alt or picture.source or 'picture'}]")

    def _size(self, picture: blocks.Picture) -> tuple[str, str]:
        """The width and height a picture is drawn at, as ODF lengths."""
        limit = self.page.text_width
        width = min(picture.width or limit / 2, limit)
        height = picture.height or width * 0.75
        return f"{width:.2f}pt", f"{height:.2f}pt"

    def _stored(self, picture: blocks.Picture) -> str | None:
        """The path a picture is stored at inside the package, storing it once."""
        if not picture.data:
            if picture.remote:
                logger.debug("%r is a web address and is never fetched", picture.source)
            return None
        key = picture.source
        if key in self._pictures:
            return self._pictures[key]
        stored = self._store(picture)
        self._pictures[key] = stored
        return stored

    def _store(self, picture: blocks.Picture) -> str | None:
        """Put one picture's bytes into the package, by whichever route odfdo offers."""
        name = _picture_name(picture, len(self._pictures))
        stream = BytesIO(picture.data)
        try:
            stream.name = name
        except AttributeError:
            pass
        try:
            return self.text.add_file(stream)
        except Exception as error:
            logger.debug("odfdo would not take %r as a file (%s)", name, error)
        try:
            path = f"{_PICTURES}/{name}"
            self.text.set_part(path, picture.data)
            manifest = self.text.get_part("META-INF/manifest.xml")
            manifest.add_full_path(path, picture.media_type)
            return path
        except Exception as error:
            self._once("store", "a picture could not be put into the package (%s)", error)
            return None

    # -------------------------------------------------------------- the styles

    def _text_style(self, style: blocks.Style) -> str | None:
        """The automatic text style one run needs, or ``None`` where it needs none."""
        properties: dict[str, str] = {}
        if style.bold:
            properties["fo:font-weight"] = "bold"
        if style.italic:
            properties["fo:font-style"] = "italic"
        if style.underline or style.link:
            properties["style:text-underline-style"] = "solid"
            properties["style:text-underline-width"] = "auto"
        if style.strike:
            properties["style:text-line-through-style"] = "solid"
        if style.superscript:
            properties["style:text-position"] = "super 58%"
        if style.subscript:
            properties["style:text-position"] = "sub 58%"
        if style.color:
            properties["fo:color"] = f"#{style.color.lower()}"
        if style.background:
            properties["fo:background-color"] = f"#{style.background.lower()}"
        name = style.font or (MONOSPACE_FONT if style.monospace else None)
        if name:
            properties["style:font-name"] = name
            properties["fo:font-family"] = name
        if style.size:
            properties["fo:font-size"] = f"{style.size:.1f}pt"
        return self._style("text", properties, None)

    def _paragraph_style(self, block: blocks.Block, parent: str | None = None) -> str | None:
        """The automatic paragraph style one block needs, or ``None``."""
        requested = parent
        properties: dict[str, str] = {}
        align = _ALIGNMENTS.get(block.align or "")
        if align:
            properties["fo:text-align"] = align
            properties["style:justify-single-word"] = "false"
        steps = block.indent + block.quote
        if steps:
            properties["fo:margin-left"] = f"{INDENT_STEP * steps:.1f}pt"
        if parent is None:
            if block.kind == "heading":
                parent = _HEADING.format(level=max(1, min(block.level, 10)))
            elif block.quote:
                parent = _QUOTATION
            else:
                parent = _STANDARD
        if not properties and requested is None:
            # A heading already reads as its level and a paragraph as the body style, so
            # neither needs a style of its own until it carries formatting.
            return None
        return self._style("paragraph", properties, parent)

    def _rule_style(self, block: blocks.Block) -> str | None:
        """The paragraph style a horizontal rule is drawn with."""
        properties = {
            "fo:border-bottom": "0.06pt solid #808080",
            "fo:padding-bottom": "2pt",
            "fo:margin-bottom": "6pt",
        }
        steps = block.indent + block.quote
        if steps:
            properties["fo:margin-left"] = f"{INDENT_STEP * steps:.1f}pt"
        return self._style("paragraph", properties, _STANDARD)

    def _style(self, family: str, properties: dict[str, str], parent: str | None) -> str | None:
        """Build and register one automatic style, or reuse the identical one already made.

        Args:
            family: ``"text"`` or ``"paragraph"``.
            properties: The ODF properties it carries.
            parent: The named style it inherits from, or ``None``.

        Returns:
            The style's name, or ``None`` when there is nothing to declare or when odfdo
            would not take it, in which case the content is written unstyled.
        """
        import odfdo

        if not properties and not parent:
            return None
        key = (family, parent, tuple(sorted(properties.items())))
        if key in self._styles:
            return self._styles[key]
        stem = _TEXT_STEM if family == "text" else _PARAGRAPH_STEM
        name = f"{stem}{len(self._styles) + 1}"
        try:
            style = (
                odfdo.Style(family, name=name, parent_style=parent)
                if parent
                else odfdo.Style(family, name=name)
            )
            if properties:
                style.set_properties(properties)
            self.text.insert_style(style, automatic=True)
        except Exception as error:
            self._once(
                f"style:{family}",
                "the exported .odt keeps its text and loses some of its formatting (%s)",
                error,
            )
            self._styles[key] = None
            return None
        self._styles[key] = name
        return name

    def _list_style(self, ordered: bool) -> str | None:
        """The list style numbered or bulleted lists are written with."""
        if ordered in self._lists:
            return self._lists[ordered]
        name = _NUMBER_STYLE if ordered else _BULLET_STYLE
        declared = _register_list_style(self.text, name, ordered)
        self._lists[ordered] = name if declared else None
        return self._lists[ordered]

    def _once(self, key: str, message: str, *args) -> None:
        """Report one kind of refusal a single time per document."""
        if key in self._reported:
            return
        self._reported.add(key)
        logger.warning("odt export: " + message, *args)


def _register_list_style(text, name: str, ordered: bool) -> bool:
    """Declare a list style so numbering and bullets are the document's own.

    Args:
        text: The odfdo document.
        name: Name the style is declared under.
        ordered: Whether items are numbered rather than bulleted.

    Returns:
        Whether the style was declared. Without it a reader falls back to its own default
        marker, which for a numbered list is usually a bullet.
    """
    import odfdo

    levels = []
    for level in range(1, LIST_LEVELS + 1):
        indent = LIST_INDENT_CM * level
        properties = (
            "<style:list-level-properties "
            f'text:space-before="{indent:.2f}cm" '
            f'text:min-label-width="{LIST_LABEL_CM:.2f}cm"/>'
        )
        if ordered:
            levels.append(
                f'<text:list-level-style-number text:level="{level}" '
                f'style:num-suffix="." style:num-format="1">{properties}'
                "</text:list-level-style-number>"
            )
        else:
            bullet = BULLETS[(level - 1) % len(BULLETS)]
            levels.append(
                f'<text:list-level-style-bullet text:level="{level}" '
                f'text:bullet-char="{bullet}">{properties}'
                "</text:list-level-style-bullet>"
            )
    markup = f'<text:list-style style:name="{name}">{"".join(levels)}</text:list-style>'
    try:
        text.insert_style(odfdo.Element.from_tag(markup), automatic=False)
        return True
    except Exception as error:
        logger.warning(
            "odt export: the %s list style was not declared, so a reader marks those items "
            "its own way (%s)", "numbered" if ordered else "bulleted", error,
        )
        return False


def _page(text, page: Page) -> None:
    """Set the page size, orientation and margins on the document's page layout."""
    properties = {
        "fo:page-width": f"{page.width:.2f}pt",
        "fo:page-height": f"{page.height:.2f}pt",
        "style:print-orientation": (
            "landscape" if page.width > page.height else "portrait"
        ),
        "fo:margin-top": f"{page.margin:.2f}pt",
        "fo:margin-bottom": f"{page.margin:.2f}pt",
        "fo:margin-left": f"{page.margin:.2f}pt",
        "fo:margin-right": f"{page.margin:.2f}pt",
    }
    try:
        layouts = text.get_styles("page-layout")
        if not layouts:
            raise LookupError("this odfdo template declares no page layout")
        for layout in layouts:
            layout.set_properties(properties)
    except Exception as error:
        logger.warning(
            "odt export: the page size and margins were left as the template's own (%s)",
            error,
        )


def _metadata(text, document: "Document") -> None:
    """Write the document's metadata into ``meta.xml``."""
    metadata = document.metadata
    meta = getattr(text, "meta", None) or text.get_part("meta.xml")
    _set(meta, "title", metadata.title)
    _set(meta, "description", metadata.description)
    _set(meta, "creator", metadata.author)
    _set(meta, "language", metadata.language)
    _set(meta, "generator", metadata.generator)
    _set(meta, "keywords", keywords_text(metadata.keywords))
    _set(meta, "creation_date", metadata.created)
    _set(meta, "modification_date", metadata.modified)
    user_defined = dict(metadata.custom)
    if metadata.copyright:
        user_defined.setdefault("Copyright", metadata.copyright)
    if user_defined:
        _set(meta, "user_defined_metadata", user_defined)


def _set(meta, field: str, value) -> None:
    """Write one metadata field, by whichever spelling the installed odfdo offers."""
    if not value:
        return
    if field.endswith("_date"):
        value = _datetime(value)
        if value is None:
            return
    declared = getattr(type(meta), field, None)
    if isinstance(declared, property) and declared.fset is not None:
        try:
            setattr(meta, field, value)
            return
        except Exception as error:
            logger.debug("meta.%s would not take its value (%s)", field, error)
    setter = getattr(meta, f"set_{field}", None)
    if callable(setter):
        try:
            setter(value)
            return
        except Exception as error:
            logger.debug("meta.set_%s would not take its value (%s)", field, error)
    logger.debug("this odfdo has nowhere to write %s, so it was left out", field)


def _datetime(stamp: str):
    """One document timestamp as a datetime, or ``None`` where it is not one."""
    from datetime import datetime

    from ..metadata import STAMP_FORMAT

    try:
        return datetime.strptime(stamp, STAMP_FORMAT)
    except (TypeError, ValueError):
        logger.debug("%r is not a timestamp, so it was left out of the exported file", stamp)
        return None


def _picture_name(picture: blocks.Picture, index: int) -> str:
    """A file name for one picture inside the package."""
    suffix = {
        "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
        "image/bmp": ".bmp", "image/tiff": ".tif", "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }.get(picture.media_type, "")
    if not suffix:
        tail = Path(picture.source.split("?")[0]).suffix.lower()
        suffix = tail if 1 < len(tail) <= 5 else ".img"
    return f"image{index + 1}{suffix}"


def _serialized(text) -> bytes:
    """The document's bytes, written in memory where odfdo will and to a file otherwise.

    Args:
        text: The odfdo document to save.

    Returns:
        The ``.odt`` file's bytes.

    Raises:
        OSError: The temporary file the fallback route needs could not be written.
    """
    buffer = BytesIO()
    try:
        text.save(target=buffer, packaging="zip")
        data = buffer.getvalue()
        if data:
            return data
        raise ValueError("odfdo wrote nothing to the buffer")
    except Exception as error:
        logger.debug("odfdo would not save in memory (%s), saving through a file", error)
    with tempfile.TemporaryDirectory(prefix="was-odt-") as directory:
        path = Path(directory) / "document.odt"
        text.save(target=str(path), packaging="zip")
        return path.read_bytes()
