"""Writing a document out in a format another program reads.

:data:`FORMATS` is the three formats, each spelled as the extension its file gets, and
:func:`export` returns the bytes of one for a :class:`..container.Document`. Each needs the
package :data:`PACKAGES` names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ... import log

if TYPE_CHECKING:  # pragma: no cover - imported for the annotation alone
    from ..container import Document

__all__ = [
    "DEFAULT_MARGIN_MM",
    "DOCX",
    "FEATURE",
    "FORMATS",
    "MM_PER_POINT",
    "ODT",
    "ORIENTATIONS",
    "PACKAGES",
    "PAGE_SIZES",
    "PDF",
    "Page",
    "export",
    "normalized",
    "writes",
]

logger = log.get_logger("document.export")

#: Config key of the feature group these formats are gated on.
FEATURE = "features.document_export"

#: The three formats, each spelled as the extension its file gets. An entry is permanent once
#: it ships: it is a combo option, and a saved workflow holds the option it was set to.
DOCX = ".docx"
ODT = ".odt"
PDF = ".pdf"
FORMATS = (DOCX, ODT, PDF)

#: Import name of the package each format needs. The pip name of ``docx`` is ``python-docx``,
#: which ``modules.deps`` knows, so an error names the command that works.
PACKAGES = {DOCX: "docx", ODT: "odfdo", PDF: "xhtml2pdf"}

#: Page sizes offered, in points, portrait. A point is 1/72 inch, which is the unit ODF,
#: OOXML and PDF all measure a page in.
PAGE_SIZES = {
    "A4": (595.28, 841.89),
    "Letter": (612.0, 792.0),
    "Legal": (612.0, 1008.0),
    "A5": (419.53, 595.28),
    "A3": (841.89, 1190.55),
    "Tabloid": (792.0, 1224.0),
}

#: The two page orientations, default first.
ORIENTATIONS = ("portrait", "landscape")

#: Millimetres in one point, for reading a margin given in millimetres.
MM_PER_POINT = 25.4 / 72.0

#: Margin used when none is given, in millimetres. Near enough to what a word processor's own
#: template leaves that a page does not read as unusually cramped or wide.
DEFAULT_MARGIN_MM = 20.0


@dataclass(frozen=True)
class Page:
    """The page a document is laid out on, in points.

    Attributes:
        width: Page width.
        height: Page height.
        margin: Space left on all four sides.
    """

    width: float = PAGE_SIZES["A4"][0]
    height: float = PAGE_SIZES["A4"][1]
    margin: float = DEFAULT_MARGIN_MM / MM_PER_POINT

    @classmethod
    def build(
        cls,
        size: str = "A4",
        orientation: str = ORIENTATIONS[0],
        margin_mm: float = DEFAULT_MARGIN_MM,
    ) -> "Page":
        """The page named by a size, an orientation and a margin.

        Args:
            size: A key of :data:`PAGE_SIZES`. An unknown name is read as ``"A4"``.
            orientation: One of :data:`ORIENTATIONS`. Landscape swaps width and height.
            margin_mm: Margin in millimetres, clamped so the page keeps a text column at
                least a tenth of its width.

        Returns:
            The page, in points.
        """
        width, height = PAGE_SIZES.get(str(size), PAGE_SIZES["A4"])
        if str(orientation).strip().lower() == "landscape":
            width, height = height, width
        limit = min(width, height) * 0.45
        try:
            margin = float(margin_mm) / MM_PER_POINT
        except (TypeError, ValueError):
            margin = DEFAULT_MARGIN_MM / MM_PER_POINT
        return cls(width=width, height=height, margin=max(0.0, min(margin, limit)))

    @property
    def text_width(self) -> float:
        """Points across the page that text may occupy."""
        return max(1.0, self.width - 2 * self.margin)


def writes(extension: Any) -> bool:
    """Whether one format is written here rather than by :mod:`..formats`.

    Args:
        extension: A widget value, with or without its leading dot and in any case.

    Returns:
        True for one of :data:`FORMATS`.
    """
    return _spelled(extension) in FORMATS


def normalized(extension: Any) -> str:
    """One of :data:`FORMATS`, read from a widget value.

    Args:
        extension: The widget value. A leading dot is added where it is missing and the case
            is ignored, so ``DOCX`` and ``.docx`` both name the same format.

    Returns:
        The format, spelled as :data:`FORMATS` spells it.

    Raises:
        ValueError: The value names none of these three formats.
    """
    text = _spelled(extension)
    if text not in FORMATS:
        raise ValueError(
            f"{extension!r} is not one of the formats written through a document library.\n"
            f"  Those formats are: {', '.join(FORMATS)}."
        )
    return text


def export(document: "Document", extension: Any, page: Page | None = None) -> bytes:
    """Write a document out in one of the rich formats.

    Args:
        document: The document to convert. Its ``content.html`` is the source, its metadata
            is written into the file's own fields, and its embedded files are the pictures
            available to it.
        extension: One of :data:`FORMATS`.
        page: The page to lay it out on, or ``None`` for A4 portrait at a 20 mm margin.

    Returns:
        The file's bytes, ready to write.

    Raises:
        ValueError: The format is none of these three, or the library could not lay the
            document out.
        DependencyError: The package that format needs is missing or unusable.
    """
    chosen = normalized(extension)
    page = page or Page()
    if chosen == DOCX:
        from . import docx_writer

        return docx_writer.write(document, page)
    if chosen == ODT:
        from . import odt_writer

        return odt_writer.write(document, page)
    from . import pdf_writer

    return pdf_writer.write(document, page)


def _spelled(extension: Any) -> str:
    """One widget value as a format is spelled here: lower case, leading dot."""
    text = str(extension or "").strip().lower()
    if text and not text.startswith("."):
        text = "." + text
    return text
