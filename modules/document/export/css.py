"""The parts of CSS an export reads out of a document's markup.

Only an element's ``style`` attribute and the presentational attributes ``align``, ``color``,
``face``, ``size``, ``bgcolor``, ``width`` and ``height`` are read. Colours come back as
``"RRGGBB"``, lengths in points.
"""

from __future__ import annotations

import re

__all__ = [
    "ALIGNMENTS",
    "BASE_FONT_POINTS",
    "MAX_DECLARATIONS",
    "MAX_FONT_POINTS",
    "NAMED_COLORS",
    "alignment",
    "color",
    "declarations",
    "font_family",
    "length",
]

#: Point size a relative length such as ``1.5em`` or ``120%`` is taken from, and the size a
#: document's body text is written at when it names none.
BASE_FONT_POINTS = 11.0

#: Largest font size honoured. A stylesheet claiming a size in the thousands would give a
#: word processor a page it cannot lay out, and no real document asks for one.
MAX_FONT_POINTS = 1584.0

#: Most declarations read out of one ``style`` attribute. An attribute holding more than
#: this is a generated blob rather than an author's formatting.
MAX_DECLARATIONS = 64

#: The alignments a paragraph may carry, as CSS spells them.
ALIGNMENTS = ("left", "center", "right", "justify")

#: Colour names a document may use, as ``RRGGBB``. The sixteen CSS 2.1 names, which are the
#: ones an editor writes, and the handful of further names a hand-written document reaches
#: for. A name absent from here is left alone rather than guessed at.
NAMED_COLORS = {
    "aqua": "00FFFF", "black": "000000", "blue": "0000FF", "brown": "A52A2A",
    "cyan": "00FFFF", "darkblue": "00008B", "darkgray": "A9A9A9", "darkgreen": "006400",
    "darkgrey": "A9A9A9", "darkred": "8B0000", "fuchsia": "FF00FF", "gold": "FFD700",
    "gray": "808080", "green": "008000", "grey": "808080", "indigo": "4B0082",
    "lightblue": "ADD8E6", "lightgray": "D3D3D3", "lightgreen": "90EE90",
    "lightgrey": "D3D3D3", "lime": "00FF00", "magenta": "FF00FF", "maroon": "800000",
    "navy": "000080", "olive": "808000", "orange": "FFA500", "pink": "FFC0CB",
    "purple": "800080", "red": "FF0000", "silver": "C0C0C0", "teal": "008080",
    "violet": "EE82EE", "white": "FFFFFF", "yellow": "FFFF00",
}

#: Point size of each CSS font-size keyword, at a 16 pixel base.
_KEYWORD_POINTS = {
    "xx-small": 6.75, "x-small": 7.5, "small": 9.75, "medium": 12.0,
    "large": 13.5, "x-large": 18.0, "xx-large": 24.0,
}

#: How many points one unit of each absolute length is worth.
_UNIT_POINTS = {
    "pt": 1.0, "px": 0.75, "pc": 12.0, "in": 72.0, "cm": 28.3465, "mm": 2.83465,
    "q": 0.708661, "ex": 5.5, "ch": 5.5,
}

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)")

_RGB = re.compile(
    r"^rgba?\(\s*([\d.]+%?)\s*[,\s]\s*([\d.]+%?)\s*[,\s]\s*([\d.]+%?)\s*(?:[,/].*)?\)$"
)


def declarations(style: str) -> dict[str, str]:
    """Read a ``style`` attribute into property names and values.

    Args:
        style: The attribute value, such as ``"font-weight: bold; color: #c00"``. An empty
            or unreadable value gives an empty mapping rather than raising.

    Returns:
        ``{property: value}``, both stripped and folded to lower case for the name, with
        the value's own case kept. A property written twice keeps the last value, as CSS
        does. At most :data:`MAX_DECLARATIONS` are read.
    """
    if not style or not isinstance(style, str):
        return {}
    found: dict[str, str] = {}
    for part in style.split(";")[: MAX_DECLARATIONS]:
        name, separator, value = part.partition(":")
        if not separator:
            continue
        name = name.strip().lower()
        value = value.strip()
        if name and value:
            found[name] = value
    return found


def color(value: str) -> str | None:
    """Read a colour as ``RRGGBB``.

    Args:
        value: ``"#c00"``, ``"#cc0000"``, ``"rgb(204, 0, 0)"``, ``"rgba(204,0,0,.5)"`` or a
            name from :data:`NAMED_COLORS`. Case does not matter.

    Returns:
        Six upper-case hexadecimal digits, or ``None`` where the value is not a colour this
        reads. An ``rgba`` colour keeps its channels and loses its transparency.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in (3, 4) and _is_hex(digits):
            return "".join(char * 2 for char in digits[:3]).upper()
        if len(digits) in (6, 8) and _is_hex(digits):
            return digits[:6].upper()
        return None
    match = _RGB.match(text)
    if match:
        channels = [_channel(part) for part in match.groups()]
        if None in channels:
            return None
        return "".join(f"{part:02X}" for part in channels)
    return NAMED_COLORS.get(text)


def length(value: str, base: float = BASE_FONT_POINTS) -> float | None:
    """Read a CSS length as points.

    Args:
        value: ``"12pt"``, ``"16px"``, ``"1.5em"``, ``"120%"``, ``"2cm"`` or a font-size
            keyword such as ``"large"``. A bare number is read as pixels, which is how HTML
            reads a ``width`` attribute.
        base: Points a relative length is taken from.

    Returns:
        The length in points, clamped to :data:`MAX_FONT_POINTS` and never negative, or
        ``None`` where the value is not a length. ``vw``, ``vh`` and ``calc()`` are not
        lengths here: they need a viewport or an evaluator, and a page has neither.
    """
    if value is None or not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text in _KEYWORD_POINTS:
        return _KEYWORD_POINTS[text]
    match = _NUMBER.match(text)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    unit = text[match.end():].strip()
    if unit in ("em", "rem"):
        points = number * base
    elif unit == "%":
        points = number * base / 100.0
    elif unit in ("", "px"):
        points = number * _UNIT_POINTS["px"]
    elif unit in _UNIT_POINTS:
        points = number * _UNIT_POINTS[unit]
    else:
        return None
    return max(0.0, min(points, MAX_FONT_POINTS))


def font_family(value: str) -> str | None:
    """The first family named in a ``font-family`` value.

    Args:
        value: ``"Georgia, 'Times New Roman', serif"`` or one bare name.

    Returns:
        The first name, with its quotes taken off, or ``None`` where there is none. Only
        the first is kept: a word processor names one font per run and has no fallback
        list to put the rest in.
    """
    if not value or not isinstance(value, str):
        return None
    first = value.split(",")[0].strip().strip("\"'").strip()
    return first or None


def alignment(value: str) -> str | None:
    """Read a ``text-align`` value.

    Args:
        value: ``"left"``, ``"center"``, ``"right"``, ``"justify"``, or one of the writing
            direction spellings ``"start"`` and ``"end"``.

    Returns:
        One of :data:`ALIGNMENTS`, or ``None``. ``start`` and ``end`` are read as left and
        right, which is what they mean in a left-to-right document, and the document's own
        direction is not consulted.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in ALIGNMENTS:
        return text
    if text == "centre":
        return "center"
    return {"start": "left", "end": "right"}.get(text)


def _is_hex(digits: str) -> bool:
    """Whether every character is a hexadecimal digit."""
    return all(char in "0123456789abcdef" for char in digits)


def _channel(part: str) -> int | None:
    """One ``rgb()`` channel as 0 to 255, or ``None`` where it is not a number."""
    try:
        number = float(part[:-1]) * 255.0 / 100.0 if part.endswith("%") else float(part)
    except ValueError:
        return None
    return max(0, min(255, int(round(number))))
