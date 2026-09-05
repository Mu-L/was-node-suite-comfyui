"""Colour space profiles, built from their published primaries.

:func:`build` writes an ICC v2 matrix profile for a key of :data:`SPACES`. Primaries and
white points are CIE xy pairs and the matrix is adapted to the D50 the format stores.
"""

from __future__ import annotations

import struct

__all__ = ["SPACES", "build", "describe"]

#: The profile version and class this writes: v2.1, a display device, RGB into XYZ.
VERSION = 0x02100000
DEVICE_CLASS = b"mntr"
COLOUR_SPACE = b"RGB "
CONNECTION_SPACE = b"XYZ "
SIGNATURE = b"acsp"

#: Who the file says made it.
CREATOR = b"WAS "

#: The white the connection space is always written against.
D50 = (0.34567, 0.35850)

#: White points, by name.
D65 = (0.31270, 0.32900)

#: Every space offered: its red, green and blue primaries as CIE xy, its white point, and
#: the exponent of its transfer curve. A gamma of 1.0 is light itself.
SPACES = {
    "sRGB": ((0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600), D65, 2.2),
    "Adobe RGB (1998)": ((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600), D65, 2.19921875),
    "Display P3": ((0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600), D65, 2.2),
    "ProPhoto RGB": ((0.734699, 0.265301), (0.159597, 0.840403), (0.036598, 0.000105),
                     D50, 1.8),
    "Rec. 2020": ((0.7080, 0.2920), (0.1700, 0.7970), (0.1310, 0.0460), D65, 2.4),
    "linear sRGB": ((0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600), D65, 1.0),
}

#: The Bradford cone response, and its inverse, which every adaptation goes through.
BRADFORD = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)
BRADFORD_INVERSE = (
    (0.9869929, -0.1470543, 0.1599627),
    (0.4323053, 0.5183603, 0.0492912),
    (-0.0085287, 0.0400428, 0.9684867),
)

#: What one s15Fixed16 number is scaled by, and what a u8Fixed8 gamma is scaled by.
FIXED16 = 65536
FIXED8 = 256


def _white(xy) -> tuple:
    """One xy pair as an XYZ triple at a luminance of one."""
    x, y = xy
    return (x / y, 1.0, (1.0 - x - y) / y)


def _multiply(left, right) -> tuple:
    """One three by three matrix times another."""
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _apply(matrix, vector) -> tuple:
    """One three by three matrix times a vector."""
    return tuple(sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3))


def _inverse(matrix) -> tuple:
    """The inverse of a three by three matrix.

    Raises:
        ValueError: The matrix does not invert, which no published set of primaries gives.
    """
    (a, b, c), (d, e, f), (g, h, i) = matrix
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise ValueError("these primaries do not describe a colour space")
    return (
        ((e * i - f * h) / determinant, (c * h - b * i) / determinant,
         (b * f - c * e) / determinant),
        ((f * g - d * i) / determinant, (a * i - c * g) / determinant,
         (c * d - a * f) / determinant),
        ((d * h - e * g) / determinant, (b * g - a * h) / determinant,
         (a * e - b * d) / determinant),
    )


def _to_xyz(red, green, blue, white) -> tuple:
    """The matrix taking one space's RGB to CIE XYZ at its own white point."""
    columns = tuple(
        (x / y, 1.0, (1.0 - x - y) / y) for x, y in (red, green, blue)
    )
    shaped = tuple(tuple(columns[column][row] for column in range(3)) for row in range(3))
    scale = _apply(_inverse(shaped), _white(white))
    return tuple(
        tuple(shaped[row][column] * scale[column] for column in range(3)) for row in range(3)
    )


def _adapted(matrix, white) -> tuple:
    """One RGB to XYZ matrix moved from its own white point onto D50."""
    source = _apply(BRADFORD, _white(white))
    target = _apply(BRADFORD, _white(D50))
    ratio = tuple(
        tuple((target[row] / source[row]) if row == column else 0.0 for column in range(3))
        for row in range(3)
    )
    return _multiply(_multiply(BRADFORD_INVERSE, ratio), _multiply(BRADFORD, matrix))


def _fixed(value: float) -> bytes:
    """One number as the s15Fixed16 the format stores."""
    return struct.pack(">i", int(round(value * FIXED16)))


def _xyz_tag(column) -> bytes:
    """One colourant or white point, as an XYZType tag."""
    return b"XYZ " + bytes(4) + b"".join(_fixed(value) for value in column)


def _curve_tag(gamma: float) -> bytes:
    """One transfer curve, as the single gamma a curveType holds."""
    return b"curv" + bytes(4) + struct.pack(">I", 1) + struct.pack(
        ">H", int(round(gamma * FIXED8))
    )


def _text_tag(text: str) -> bytes:
    """One line of ASCII, as a textType tag."""
    return b"text" + bytes(4) + text.encode("ascii", "replace") + bytes(1)


def _description_tag(text: str) -> bytes:
    """One name, as the textDescriptionType a version 2 profile carries."""
    ascii_text = text.encode("ascii", "replace") + bytes(1)
    return (
        b"desc" + bytes(4)
        + struct.pack(">I", len(ascii_text)) + ascii_text
        + struct.pack(">II", 0, 0)
        + struct.pack(">HB", 0, 0) + bytes(67)
    )


def describe(space: str) -> str:
    """The name a built profile gives itself.

    Args:
        space: A key of :data:`SPACES`.

    Returns:
        The name, which is the key itself.
    """
    return space


def build(space: str) -> bytes:
    """Write one colour space as an ICC profile.

    Args:
        space: A key of :data:`SPACES`.

    Returns:
        The profile, as the bytes a file carries.

    Raises:
        ValueError: ``space`` names nothing known.
    """
    if space not in SPACES:
        raise ValueError(
            f"colour space must be one of {', '.join(SPACES)}, not {space!r}"
        )
    red, green, blue, white, gamma = SPACES[space]
    matrix = _adapted(_to_xyz(red, green, blue, white), white)
    columns = tuple(
        tuple(matrix[row][column] for row in range(3)) for column in range(3)
    )

    curve = _curve_tag(gamma)
    tags = [
        (b"desc", _description_tag(space)),
        (b"wtpt", _xyz_tag(_white(D50))),
        (b"rXYZ", _xyz_tag(columns[0])),
        (b"gXYZ", _xyz_tag(columns[1])),
        (b"bXYZ", _xyz_tag(columns[2])),
        (b"rTRC", curve),
        (b"gTRC", curve),
        (b"bTRC", curve),
        (b"cprt", _text_tag("Public domain colorimetry, no rights reserved.")),
    ]

    table = 4 + 12 * len(tags)
    at = 128 + table
    entries, body = bytearray(), bytearray()
    # The three curves are one tag written once and pointed at three times.
    placed = {}
    for signature, payload in tags:
        key = bytes(payload)
        if key not in placed:
            placed[key] = (at + len(body), len(payload))
            body += payload + bytes(-len(payload) % 4)
        offset, size = placed[key]
        entries += struct.pack(">4sII", signature, offset, size)

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + table + len(body))
    struct.pack_into(">4s", header, 4, CREATOR)
    struct.pack_into(">I", header, 8, VERSION)
    struct.pack_into(">4s", header, 12, DEVICE_CLASS)
    struct.pack_into(">4s", header, 16, COLOUR_SPACE)
    struct.pack_into(">4s", header, 20, CONNECTION_SPACE)
    struct.pack_into(">4s", header, 36, SIGNATURE)
    header[68:80] = b"".join(_fixed(value) for value in _white(D50))
    struct.pack_into(">4s", header, 80, CREATOR)

    return bytes(header) + struct.pack(">I", len(tags)) + bytes(entries) + bytes(body)
