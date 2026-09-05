"""A sensor reading written out as a DNG.

A DNG is a TIFF whose image is the reading, beside the tags a converter develops it with.
Readings are 16 bit, one sample per pixel or three.
"""

from __future__ import annotations

import struct
from pathlib import Path

__all__ = ["write"]

#: TIFF field types, by the code written into an entry.
BYTE = 1
ASCII = 2
SHORT = 3
LONG = 4
RATIONAL = 5
SRATIONAL = 10

#: Denominator every rational is written over.
SCALE = 1000000

#: The version this file declares, and the oldest reader expected to open it.
DNG_VERSION = (1, 4, 0, 0)
DNG_BACKWARD = (1, 1, 0, 0)

#: Photometric interpretation of a colour filter array, of a demosaiced raw image, and the
#: D65 illuminant code.
PHOTOMETRIC_CFA = 32803
PHOTOMETRIC_LINEAR = 34892
ILLUMINANT_D65 = 21

#: Colour of each plane a filter pattern indexes into, red then green then blue.
PLANE_COLOURS = (0, 1, 2)

#: Levels a 16-bit sensor reading spans.
FULL_SCALE = 65535


def _pack(kind: int, values) -> bytes:
    """One field's values as the bytes TIFF writes them in."""
    if kind == ASCII:
        return values.encode("ascii", "replace") + b"\0"
    if kind == BYTE:
        return bytes(values)
    if kind == SHORT:
        return b"".join(struct.pack("<H", int(v)) for v in values)
    if kind == LONG:
        return b"".join(struct.pack("<I", int(v)) for v in values)
    if kind == RATIONAL:
        return b"".join(struct.pack("<II", int(round(v * SCALE)), SCALE) for v in values)
    return b"".join(struct.pack("<ii", int(round(v * SCALE)), SCALE) for v in values)


def _count(kind: int, values) -> int:
    """How many values a field holds, counting a string's terminator."""
    return len(values) + 1 if kind == ASCII else len(values)


def write(path, plane, profile, camera: str = "", samples: int = 1) -> Path:
    """Write one sensor reading as a DNG.

    Args:
        path: Where to write. The parent directory must exist.
        plane: ``(height, width, body)``, the body holding 16-bit readings in row order,
            ``samples`` of them per pixel.
        profile: A :class:`~modules.image.raw.Profile`.
        camera: What to record as the camera. Defaults to the profile's name.
        samples: 1 for a colour filter array, 3 for a demosaiced raw image whose pixels each
            carry red, green and blue.

    Returns:
        The path written.

    Raises:
        ValueError: ``samples`` is neither 1 nor 3, or the buffer does not hold two bytes
            for every sample.
    """
    from .raw import CFA_PATTERNS

    if samples not in (1, 3):
        raise ValueError(f"a DNG holds 1 or 3 samples per pixel here, not {samples}")

    height, width, body = plane
    wanted = height * width * 2 * samples
    if len(body) != wanted:
        raise ValueError(
            f"a {width}x{height} plane at 16 bits and {samples} sample(s) needs {wanted} "
            f"bytes, not {len(body)}"
        )

    named = camera or profile.name
    neutral = (1.0 / max(profile.red_gain, 1e-6), 1.0, 1.0 / max(profile.blue_gain, 1e-6))
    matrix = [value for row in profile.xyz_to_camera for value in row]
    black = int(round(profile.black_level * FULL_SCALE))
    white = int(round(profile.white_level * FULL_SCALE))

    fields = [
        (254, LONG, [0]),
        (256, LONG, [width]),
        (257, LONG, [height]),
        (258, SHORT, [16] * samples),
        (259, SHORT, [1]),
        (262, SHORT, [PHOTOMETRIC_CFA if samples == 1 else PHOTOMETRIC_LINEAR]),
        (271, ASCII, "WAS"),
        (272, ASCII, named),
        (273, LONG, [0]),
        (274, SHORT, [1]),
        (277, SHORT, [samples]),
        (278, LONG, [height]),
        (279, LONG, [len(body)]),
        (282, RATIONAL, [72.0]),
        (283, RATIONAL, [72.0]),
        (284, SHORT, [1]),
        (296, SHORT, [2]),
        (305, ASCII, "WAS Node Suite"),
        (50706, BYTE, DNG_VERSION),
        (50707, BYTE, DNG_BACKWARD),
        (50708, ASCII, named),
        (50713, SHORT, [1, 1]),
        (50714, SHORT, [black] * samples),
        (50717, LONG, [white] * samples),
        (50721, SRATIONAL, matrix),
        (50728, RATIONAL, list(neutral)),
        (50778, SHORT, [ILLUMINANT_D65]),
    ]
    if samples == 1:
        fields += [
            (33421, SHORT, [2, 2]),
            (33422, BYTE, CFA_PATTERNS[profile.cfa.upper()]),
            (50710, BYTE, PLANE_COLOURS),
            (50711, SHORT, [1]),
        ]
    fields.sort(key=lambda field: field[0])

    # An entry holds its values inline under four bytes and an offset over them.
    header = 8
    directory = 2 + 12 * len(fields) + 4
    overflow_at = header + directory
    overflow, placed = bytearray(), {}
    for tag, kind, values in fields:
        blob = _pack(kind, values)
        if len(blob) > 4:
            placed[tag] = overflow_at + len(overflow)
            overflow += blob + (b"\0" if len(blob) % 2 else b"")
    body_at = overflow_at + len(overflow)

    entries = bytearray()
    for tag, kind, values in fields:
        blob = _pack(kind, values)
        count = _count(kind, values)
        if tag == 273:
            payload = struct.pack("<I", body_at)
        elif len(blob) > 4:
            payload = struct.pack("<I", placed[tag])
        else:
            payload = blob.ljust(4, b"\0")
        entries += struct.pack("<HHI", tag, kind, count) + payload

    out = Path(path)
    with out.open("wb") as handle:
        handle.write(struct.pack("<2sHI", b"II", 42, header))
        handle.write(struct.pack("<H", len(fields)))
        handle.write(bytes(entries))
        handle.write(struct.pack("<I", 0))
        handle.write(bytes(overflow))
        handle.write(body)
    return out


