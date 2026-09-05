"""Sample depths a saved image can carry, and which formats hold each one.

``OPTIONS`` is the widget's option list and ``FORMATS`` maps an extension to the depths it
holds.
"""

from __future__ import annotations

__all__ = ["BITS", "DEFAULT", "EXR_DEPTHS", "FORMATS", "OPTIONS", "refusal"]

#: Depths a file can be written at, shallowest first.
OPTIONS = ("8-bit", "16-bit", "32-bit float")

#: The depth a fresh node writes at.
DEFAULT = OPTIONS[0]

#: Bits one channel occupies at each depth.
BITS = {"8-bit": 8, "16-bit": 16, "32-bit float": 32}

#: Extension -> the depths that format holds.
FORMATS = {
    "png": frozenset({"8-bit", "16-bit"}),
    "jpg": frozenset({"8-bit"}),
    "jpeg": frozenset({"8-bit"}),
    "gif": frozenset({"8-bit"}),
    "tiff": frozenset({"8-bit"}),
    "webp": frozenset({"8-bit"}),
    "bmp": frozenset({"8-bit"}),
    "exr": frozenset({"16-bit", "32-bit float"}),
}

#: Depth -> the sample type key ``modules.image.exr.write`` takes.
EXR_DEPTHS = {"16-bit": "16 bit half", "32-bit float": "32 bit float"}


def _listed(names) -> str:
    """A run of names as ``a``, ``a or b``, or ``a, b or c``."""
    values = list(names)
    if len(values) < 2:
        return "".join(values)
    return f"{', '.join(values[:-1])} or {values[-1]}"


def refusal(extension: str, depth: str) -> str:
    """Why a format cannot hold a depth, as a sentence, or ``""`` when it can.

    Args:
        extension: File extension without its dot.
        depth: A member of :data:`OPTIONS`.

    Returns:
        A message naming the depths that format takes and the formats that take this depth,
        or an empty string when the pair is writable.
    """
    held = FORMATS.get(extension)
    if held is None:
        return f"{extension} is not a format this writes"
    if depth in held:
        return ""
    article = "an" if extension[:1] in "aeiou" else "a"
    takers = _listed(sorted(name for name, holds in FORMATS.items() if depth in holds))
    return (
        f"{article} {extension} file cannot hold {depth} samples. Set bit_depth to "
        f"{_listed(option for option in OPTIONS if option in held)}, or set extension to "
        f"{takers}"
    )
