"""Custom document properties added to a written ``.docx`` package.

:func:`with_custom_properties` takes the bytes ``python-docx`` produced and returns them
again with ``docProps/custom.xml`` added, its content type declared and its package
relationship written.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from typing import Mapping
from xml.sax.saxutils import escape

from ... import log

__all__ = [
    "CUSTOM_PART",
    "FIRST_PROPERTY_ID",
    "MAX_PROPERTIES",
    "with_custom_properties",
]

logger = log.get_logger("document.export.ooxml")

#: Where the properties live inside the package, and the two declarations that reach it.
CUSTOM_PART = "docProps/custom.xml"
CONTENT_TYPES_PART = "[Content_Types].xml"
RELATIONSHIPS_PART = "_rels/.rels"

#: Content type of the part, and the relationship type that points at it.
CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.custom-properties+xml"
)
RELATIONSHIP_TYPE = (
    "http://schemas.openxmlformats.org/package/2006/relationships/metadata/"
    "custom-properties"
)

#: Namespaces the part is written in.
_PROPERTIES_NS = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
_TYPES_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

#: The format identifier every custom property carries, which OOXML fixes.
FMT_ID = "{D5CDD505-2E9C-101B-9397-08002B2CF9AE}"

#: Property ids start here: 0 and 1 are reserved by the format.
FIRST_PROPERTY_ID = 2

#: Most properties written. A document with more than this is carrying a database rather
#: than a description, and every one of them is read by hand in a properties dialog.
MAX_PROPERTIES = 128

#: Longest value written for one property. Word truncates a long one anyway, and a value
#: this long is not a property.
MAX_VALUE_CHARS = 4096

#: Characters XML 1.0 cannot carry at all, whatever they are escaped as.
_FORBIDDEN = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")

_RELATIONSHIP_ID = re.compile(r'Id="rId(\d+)"')


def with_custom_properties(package: bytes, properties: Mapping[str, str]) -> bytes:
    """Add custom document properties to a ``.docx`` package.

    Args:
        package: The bytes of a written package.
        properties: ``{name: value}``. An empty name or value is skipped, at most
            :data:`MAX_PROPERTIES` are written and each value is cut to
            :data:`MAX_VALUE_CHARS` characters.

    Returns:
        The package with the part added, or the bytes exactly as they arrived when there is
        nothing to write, when the package already carries the part, or when it cannot be
        read or rewritten.
    """
    pairs = [
        (str(name).strip(), _value(value))
        for name, value in (properties or {}).items()
        if str(name).strip() and _value(value)
    ][:MAX_PROPERTIES]
    if not pairs:
        return package
    try:
        return _rewritten(package, pairs)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        logger.warning(
            "the exported .docx was written without its %d custom propert(ies) (%s). The "
            "file itself is complete; the copyright statement and any custom pairs are in "
            "the document's own metadata, which View DOC Metadata reports.",
            len(pairs), error,
        )
        return package


def _rewritten(package: bytes, pairs: list[tuple[str, str]]) -> bytes:
    """Rebuild the package with the custom part in it.

    Args:
        package: The bytes of a written package.
        pairs: The properties to write, already checked.

    Returns:
        The rebuilt package.

    Raises:
        BadZipFile: The bytes are not a readable zip.
        KeyError: The package carries no ``[Content_Types].xml`` or no ``_rels/.rels``.
        ValueError: One of those two parts cannot be rewritten.
    """
    with zipfile.ZipFile(BytesIO(package)) as archive:
        names = archive.namelist()
        if CUSTOM_PART in names:
            return package
        for required in (CONTENT_TYPES_PART, RELATIONSHIPS_PART):
            if required not in names:
                raise KeyError(f"the package carries no {required}")
        content_types = _with_override(archive.read(CONTENT_TYPES_PART).decode("utf-8"))
        relationships = _with_relationship(archive.read(RELATIONSHIPS_PART).decode("utf-8"))
        entries = [
            (info, archive.read(info.filename))
            for info in archive.infolist()
            if not info.is_dir()
        ]

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out:
        for info, payload in entries:
            if info.filename == CONTENT_TYPES_PART:
                payload = content_types.encode("utf-8")
            elif info.filename == RELATIONSHIPS_PART:
                payload = relationships.encode("utf-8")
            copy = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            copy.compress_type = info.compress_type
            copy.external_attr = info.external_attr
            out.writestr(copy, payload)
        out.writestr(CUSTOM_PART, _part(pairs).encode("utf-8"))
    return buffer.getvalue()


def _part(pairs: list[tuple[str, str]]) -> str:
    """The XML of ``docProps/custom.xml`` holding every property."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Properties xmlns="{_PROPERTIES_NS}" xmlns:vt="{_TYPES_NS}">',
    ]
    for index, (name, value) in enumerate(pairs, start=FIRST_PROPERTY_ID):
        lines.append(
            f'<property fmtid="{FMT_ID}" pid="{index}" name="{_attribute(name)}">'
            f"<vt:lpwstr>{escape(value)}</vt:lpwstr></property>"
        )
    lines.append("</Properties>")
    return "".join(lines)


def _with_override(content_types: str) -> str:
    """``[Content_Types].xml`` with the custom part's content type declared.

    Args:
        content_types: The part as the package holds it.

    Returns:
        The part with one ``Override`` added before its closing tag.

    Raises:
        ValueError: The part has no ``</Types>`` to insert before.
    """
    closing = "</Types>"
    at = content_types.rfind(closing)
    if at < 0:
        raise ValueError(f"{CONTENT_TYPES_PART} has no {closing}")
    override = f'<Override PartName="/{CUSTOM_PART}" ContentType="{CONTENT_TYPE}"/>'
    return content_types[:at] + override + content_types[at:]


def _with_relationship(relationships: str) -> str:
    """``_rels/.rels`` with a relationship pointing at the custom part.

    Args:
        relationships: The part as the package holds it.

    Returns:
        The part with one ``Relationship`` added, its id one past the highest already
        written, so it cannot collide with a relationship ``python-docx`` wrote.

    Raises:
        ValueError: The part has no ``</Relationships>`` to insert before.
    """
    closing = "</Relationships>"
    at = relationships.rfind(closing)
    if at < 0:
        raise ValueError(f"{RELATIONSHIPS_PART} has no {closing}")
    used = [int(number) for number in _RELATIONSHIP_ID.findall(relationships)]
    identifier = f"rId{max(used, default=0) + 1}"
    # A target in _rels/.rels is relative to the package root, which is where every
    # relationship python-docx writes there points from.
    added = (
        f'<Relationship Id="{identifier}" Type="{RELATIONSHIP_TYPE}" '
        f'Target="{CUSTOM_PART}"/>'
    )
    return relationships[:at] + added + relationships[at:]


def _attribute(value: str) -> str:
    """One attribute value with every character XML gives a meaning to escaped."""
    return escape(value).replace('"', "&quot;")


def _value(value) -> str:
    """One property value as text, bounded and stripped of what XML cannot carry."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return _FORBIDDEN.sub("", text).strip()[:MAX_VALUE_CHARS]
