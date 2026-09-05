"""The DOC container: a zip holding ``content.html``, ``meta.json`` and ``assets/``.

:class:`Document` is what a DOC socket carries: the container bytes, the HTML, the metadata
and the assets. It is immutable, and an edit answers a new one.
"""

from __future__ import annotations

import io
import json
import zipfile
from types import MappingProxyType
from typing import Any, Mapping

from .. import log
from . import text
from .metadata import Metadata, from_dict, touched, with_stamps

__all__ = [
    "ASSET_PREFIX",
    "CHARACTER_COUNT_KEY",
    "CONTAINER_VERSION",
    "CONTENT_ENTRY",
    "Document",
    "DocumentError",
    "DocumentIsImmutable",
    "ENCODING",
    "MAX_ASSETS",
    "MAX_UNPACKED_BYTES",
    "METADATA_ENTRY",
    "MIMETYPE",
    "MIMETYPE_ENTRY",
    "NotADocument",
    "READ_ENCODING",
    "SUFFIX",
    "UNVERSIONED",
    "UnsupportedVersion",
    "VERSION_KEY",
    "WORD_COUNT_KEY",
    "is_document",
    "require_document",
]

logger = log.get_logger("document.container")

#: The layout this build writes, and the highest it reads. Raised when the layout changes in
#: a way a reader has to know about, never for a change a version 1 reader handles correctly.
CONTAINER_VERSION = 1

#: What a container naming no version is read as. Version 1 is the first layout, and
#: ``content.html`` plus ``assets/`` describes it completely, so a zip assembled by hand or
#: by another tool is read rather than refused.
UNVERSIONED = 1

#: Media type of the container, written into the ``mimetype`` entry. Following the ODF
#: convention, that entry is written first and uncompressed, so the type is readable in the
#: first bytes of the file by a tool that knows nothing else about it.
MIMETYPE = "application/vnd.was-node-suite.document+zip"

#: File extension a saved container carries.
SUFFIX = ".wasdoc"

#: The three fixed entry names, and the directory every embedded file sits under.
MIMETYPE_ENTRY = "mimetype"
CONTENT_ENTRY = "content.html"
METADATA_ENTRY = "meta.json"
ASSET_PREFIX = "assets/"

#: Keys ``meta.json`` carries that are the container's rather than the author's.
VERSION_KEY = "container_version"
WORD_COUNT_KEY = "word_count"
CHARACTER_COUNT_KEY = "character_count"

#: The codec both text entries are written with. Reading uses the ``-sig`` form, which
#: drops a byte order mark a foreign writer may have left at the front of the file, where it
#: would otherwise be an invisible character at the start of the document.
ENCODING = "utf-8"
READ_ENCODING = "utf-8-sig"

#: Timestamp written on every entry: the earliest a zip can hold. A real clock would make
#: the bytes differ between two writes of identical content, which would cost every reader
#: the ability to tell whether a document has actually changed.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

#: File mode written on every entry, a regular file readable by anybody, so a container
#: written on Windows and one written on Linux from the same document are the same bytes.
ENTRY_MODE = 0o100644 << 16

#: How much a container may unpack to. Far above any real document, and low enough that a
#: zip crafted to unpack to hundreds of gigabytes is refused rather than taking the process
#: down with it.
MAX_UNPACKED_BYTES = 256 * 1024 * 1024

#: How many embedded files a container may hold. A document with more than this is not a
#: document, and an entry costs memory whatever its size.
MAX_ASSETS = 1024

#: Entry names listed in the message when a zip turns out not to be a document.
_NAMES_SHOWN = 12


class DocumentError(ValueError):
    """A document could not be read or built."""


class DocumentIsImmutable(AttributeError):
    """Something tried to assign to a document, or to delete part of one."""


class NotADocument(DocumentError):
    """The value or the bytes offered are not a document container."""


class UnsupportedVersion(DocumentError):
    """The container names a layout version this build does not read."""


class Document:
    """One document: its HTML, its metadata and the files embedded in it.

    Attributes:
        content: The document markup, as ``content.html`` holds it.
        metadata: The authored metadata, as a :class:`~modules.document.metadata.Metadata`.
        assets: Read-only view of ``{name: bytes}``, one entry per file under ``assets/``.
            A name is relative to that directory and always spelled with ``/``.
        data: The container bytes. Read from the file when the document came from one, and
            built on first use otherwise, then kept.
        plain_text: The text of ``content`` that the two counts are taken over.
        word_count: Words in ``plain_text``.
        character_count: Characters in ``plain_text``.
    """

    __slots__ = ("_assets", "_content", "_counts", "_data", "_metadata", "_text")

    def __init__(
        self,
        content: str,
        metadata: Metadata | None = None,
        assets: Mapping[str, bytes] | None = None,
        data: bytes | None = None,
    ) -> None:
        """Hold one document.

        Args:
            content: The document markup.
            metadata: The authored metadata, empty when omitted.
            assets: ``{name: bytes}`` of embedded files, named relative to ``assets/``.
            data: The container bytes these parts were read out of, when they were. Held,
                so loading a document and saving it again writes the same file back.

        Raises:
            DocumentError: ``content`` is not a string, or an asset name or its bytes
                cannot go in a container.
        """
        if not isinstance(content, str):
            raise DocumentError(
                f"a document's content is HTML text and this one is {_named(content)}."
            )
        # Every slot is written through object.__setattr__, since __setattr__ below refuses
        # an assignment from anywhere. The embedded files go into a read-only view whose
        # dictionary nothing else holds, so reaching the slot reaches no writable mapping.
        object.__setattr__(self, "_content", content)
        object.__setattr__(
            self, "_metadata", metadata if isinstance(metadata, Metadata) else Metadata()
        )
        object.__setattr__(self, "_assets", MappingProxyType(_checked_assets(assets)))
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_text", None)
        object.__setattr__(self, "_counts", None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to any attribute.

        Args:
            name: The attribute that was assigned to.
            value: What it was assigned.

        Raises:
            DocumentIsImmutable: Always.
        """
        raise DocumentIsImmutable(_no_edits("set", name))

    def __delattr__(self, name: str) -> None:
        """Refuse a deletion of any attribute.

        Args:
            name: The attribute that was deleted.

        Raises:
            DocumentIsImmutable: Always.
        """
        raise DocumentIsImmutable(_no_edits("delete", name))

    def __reduce__(self) -> tuple:
        """Rebuild through the constructor, so a copy assigns no attribute.

        Returns:
            The class and the four arguments that remake this document. Copying and
            pickling both come here.
        """
        return (
            self.__class__,
            (self._content, self._metadata, dict(self._assets), self._data),
        )

    # ------------------------------------------------------------------ building

    @classmethod
    def build(
        cls,
        content: str,
        metadata: Metadata | None = None,
        assets: Mapping[str, bytes] | None = None,
    ) -> "Document":
        """Make a new document, stamping the timestamps it has none of.

        Args:
            content: The document markup.
            metadata: The authored metadata. ``created``, ``modified`` and ``generator``
                are filled in where they are empty, and left alone where they are not.
            assets: ``{name: bytes}`` of embedded files, named relative to ``assets/``.

        Returns:
            The document. Its container bytes are built the first time anything asks for
            them.

        Raises:
            DocumentError: ``content`` is not a string, or an asset cannot go in a
                container.
        """
        return cls(content, with_stamps(metadata or Metadata()), assets)

    @classmethod
    def from_bytes(cls, data: Any) -> "Document":
        """Read a container.

        Args:
            data: The container bytes, as read from a file or an archive.

        Returns:
            The document. The bytes are kept as they arrived, so writing it back out
            produces the same file.

        Raises:
            NotADocument: The bytes are not a zip, hold no ``content.html``, or hold a
                ``meta.json`` that is not readable JSON.
            UnsupportedVersion: The container names a version above
                :data:`CONTAINER_VERSION`.
            DocumentError: The container unpacks to more than :data:`MAX_UNPACKED_BYTES`
                or holds more than :data:`MAX_ASSETS` embedded files.
        """
        payload = _as_bytes(data)
        with _opened(payload) as archive:
            entries = _entries(archive)
            if CONTENT_ENTRY not in entries:
                raise NotADocument(_missing_content(entries))

            budget = _unpacked_budget(archive, entries)
            raw, budget = _member(archive, entries[CONTENT_ENTRY], budget)
            content = _decoded(raw, CONTENT_ENTRY)

            metadata = Metadata()
            if METADATA_ENTRY in entries:
                raw, budget = _member(archive, entries[METADATA_ENTRY], budget)
                metadata = _read_metadata(_decoded(raw, METADATA_ENTRY))

            assets, budget = _assets(archive, entries, budget)
        return cls(content, metadata, assets, payload)

    def with_content(self, content: str) -> "Document":
        """The same document with different markup, and ``modified`` stamped now.

        Args:
            content: The new markup.

        Returns:
            A new document carrying the new content, the same assets, and metadata whose
            ``modified`` is the current time.

        Raises:
            DocumentError: ``content`` is not a string.
        """
        return Document(content, touched(self._metadata), self._assets)

    def with_metadata(self, metadata: Metadata) -> "Document":
        """The same document with different metadata.

        Args:
            metadata: The new record.

        Returns:
            A new document with the same content and assets.
        """
        return Document(self._content, metadata, self._assets)

    def with_assets(self, assets: Mapping[str, bytes] | None) -> "Document":
        """The same document with a different set of embedded files.

        Args:
            assets: ``{name: bytes}`` replacing every current entry. ``None`` empties it.

        Returns:
            A new document with the same content and metadata whose ``modified`` is the
            current time.

        Raises:
            DocumentError: An asset name or its bytes cannot go in a container.
        """
        return Document(self._content, touched(self._metadata), assets)

    # ------------------------------------------------------------------ reading

    @property
    def content(self) -> str:
        """The document markup."""
        return self._content

    @property
    def metadata(self) -> Metadata:
        """The authored metadata."""
        return self._metadata

    @property
    def assets(self) -> Mapping[str, bytes]:
        """Read-only view of the embedded files, keyed on their name under ``assets/``."""
        return self._assets

    @property
    def data(self) -> bytes:
        """The container bytes, built on first use when the document was not read from any."""
        if self._data is None:
            object.__setattr__(self, "_data", self._serialize())
        return self._data

    @property
    def plain_text(self) -> str:
        """The text the two counts are taken over, one line break to a block boundary."""
        if self._text is None:
            object.__setattr__(self, "_text", text.plain_text(self._content))
        return self._text

    @property
    def word_count(self) -> int:
        """Words in :attr:`plain_text`, by the rule in :mod:`modules.document.text`."""
        return self._derived()[0]

    @property
    def character_count(self) -> int:
        """Characters in :attr:`plain_text`, by the rule in :mod:`modules.document.text`."""
        return self._derived()[1]

    def metadata_json(self) -> dict[str, Any]:
        """The whole of ``meta.json`` as it is written, counts and version included.

        Returns:
            A new dictionary: the container version, then the authored fields in the order
            the file lists them, then the two derived counts.
        """
        return {
            VERSION_KEY: CONTAINER_VERSION,
            **self._metadata.to_dict(),
            WORD_COUNT_KEY: self.word_count,
            CHARACTER_COUNT_KEY: self.character_count,
        }

    def __repr__(self) -> str:
        return (
            f"<Document {self._metadata.title or 'untitled'!r} "
            f"{self.word_count} word(s), {len(self._assets)} asset(s)>"
        )

    def _derived(self) -> tuple[int, int]:
        """The word and character counts, computed once and kept."""
        if self._counts is None:
            object.__setattr__(
                self,
                "_counts",
                (text.word_count(self.plain_text), text.character_count(self.plain_text)),
            )
        return self._counts

    # ------------------------------------------------------------------ writing

    def _serialize(self) -> bytes:
        """Build the container bytes from the parts in hand."""
        payload = json.dumps(self.metadata_json(), ensure_ascii=False, indent=2) + "\n"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            _put(archive, MIMETYPE_ENTRY, MIMETYPE.encode("ascii"), zipfile.ZIP_STORED)
            _put(archive, METADATA_ENTRY, payload.encode(ENCODING), zipfile.ZIP_DEFLATED)
            _put(archive, CONTENT_ENTRY, self._content.encode(ENCODING), zipfile.ZIP_DEFLATED)
            for name in sorted(self._assets):
                _put(archive, ASSET_PREFIX + name, self._assets[name], zipfile.ZIP_DEFLATED)
        return buffer.getvalue()


def is_document(value: Any) -> bool:
    """Whether a value is a document, for a node choosing between two branches."""
    return isinstance(value, Document)


def require_document(value: Any, label: str = "doc") -> Document:
    """Read a DOC socket, or raise naming what arrived instead.

    Args:
        value: Whatever arrived on the socket.
        label: The input's name, used in the message.

    Returns:
        The document.

    Raises:
        NotADocument: ``value`` is not a document.
    """
    if isinstance(value, Document):
        return value
    raise NotADocument(
        f"the {label} input needs a document and was given {_described(value)}.\n"
        f"  A DOC carries a whole document: its HTML, its metadata and any file embedded "
        f"in it.\n"
        f"  Connect a node that produces a document to this input."
    )


# ---------------------------------------------------------------------- helpers


def _no_edits(verb: str, name: str) -> str:
    """The message an attempted edit to a document is refused with.

    Args:
        verb: What was attempted, ``"set"`` or ``"delete"``.
        name: The attribute it was attempted on.

    Returns:
        The message, naming the attribute and the three methods that build a new document.
    """
    return (
        f"a document cannot be changed once it has been made, and something tried to "
        f"{verb} {name!r} on this one.\n"
        f"  A DOC on a wire is read by every node below it, so an edit in place would "
        f"change a document those nodes are still holding, and any container bytes, text "
        f"and counts already taken from it would no longer describe it.\n"
        f"  Build the document wanted with with_content, with_metadata or with_assets. "
        f"Each answers a new document and leaves this one alone."
    )


def _named(value: Any) -> str:
    """The type of a value with the article that belongs in front of it, ``"an int"``."""
    name = type(value).__name__
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


def _described(value: Any) -> str:
    """Name what arrived on a socket, for the message that refuses it."""
    if value is None:
        return "nothing at all"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return (
            f"{len(bytes(value))} raw byte(s); a DOC carries the document itself rather "
            f"than the bytes of its container"
        )
    if isinstance(value, str):
        return "a string; text becomes a document by going through a document node"
    return _named(value)


def _as_bytes(data: Any) -> bytes:
    """A container's bytes, copied where the original could still be written to.

    Args:
        data: The value offered.

    Returns:
        Immutable bytes. A ``bytearray`` or a ``memoryview`` is copied, so the document
        cannot change under a consumer that still holds the buffer.

    Raises:
        NotADocument: ``data`` is not bytes at all.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise NotADocument(
        f"a document container is read from bytes, and this is {_named(data)}."
    )


def _opened(payload: bytes) -> zipfile.ZipFile:
    """Open the container.

    Args:
        payload: The container bytes.

    Returns:
        The open archive, to be used as a context manager.

    Raises:
        NotADocument: The bytes are not a readable zip.
    """
    try:
        return zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise NotADocument(_not_a_zip(payload, error)) from error


def _not_a_zip(payload: bytes, error: Exception) -> str:
    """The message for bytes that are not a container at all."""
    opening = payload.lstrip()[:1]
    if not payload:
        hint = "There are no bytes here at all."
    elif opening == b"<":
        hint = (
            "These bytes open with '<', so this looks like plain HTML. HTML becomes a "
            "document by going through a document node, not by being renamed."
        )
    elif opening == b"{":
        hint = "These bytes look like JSON on its own rather than a container."
    else:
        hint = "A container starts with the two bytes 'PK', as every zip file does."
    return (
        f"this is not a document: a document is a zip container holding {CONTENT_ENTRY}, "
        f"{METADATA_ENTRY} and {ASSET_PREFIX}, and these {len(payload)} byte(s) are not a "
        f"zip file ({error}).\n  {hint}"
    )


def _entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """``{name: entry}`` for every file in the container, directories left out."""
    return {
        info.filename: info
        for info in archive.infolist()
        if not info.is_dir() and info.filename
    }


def _missing_content(entries: Mapping[str, zipfile.ZipInfo]) -> str:
    """The message for a zip that holds no ``content.html``."""
    names = sorted(entries)
    listed = ", ".join(names[:_NAMES_SHOWN]) or "nothing"
    if len(names) > _NAMES_SHOWN:
        listed += f", and {len(names) - _NAMES_SHOWN} more"
    if MIMETYPE_ENTRY in entries:
        hint = (
            f"It carries the {MIMETYPE_ENTRY} entry of a document, so it is a document "
            f"whose content has been removed. Restore it from a backup of the file."
        )
    else:
        hint = (
            "It is some other zip file. An archive of documents is read with the archive "
            "nodes rather than opened as one document."
        )
    return (
        f"this zip file is not a document: every document holds a {CONTENT_ENTRY} entry, "
        f"and this one holds {listed}.\n  {hint}"
    )


def _unpacked_budget(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]
) -> int:
    """How many bytes may still be unpacked out of this container.

    Args:
        archive: The open container.
        entries: Its files, from :func:`_entries`.

    Returns:
        :data:`MAX_UNPACKED_BYTES`, once the sizes the container declares are known to fit
        inside it. Every read is bounded against the same figure as well, since the sizes
        in a zip's index are only what the file claims.

    Raises:
        DocumentError: The container declares more than the limit, or holds more than
            :data:`MAX_ASSETS` embedded files.
    """
    declared = sum(max(int(info.file_size), 0) for info in archive.infolist())
    if declared > MAX_UNPACKED_BYTES:
        raise DocumentError(
            f"refusing to read a document that unpacks to {_megabytes(declared)}; the "
            f"limit is {_megabytes(MAX_UNPACKED_BYTES)}.\n"
            f"  A document that large is either damaged or built to exhaust memory."
        )
    assets = sum(1 for name in entries if name.startswith(ASSET_PREFIX))
    if assets > MAX_ASSETS:
        raise DocumentError(
            f"refusing to read a document holding {assets} embedded files; the limit is "
            f"{MAX_ASSETS}."
        )
    return MAX_UNPACKED_BYTES


def _megabytes(count: int) -> str:
    """A byte count as a rounded number of megabytes, for a message."""
    return f"{count / (1024 * 1024):.1f} MB"


def _member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int
) -> tuple[bytes, int]:
    """Read one entry, bounded by what is left of the unpacking budget.

    Args:
        archive: The open container.
        info: The entry to read.
        budget: Bytes that may still be unpacked.

    Returns:
        ``(the entry's bytes, what is left of the budget)``.

    Raises:
        NotADocument: The entry cannot be read, which covers a damaged stream and an
            encrypted one.
        DocumentError: The entry is larger than the budget allows.
    """
    try:
        with archive.open(info) as handle:
            # One byte past the budget, so an entry whose real size is larger than the size
            # the container declares is caught rather than trusted.
            raw = handle.read(budget + 1)
    except (KeyError, zipfile.BadZipFile, RuntimeError, OSError, ValueError, EOFError) as error:
        raise NotADocument(
            f"{info.filename} could not be read out of this document ({error}).\n"
            f"  The file is damaged, or it is a zip whose entries are encrypted, which a "
            f"document never is."
        ) from error
    if len(raw) > budget:
        raise DocumentError(
            f"refusing to read {info.filename}: it unpacks past the "
            f"{_megabytes(MAX_UNPACKED_BYTES)} a document may unpack to."
        )
    return raw, budget - len(raw)


def _decoded(raw: bytes, name: str) -> str:
    """One text entry as a string.

    Args:
        raw: The entry's bytes.
        name: The entry name, for the message.

    Returns:
        The decoded text, without a leading byte order mark.

    Raises:
        NotADocument: The bytes are not UTF-8.
    """
    try:
        return raw.decode(READ_ENCODING)
    except UnicodeDecodeError as error:
        raise NotADocument(
            f"{name} in this document is not UTF-8 text: byte {error.start} of it is not "
            f"valid UTF-8.\n"
            f"  Every text entry in a document is UTF-8. Reading this one any other way "
            f"would replace characters and then write the replacements back over the "
            f"original, so it is refused instead. Convert the file to UTF-8 and read it "
            f"again."
        ) from error


def _read_metadata(payload: str) -> Metadata:
    """Read ``meta.json``, version check included.

    Args:
        payload: The decoded text of the entry.

    Returns:
        The metadata record it holds.

    Raises:
        NotADocument: The text is not a readable JSON object, or its version field is not
            a version number.
        UnsupportedVersion: It names a version above :data:`CONTAINER_VERSION`.
    """
    try:
        decoded = json.loads(payload)
    except ValueError as error:
        raise NotADocument(
            f"the {METADATA_ENTRY} in this document is not readable JSON ({error}).\n"
            f"  It holds the title, the author and the copyright statement, so reading the "
            f"document without it would drop them and saving it again would write that "
            f"loss to the file. Repair {METADATA_ENTRY} inside the container, or delete it "
            f"to open the document with no metadata at all."
        ) from error
    if not isinstance(decoded, Mapping):
        raise NotADocument(
            f"the {METADATA_ENTRY} in this document holds a "
            f"{type(decoded).__name__.replace('NoneType', 'null')} where the metadata "
            f"object should be."
        )
    _require_version(decoded.get(VERSION_KEY, UNVERSIONED))
    return from_dict(decoded)


def _require_version(value: Any) -> int:
    """Confirm this build reads the layout the container names.

    Args:
        value: What ``container_version`` held, or :data:`UNVERSIONED` where the field was
            absent.

    Returns:
        The version, once it is one this build reads.

    Raises:
        NotADocument: The value is not a whole number of at least 1.
        UnsupportedVersion: The value is above :data:`CONTAINER_VERSION`.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise NotADocument(
            f"this document says its {VERSION_KEY} is {value!r}, which is not a version "
            f"number. A version is a whole number from 1 upwards."
        )
    if value > CONTAINER_VERSION:
        raise UnsupportedVersion(
            f"this document is a version {value} container and this build of WAS Node "
            f"Suite reads up to version {CONTAINER_VERSION}.\n"
            f"  The version is raised only when the layout changes in a way a reader has "
            f"to know about, so opening it here would drop whatever is new in it, and "
            f"saving it again would write that loss back to the file.\n"
            f"  Update WAS Node Suite to open this document."
        )
    return value


def _assets(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo], budget: int
) -> tuple[dict[str, bytes], int]:
    """Read every embedded file.

    Args:
        archive: The open container.
        entries: The container's files, from :func:`_entries`.
        budget: Bytes that may still be unpacked.

    Returns:
        ``({name: bytes}, what is left of the budget)``, names relative to ``assets/``.
        An entry whose name would place it outside that directory is skipped and reported,
        so nothing a caller writes to disk can escape the directory it chose.

    Raises:
        NotADocument: An entry cannot be read.
        DocumentError: An entry unpacks past the budget.
    """
    found: dict[str, bytes] = {}
    for name in sorted(entries):
        if not name.startswith(ASSET_PREFIX):
            if name not in (CONTENT_ENTRY, METADATA_ENTRY, MIMETYPE_ENTRY):
                logger.debug("%s is not part of the document format and was not read", name)
            continue
        relative = _safe_name(name[len(ASSET_PREFIX):])
        if relative is None:
            logger.warning(
                "the embedded file %r names somewhere outside %s and was skipped",
                name, ASSET_PREFIX,
            )
            continue
        if relative in found:
            # Two entries spelled differently, such as a/b.png and a/./b.png, name one
            # file. Names are read in sorted order, so which one is kept is not a matter
            # of where they sit in the container.
            logger.warning(
                "the embedded file %r names %r, which this document already holds, and was "
                "skipped", name, relative,
            )
            continue
        found[relative], budget = _member(archive, entries[name], budget)
    return found, budget


def _safe_name(name: str) -> str | None:
    """One asset name, checked and normalized.

    Args:
        name: The name as it sits under ``assets/``, or as a caller supplied it.

    Returns:
        The name spelled with ``/``, with any ``.`` segment dropped, or ``None`` when it
        cannot be one: empty, absolute, carrying a drive, holding a ``..`` segment, or
        holding a null byte. A caller writing an asset to disk still resolves the name
        through ``modules.util.sandbox``; this only stops a container naming a place
        outside the directory that caller picked.
    """
    if not isinstance(name, str):
        return None
    candidate = name.strip().replace("\\", "/")
    if not candidate or candidate.endswith("/") or "\x00" in candidate:
        return None
    if candidate.startswith("/") or ":" in candidate.split("/")[0]:
        return None
    parts = [part for part in candidate.split("/") if part and part != "."]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def _checked_assets(assets: Mapping[str, bytes] | None) -> dict[str, bytes]:
    """The embedded files a document is being built with, checked and copied.

    Args:
        assets: ``{name: bytes}``, or ``None``.

    Returns:
        A new dictionary the document owns, so a caller that goes on using its own is not
        editing the document.

    Raises:
        DocumentError: A name cannot go in a container, two names normalize to one, a
            value is not bytes, or there are more than :data:`MAX_ASSETS` of them.
    """
    if not assets:
        return {}
    if not isinstance(assets, Mapping):
        raise DocumentError(
            f"a document's embedded files are given as a mapping of names to bytes, and "
            f"this is {_named(assets)}."
        )
    if len(assets) > MAX_ASSETS:
        raise DocumentError(
            f"a document holds at most {MAX_ASSETS} embedded files and this one was given "
            f"{len(assets)}."
        )
    checked: dict[str, bytes] = {}
    for name, value in assets.items():
        if isinstance(name, str) and name.strip().replace("\\", "/").startswith(ASSET_PREFIX):
            raise DocumentError(
                f"the embedded file {name!r} is named with the {ASSET_PREFIX} prefix "
                f"already on it. A name is relative to that directory, so it is "
                f"{name.strip()[len(ASSET_PREFIX):]!r}."
            )
        relative = _safe_name(name)
        if relative is None:
            raise DocumentError(
                f"{name!r} cannot name an embedded file. A name is a relative path inside "
                f"{ASSET_PREFIX}, such as 'logo.png' or 'figures/plot.png': it may not be "
                f"empty, start at a root, carry a drive, or step out with '..'."
            )
        if relative in checked:
            raise DocumentError(
                f"two embedded files are both named {relative!r}, so one would replace the "
                f"other in the container."
            )
        if isinstance(value, (bytearray, memoryview)):
            value = bytes(value)
        if not isinstance(value, bytes):
            raise DocumentError(
                f"the embedded file {relative!r} holds {_named(value)} and an embedded file is "
                f"bytes. Encode it before putting it in a document."
            )
        checked[relative] = value
    return checked


def _put(archive: zipfile.ZipFile, name: str, payload: bytes, compression: int) -> None:
    """Write one entry with the fixed timestamp and permissions.

    Args:
        archive: The archive being written.
        name: Entry name, spelled with ``/``.
        payload: The entry's bytes.
        compression: ``zipfile.ZIP_STORED`` or ``zipfile.ZIP_DEFLATED``.
    """
    info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    info.compress_type = compression
    info.external_attr = ENTRY_MODE
    archive.writestr(info, payload)
