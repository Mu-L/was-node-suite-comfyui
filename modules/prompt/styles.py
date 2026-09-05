"""The style library behind the prompt style selectors.

A style is a named ``{"prompt": ..., "negative_prompt": ...}`` pair in the state database,
reached through :class:`PromptStyles`. A ``.json`` library or an AUTOMATIC1111
``styles.csv`` imports into it and exports back out.
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

from . import state_path
from .. import log
from ..state import store as store_module

__all__ = [
    "SOURCE",
    "STYLES_FILE",
    "PromptStyles",
    "export_styles",
    "import_styles",
    "library",
    "open_styles",
    "save_style",
]

logger = log.get_logger("prompt.styles")

#: The file name a library is imported from and exported to when no other is named.
STYLES_FILE = "styles.json"

#: Record field naming the file a style came from. A style saved on this machine has none.
SOURCE = "source"

#: Meta key holding the size and modification time of each library that was imported.
IMPORTED = "styles:imported"

#: The two fields a style carries, in the order an export writes them.
FIELDS = ("prompt", "negative_prompt")


def _database():
    """The shared state database."""
    return store_module.shared_store()


def _rows() -> list[tuple[str, dict]]:
    """Every stored style as ``(name, body)``, in library order, provenance included."""
    return [
        (record.name, dict(record.fields or {}))
        for record in _database().records(store_module.STYLES, store_module.DEFAULT_CATEGORY)
    ]


def _prompts(body) -> dict[str, str]:
    """One style's two prompts, with anything else in the record left out."""
    return {field: str((body or {}).get(field, "") or "") for field in FIELDS}


def library() -> dict[str, dict[str, str]]:
    """Every style in the library.

    Returns:
        ``{name: {"prompt": ..., "negative_prompt": ...}}``, in library order.
    """
    return {name: _prompts(body) for name, body in _rows()}


def save_style(name: str, prompt: str = "", negative_prompt: str = "") -> bool:
    """Store one style under a name, replacing a style already stored under it.

    Args:
        name: The name to store it under.
        prompt: The positive prompt.
        negative_prompt: The negative prompt.

    Returns:
        True when the write committed.
    """
    return _database().set_record(
        store_module.STYLES,
        store_module.DEFAULT_CATEGORY,
        str(name),
        {"prompt": str(prompt or ""), "negative_prompt": str(negative_prompt or "")},
    )


def read_style_file(source) -> dict[str, dict[str, str]]:
    """Read a style library out of a file.

    Args:
        source: A ``.json`` library, or an AUTOMATIC1111 ``.csv`` with ``name``,
            ``prompt`` and ``negative_prompt`` columns.

    Returns:
        ``{name: {"prompt": ..., "negative_prompt": ...}}``, empty when the file holds no
        styles or could not be read.
    """
    path = Path(source)
    if not path.is_file():
        logger.error("styles file `%s` does not exist.", path)
        return {}
    try:
        if path.suffix.lower() == ".csv":
            return _read_csv(path)
        return _read_json(path)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        logger.error("the style library `%s` could not be read (%s).", path, error)
        return {}


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    """An AUTOMATIC1111 ``styles.csv`` as ``{name: {"prompt": ..., ...}}``."""
    styles: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name")
            if not name:
                continue
            styles[name] = {
                "prompt": row.get("prompt") or row.get("text") or "",
                "negative_prompt": row.get("negative_prompt") or "",
            }
    return styles


def _read_json(path: Path) -> dict[str, dict[str, str]]:
    """A JSON style library as ``{name: {"prompt": ..., "negative_prompt": ...}}``."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        logger.error(
            "the style library `%s` holds a %s where a list of named styles was expected, "
            "so no style was read from it.",
            path,
            type(data).__name__,
        )
        return {}
    return {str(name): _prompts(body) for name, body in data.items() if isinstance(body, dict)}


def _key(source) -> str:
    """The provenance value marking every style imported from one file."""
    try:
        return str(Path(source).resolve())
    except OSError:
        return str(source)


def _stamp(source) -> str | None:
    """One file's size and modification time, or None when it is not there."""
    try:
        info = Path(source).stat()
    except OSError:
        return None
    return f"{info.st_size}:{info.st_mtime_ns}"


def _changed(source) -> bool:
    """Whether a file differs from the copy of it the library already holds."""
    current = _stamp(source)
    if current is None:
        return False
    try:
        seen = json.loads(_database().meta(IMPORTED) or "{}")
    except ValueError:
        seen = {}
    return not isinstance(seen, dict) or seen.get(_key(source)) != current


def import_styles(source, replace: bool = False) -> int:
    """Import a style library from a file, keeping every style saved on this machine.

    Args:
        source: A ``.json`` library or an AUTOMATIC1111 ``.csv``. A style the same file
            brought in earlier and no longer names is removed with it.
        replace: Empty the library first, so it holds exactly what the file holds.

    Returns:
        How many styles the file held.
    """
    stamp = _stamp(source)
    if stamp is None:
        logger.error("styles file `%s` does not exist.", source)
        return 0
    styles = read_style_file(source)
    key = _key(source)
    kept: list[tuple[str, dict]] = []
    if not replace:
        kept = [
            (name, body)
            for name, body in _rows()
            if body.get(SOURCE) != key and name not in styles
        ]
    incoming = [(name, {**_prompts(body), SOURCE: key}) for name, body in styles.items()]
    rows = kept + incoming

    def work(connection):
        store_module.write_category(
            connection, store_module.STYLES, store_module.DEFAULT_CATEGORY, rows
        )
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?", (IMPORTED,)
            ).fetchone()
            seen = json.loads(row[0]) if row else {}
        except (TypeError, ValueError):
            seen = {}
        if not isinstance(seen, dict):
            seen = {}
        seen[key] = stamp
        store_module.write_meta(connection, IMPORTED, json.dumps(seen))

    if not _database().write(work, f"the styles imported from {Path(source).name}"):
        return 0
    logger.info(
        "imported %s style(s) from %s; the file is left where it is",
        len(incoming),
        Path(source).name,
    )
    return len(incoming)


def export_styles(target=None) -> int:
    """Write the style library out to a file.

    Args:
        target: Where to write. A ``.csv`` suffix writes AUTOMATIC1111 columns, any other
            suffix writes JSON. Defaults to ``styles.json`` in the config directory.

    Returns:
        How many styles were written, or 0 when the file could not be written.
    """
    path = Path(target) if target is not None else state_path(STYLES_FILE)
    styles = library()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("name", *FIELDS))
                for name, body in styles.items():
                    writer.writerow((name, body["prompt"], body["negative_prompt"]))
        else:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(styles, handle, indent=4)
    except OSError as error:
        logger.error("the style library could not be written to `%s` (%s).", path, error)
        return 0
    logger.info("wrote %s style(s) to %s", len(styles), path)
    return len(styles)


def open_styles(path=None) -> PromptStyles:
    """Open the style library, importing the configured file when it has changed.

    Args:
        path: A ``.json`` library or an A1111 ``.csv`` to import first. Defaults to the
            ``paths.styles`` config key, which falls back to ``styles.json`` in the
            config directory.

    Returns:
        The library. A file that is not there leaves the library as it stands, so a
        selector shows the styles already stored rather than failing to load.
    """
    from .. import config

    source = Path(path) if path is not None else config.styles_file()
    if _changed(source):
        import_styles(source)
    return PromptStyles()


class PromptStyles:
    """The named prompt and negative-prompt pairs held in the state database.

    Attributes:
        styles: ``{name: {"prompt": ..., "negative_prompt": ...}}``, read at construction.
        preview_length: How many characters of a prompt an auto-generated name quotes.
    """

    def __init__(self, source=None, preview_length=32):
        """Read the library, importing a file into it first.

        Args:
            source: A ``.json`` library or an A1111 ``.csv`` to import, or None to read
                the library as it stands.
            preview_length: How many characters of a prompt an auto-generated name quotes.
        """
        self.preview_length = preview_length
        if source is not None:
            import_styles(source)
        self.styles = library()

    def add_style(self, prompt="", negative_prompt="", auto=False, name=None):
        """Add a style to the library.

        Args:
            prompt: The positive prompt to store.
            negative_prompt: The negative prompt to store.
            auto: Name the style from the current date and the first
                :attr:`preview_length` characters of whichever prompt is non-empty,
                positive first.
            name: The name to store the style under. Required unless ``auto`` is set.

        Returns:
            The name the pair is stored under, which is the name it already had where the
            library holds the same two prompts. None where no name could be worked out,
            which is a missing ``name`` or ``auto`` with two empty prompts, and is logged.
        """
        if auto:
            date_format = "%A, %d %B %Y %I:%M %p"
            date_str = datetime.datetime.now().strftime(date_format)
            key = None
            if prompt.strip() != "":
                length = min(len(prompt), self.preview_length)
                key = f"[{date_str}] Positive: {prompt[:length]} ..."
            elif negative_prompt.strip() != "":
                length = min(len(negative_prompt), self.preview_length)
                key = f"[{date_str}] Negative: {negative_prompt[:length]} ..."
            else:
                logger.error("At least a `prompt`, or `negative_prompt` input is required!")
                return None
        else:
            if name is None or str(name).strip() == "":
                logger.error("A `name` input is required when not using `auto=True`")
                return None
            key = str(name)

        for stored, value in self.styles.items():
            if value["prompt"] == prompt and value["negative_prompt"] == negative_prompt:
                return stored

        save_style(key, prompt, negative_prompt)
        self.styles[key] = {"prompt": prompt, "negative_prompt": negative_prompt}
        return key

    def get_prompts(self):
        """Every style, as ``{name: {"prompt": ..., "negative_prompt": ...}}``."""
        return self.styles

    def get_prompt(self, prompt_key):
        """The prompt pair stored under a name.

        Args:
            prompt_key: The style name to look up.

        Returns:
            A ``(prompt, negative_prompt)`` pair, or ``(None, None)`` when the name is not
            in the library, which is logged as an error.
        """
        if prompt_key in self.styles:
            return self.styles[prompt_key]["prompt"], self.styles[prompt_key]["negative_prompt"]
        logger.error("Prompt style `%s` was not found!", prompt_key)
        return None, None
