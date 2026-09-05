"""The pack's key and value state, on the sqlite store.

:class:`WASDatabase` holds settings, custom tokens and node cursors as
``{category: {key: value}}``. :class:`HistoryDatabase` holds the history lists, where
each key names one ordered list of paths.
"""

from __future__ import annotations

from typing import Any

from .. import log
from . import store as store_module

__all__ = [
    "HISTORY_CATEGORY",
    "HISTORY_FILE",
    "SETTINGS_FILE",
    "HistoryDatabase",
    "WASDatabase",
    "get_history_db",
    "get_settings_db",
    "open_database",
]

logger = log.get_logger("state")

#: Settings, custom tokens and node cursors.
SETTINGS_FILE = "was_suite_settings.json"

#: The image, output-image and text-file history lists.
HISTORY_FILE = "was_history.json"

#: The single category the three history lists sit under.
HISTORY_CATEGORY = "History"

_settings_db: WASDatabase | None = None
_history_db: HistoryDatabase | None = None


class WASDatabase:
    """A key and value store namespaced one level deep by category.

    Attributes:
        store: The sqlite database the values are kept in.
        name: Which store inside it the categories belong to.
    """

    def __init__(self, name: str = store_module.SETTINGS, database=None):
        """Attach to one key and value store.

        Args:
            name: Store name, one of :data:`~modules.state.store.STORES`.
            database: The :class:`~modules.state.store.StateStore` to read and write.
                Defaults to the shared one.
        """
        self.name = name
        self.store = store_module.shared_store() if database is None else database

    def catExists(self, category: str) -> bool:
        """Whether ``category`` exists, empty or not."""
        return self.store.has_category(self.name, category)

    def keyExists(self, category: str, key: str) -> bool:
        """Whether ``key`` exists inside ``category``."""
        return self.store.has_key(self.name, category, key)

    def insert(self, category: str, key: str, value: Any) -> None:
        """Store ``value`` at ``category``/``key``, creating the category if absent.

        Args:
            category: Category name. A non-string is refused with an error.
            key: Key name. A non-string is refused with an error.
            value: Anything ``json.dumps`` accepts.
        """
        if not isinstance(category, str) or not isinstance(key, str):
            logger.error("Category and key must be strings")
            return
        self.store.set(self.name, category, key, value)

    def update(self, category: str, key: str, value: Any) -> None:
        """Overwrite an existing ``category``/``key``. A missing one is left alone."""
        self.store.update(self.name, category, key, value)

    def updateCat(self, category: str, dictionary: dict) -> None:
        """Merge ``dictionary`` into ``category``, creating the category if absent."""
        self.store.set_many(self.name, category, dictionary)

    def get(self, category: str, key: str) -> Any:
        """The value at ``category``/``key``, or ``None`` if either is absent."""
        return self.store.get(self.name, category, key)

    def getDB(self) -> dict:
        """The whole database as ``{category: {key: value}}``, read at each call."""
        return self.store.dump(self.name)

    def insertCat(self, category: str) -> None:
        """Create an empty ``category``.

        A non-string name and a category that already exists are both refused with an
        error rather than an exception.
        """
        if not isinstance(category, str):
            logger.error("Category must be a string")
            return

        if self.store.has_category(self.name, category):
            logger.error("The database category '%s' already exists!", category)
            return
        self.store.add_category(self.name, category)

    def getDict(self, category: str) -> dict:
        """Every key and value of ``category``, or an empty dict if it does not exist."""
        if not self.store.has_category(self.name, category):
            logger.error("The database category '%s' does not exist!", category)
            return {}
        return self.store.category(self.name, category)

    def delete(self, category: str, key: str) -> None:
        """Remove ``category``/``key``. A missing one is left alone."""
        self.store.delete(self.name, category, key)


class HistoryDatabase:
    """The history lists, one ordered list of paths per key.

    Attributes:
        store: The sqlite database the paths are kept in.
        name: Which store inside it the lists belong to.
    """

    def __init__(self, database=None):
        """Attach to the history store.

        Args:
            database: The :class:`~modules.state.store.StateStore` to read and write.
                Defaults to the shared one.
        """
        self.name = store_module.HISTORY
        self.store = store_module.shared_store() if database is None else database

    def catExists(self, category: str) -> bool:
        """Whether ``category`` is :data:`HISTORY_CATEGORY` and any list exists."""
        return category == HISTORY_CATEGORY and bool(self.store.record_categories(self.name))

    def keyExists(self, category: str, key: str) -> bool:
        """Whether ``key`` names a list, empty or not."""
        return category == HISTORY_CATEGORY and key in self.store.record_categories(self.name)

    def keys(self) -> list[str]:
        """Every list name, in the order the lists were first written to."""
        return self.store.record_categories(self.name)

    def get(self, category: str, key: str) -> list[str]:
        """The paths stored under ``key``, oldest first."""
        if category != HISTORY_CATEGORY:
            return []
        return self.store.record_names(self.name, key)

    def newest(self, key: str, limit: int, skip: int = 0) -> list[str]:
        """The last paths recorded under ``key``, newest first.

        Args:
            key: List name, such as ``"Images"``.
            limit: At most this many paths.
            skip: How many of the newest to pass over first.

        Returns:
            Paths in reverse order, newest first.
        """
        return self.store.newest_record_names(self.name, key, limit, skip)

    def getDB(self) -> dict:
        """The whole history as ``{"History": {key: [path, ...]}}``."""
        return {HISTORY_CATEGORY: self.getDict(HISTORY_CATEGORY)}

    def getDict(self, category: str) -> dict:
        """Every list of one category as ``{key: [path, ...]}``."""
        if category != HISTORY_CATEGORY:
            logger.error("The database category '%s' does not exist!", category)
            return {}
        grouped = self.store.dump_records(self.name)
        return {key: [record.name for record in records] for key, records in grouped.items()}

    def append(self, key: str, paths: list[str]) -> None:
        """Add paths to the end of one list, moving one already in it to the end.

        Args:
            key: List name, such as ``"Images"``.
            paths: Absolute paths, appended in the order given.
        """
        self.store.append_records(
            self.name, key, [(path, None) for path in paths], unique=True
        )

    def insert(self, category: str, key: str, value: Any) -> None:
        """Replace the list under ``key`` with ``value``.

        Args:
            category: Must be :data:`HISTORY_CATEGORY`.
            key: List name.
            value: The paths the list should hold, in order.
        """
        if category != HISTORY_CATEGORY:
            logger.error(
                "the history holds one category, '%s', and '%s' was asked for, so nothing "
                "was recorded",
                HISTORY_CATEGORY,
                category,
            )
            return
        self.store.set_records(self.name, key, [(str(path), None) for path in value or []])

    def update(self, category: str, key: str, value: Any) -> None:
        """Replace the list under ``key``, leaving a key that has no list alone."""
        if not self.keyExists(category, key):
            return
        self.insert(category, key, value)

    def updateCat(self, category: str, dictionary: dict) -> None:
        """Replace each named list with the paths given for it."""
        for key, paths in (dictionary or {}).items():
            self.insert(category, key, paths)

    def insertCat(self, category: str) -> None:
        """Accepted for the settings surface. The history has one category and it exists."""

    def delete(self, category: str, key: str) -> None:
        """Remove the whole list stored under ``key``."""
        if category != HISTORY_CATEGORY:
            return
        self.store.delete_record_category(self.name, key)


def open_database(name: str = SETTINGS_FILE):
    """Open one of the pack's stores by name.

    Args:
        name: :data:`SETTINGS_FILE` or :data:`HISTORY_FILE`.

    Returns:
        A :class:`HistoryDatabase` for :data:`HISTORY_FILE`, a :class:`WASDatabase` for
        anything else.
    """
    if name == HISTORY_FILE:
        return get_history_db()
    return get_settings_db()


def get_settings_db() -> WASDatabase:
    """The settings database.

    Returns:
        The process-wide settings database. Every read reaches the file, so a change
        another process made is visible at once.
    """
    global _settings_db
    if _settings_db is None:
        _settings_db = WASDatabase(store_module.SETTINGS)
    return _settings_db


def get_history_db() -> HistoryDatabase:
    """The history database.

    Returns:
        The process-wide history database.
    """
    global _history_db
    if _history_db is None:
        _history_db = HistoryDatabase()
    return _history_db
