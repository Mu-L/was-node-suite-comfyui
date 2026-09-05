"""The dictionary a loop reports itself through, and the keys it holds.

Every key is always present. An Open node describes the iteration about to run, a Close
node the loop that has finished.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEFAULTS", "KEYS", "build", "read"]

#: How the collected values left the loop: one value, a list of them, or one joined batch.
FINAL = "final"
LIST = "list"
BATCH = "batch"

#: What ``stopped_reason`` says before a loop has reported anything, so a reader is never handed
#: an empty line to make sense of.
NOT_STARTED = "Not started."

#: What it says while the loop is still going. An Open node reports on an iteration that has not
#: finished, so the field describes where the loop is rather than staying blank until the end.
RUNNING = "Still running, iteration {iteration}."

#: Every key, with what it means when the loop has said nothing about it yet.
DEFAULTS = {
    "mode": "iterations",
    "current_iteration": 0,
    "index": 0,
    "iterations_completed": 0,
    "limit": 0,
    "accumulated_count": 0,
    "accumulated_as": FINAL,
    "stopped_reason": NOT_STARTED,
}

KEYS = tuple(DEFAULTS)


def build(**values: Any) -> dict:
    """One metadata dictionary, with every key present.

    Args:
        **values: Any of :data:`KEYS`. Anything else is ignored rather than carried, so a
            typo cannot quietly add a key no reader looks for.

    Returns:
        A new dictionary holding every key in :data:`KEYS`. ``stopped_reason`` is never empty:
        a build that does not name one describes the loop as still running instead.
    """
    built = {key: values.get(key, default) for key, default in DEFAULTS.items()}
    # Read from what the caller passed rather than from the built value, whose default is itself
    # a sentence: testing the built one for emptiness would never fire.
    if not str(values.get("stopped_reason", "")).strip():
        iteration = built["current_iteration"]
        built["stopped_reason"] = RUNNING.format(iteration=iteration) if iteration else NOT_STARTED
    return built


def read(metadata: Any, key: str) -> Any:
    """One value out of a metadata dictionary, whatever arrived on the socket.

    Args:
        metadata: The dictionary a loop emitted, or anything else.
        key: One of :data:`KEYS`.

    Returns:
        The value, or the key's default when the dictionary is missing it or is not a
        dictionary at all.
    """
    if not isinstance(metadata, dict):
        return DEFAULTS[key]
    return metadata.get(key, DEFAULTS[key])
