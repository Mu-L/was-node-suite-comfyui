"""Prompt text processing: NSP noodles, wildcards, dynamic prompts, variables, styles.

Five parsers plus the style library, each in its own module and imported from there. All
draw from the shared :mod:`random` module rather than a private ``Random`` instance.
"""

from __future__ import annotations

from pathlib import Path

from .. import log

__all__ = ["state_path"]

logger = log.get_logger("prompt")


def state_path(name: str) -> Path:
    """Resolve a writable file in the pack's config directory and create its directory.

    Args:
        name: A file name, such as ``"nsp_pantry.json"``.

    Returns:
        The absolute path to that file, whether or not it exists yet. A directory that
        could not be created is logged rather than raised, so the path may not be
        writable.
    """
    from .. import config

    path = config.state_file(name)
    # The config directory does not exist until something puts a file in it, so it is
    # created here rather than at each write site.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.warning("%s could not be created (%s), so %s cannot be saved", path.parent, error, name)
    return path
