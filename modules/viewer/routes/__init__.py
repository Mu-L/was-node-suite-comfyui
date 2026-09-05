"""HTTP routes belonging to the content viewer's views.

A route file is any ``.py`` here whose name does not start with ``_``, and registers its
handlers on ``PromptServer.instance.routes``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ... import log

__all__ = ["load_routes"]

logger = log.get_logger("viewer.routes")

#: Prefix marking a file as private to this package rather than a route file.
PRIVATE = "_"


def route_files() -> list[Path]:
    """The route files in this directory, in a stable order."""
    directory = Path(__file__).resolve().parent
    try:
        return sorted(
            path
            for path in directory.iterdir()
            if path.suffix == ".py" and not path.name.startswith(PRIVATE)
        )
    except OSError as error:
        logger.debug("%s could not be listed (%s)", directory, error)
        return []


# Registration happens on this call rather than at import: a route registered at import
# time would be added by any tool that merely reads the module tree.
def load_routes() -> int:
    """Register the handlers in every route file.

    Returns:
        How many files registered without raising. A file that raises is logged with its
        name and skipped, leaving the others registered.
    """
    loaded = 0
    for path in route_files():
        module_name = f"{__name__}.{path.stem}"
        try:
            # An extension is unpacked here after this package was first imported, so a
            # stale package cache would hide its route file from a normal import.
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.warning("%s could not be read as a module, so its routes are not served", path)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as error:
            sys.modules.pop(module_name, None)
            logger.warning(
                "the viewer routes in %s were not registered (%s: %s), so whichever view "
                "needs them will report a failed request",
                path.name, type(error).__name__, error,
            )
            logger.debug("%s failed to load", path, exc_info=True)
            continue
        loaded += 1
    if loaded:
        logger.debug("registered the routes in %s file(s)", loaded)
    return loaded
