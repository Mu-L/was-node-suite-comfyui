"""Registering the ComfyUI nodes a view extension ships.

Every module under an extension's unpacked ``nodes/`` directory is imported, and its
``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` entries are written into
ComfyUI's own registries.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ... import log
from ..install import EXTENSION_PREFIX, extension_slug, loads_as_pack

__all__ = ["register_extension_nodes"]

logger = log.get_logger("viewer.nodes")

#: Prefix marking a name as private to this package rather than part of an extension.
PRIVATE = "_"

#: This directory, which every extension's node modules sit under.
ROOT = Path(__file__).resolve().parent

#: How ComfyUI names the pack a registered class came from, read by ComfyUI Manager. The
#: loader writes ``custom_nodes`` joined to the pack directory's own name.
OWNER = f"custom_nodes.{ROOT.parents[2].name}"

#: Node ids named in full in a clash warning before the remainder is counted. A module that
#: binds the name ``NODE_CLASS_MAPPINGS`` to a dict it imported rather than declared can
#: offer thousands of ids at once, and a console line each buries every other message.
CLASH_SAMPLE = 5

#: ``node id -> the extension that registered it``. ComfyUI's registry says an id is taken
#: and never by whom, so an id one of an extension's own modules re-exports from a sibling,
#: and every id still standing from an earlier call, would read as a foreign pack's.
_REGISTERED: dict[str, str] = {}


def extension_dirs() -> list[Path]:
    """One directory per installed extension, in a stable order."""
    try:
        return sorted(
            path
            for path in ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(PRIVATE)
        )
    except OSError as error:
        logger.debug("%s could not be listed (%s)", ROOT, error)
        return []


def clone_names() -> set[str]:
    """Extension names ComfyUI registers the nodes of from ``custom_nodes`` itself.

    Returns:
        Both the directory name and the package slug of every ``ComfyUI_Viewer_*`` sibling
        holding an ``__init__.py``, which is what ComfyUI imports a pack through. A copy of
        one's nodes under this directory is skipped, whichever pack ComfyUI loads first.
    """
    try:
        siblings = sorted(ROOT.parents[3].iterdir())
    except (OSError, IndexError) as error:
        logger.debug("the packs beside this one could not be listed (%s)", error)
        return set()

    names: set[str] = set()
    for entry in siblings:
        if not entry.name.startswith(EXTENSION_PREFIX):
            continue
        try:
            if not entry.is_dir() or not loads_as_pack(entry):
                continue
        except OSError:
            continue
        names.add(entry.name)
        names.add(extension_slug(entry.name))
    return names


def node_modules(extension: Path) -> list[Path]:
    """Every module under one extension's directory, in a stable order.

    Args:
        extension: The extension's directory.

    Returns:
        The ``.py`` files in it and below it, private names and their subtrees excluded.
    """
    try:
        found = sorted(extension.rglob("*.py"))
    except OSError as error:
        logger.debug("%s could not be walked (%s)", extension, error)
        return []
    return [
        path
        for path in found
        if not any(part.startswith(PRIVATE) for part in path.relative_to(extension).parts)
    ]


def register_extension_nodes(reserved: set[str] | None = None) -> int:
    """Register the nodes of every installed view extension.

    Args:
        reserved: Node ids this pack is about to register itself. An extension may not
            take one, since ComfyUI would let the pack's own overwrite it unannounced.

    Returns:
        How many node ids were registered. Zero outside ComfyUI, where there are no
        registries to write into.
    """
    # ComfyUI's registries are mutated in place rather than exported from this pack.
    # load_custom_node reads NODE_CLASS_MAPPINGS off the pack's own module first and
    # returns before it reaches comfy_entrypoint, so binding that name here would cost
    # every node the pack has.
    registry = sys.modules.get("nodes")
    classes = getattr(registry, "NODE_CLASS_MAPPINGS", None)
    labels = getattr(registry, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if not isinstance(classes, dict) or not isinstance(labels, dict):
        logger.debug("ComfyUI's node registries are absent, so no extension nodes were registered")
        return 0

    directories = extension_dirs()
    if not directories:
        return 0

    # An extension is unpacked before this package is first imported, and the directories
    # it wrote carry no __init__.py, so the finders are refreshed before they are searched.
    importlib.invalidate_caches()

    clones = clone_names()
    taken = set(classes) | set(reserved or ())
    registered = 0
    extensions = 0
    for extension in directories:
        if extension.name in clones:
            logger.debug(
                "%s is installed in custom_nodes as well, where ComfyUI registers its "
                "nodes, so the copy under %s was left alone",
                extension.name, ROOT.name,
            )
            continue
        extensions += 1
        registered += _register_extension(extension, classes, labels, taken)
    if registered:
        logger.info(
            "content viewer: %s node(s) registered from %s view extension(s)",
            registered, extensions,
        )
    return registered


def _register_extension(extension: Path, classes: dict, labels: dict, taken: set[str]) -> int:
    """Import one extension's modules and register the nodes they declare."""
    modules = node_modules(extension)
    if not modules or not _import_package(extension):
        return 0

    registered = 0
    clashes: list[str] = []
    for path in modules:
        module = _import_module(path, extension.name)
        if module is not None:
            registered += _register_module(module, extension.name, classes, labels, taken, clashes)
    if clashes:
        _report_clashes(extension.name, clashes)
    return registered


def _import_package(extension: Path) -> bool:
    """Import an extension's directory as a package, before any module inside it.

    Args:
        extension: The extension's directory.

    Returns:
        Whether it imported. A directory with no ``__init__.py`` is a namespace package
        and always does; one copied in with an ``__init__.py`` runs it here, where a
        failure is reported against the directory rather than against whichever module
        happened to trigger the import.
    """
    try:
        importlib.import_module(f"{__name__}.{extension.name}")
    except (Exception, SystemExit) as error:
        logger.warning(
            "the view extension %s could not be imported (%s: %s), so none of its nodes "
            "are loaded. An __init__.py in its directory runs as the package, and a copy "
            "taken by hand should leave that file behind: each node module here is "
            "imported on its own",
            extension.name, type(error).__name__, error,
        )
        logger.debug("%s failed to import", extension, exc_info=True)
        return False
    return True


def _import_module(path: Path, label: str):
    """Import one of an extension's modules, or return ``None`` when it will not import."""
    name = ".".join((__name__, *path.relative_to(ROOT).with_suffix("").parts))
    try:
        return importlib.import_module(name)
    # SystemExit is not an Exception, and a node module written against the V1 API calls
    # sys.exit() to bail on a missing dependency. Uncaught it leaves get_node_list, leaves
    # load_custom_node, and reaches main.py, where nothing guards the call: one extension
    # module would stop ComfyUI starting at all. KeyboardInterrupt still propagates, so
    # Ctrl+C during startup keeps working.
    except (Exception, SystemExit) as error:
        logger.warning(
            "%s from the view extension %s could not be imported (%s: %s), so any node in "
            "it is missing and a workflow naming one opens with a hole where it was",
            path.name, label, type(error).__name__, error,
        )
        logger.debug("%s failed to import", path, exc_info=True)
        return None


def _register_module(
    module, label: str, classes: dict, labels: dict, taken: set[str], clashes: list[str]
) -> int:
    """Write one module's node mappings into ComfyUI's, skipping ids already spoken for."""
    declared = getattr(module, "NODE_CLASS_MAPPINGS", None)
    # `from nodes import NODE_CLASS_MAPPINGS` binds ComfyUI's live registry as a module
    # attribute, and an attribute cannot say whether a module declared a mapping or
    # imported one. Walking that dict would offer every node id in the install as the
    # extension's own and report each one as a clash.
    if not isinstance(declared, dict) or declared is classes:
        return 0
    named = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", None)
    named = named if isinstance(named, dict) and named is not labels else {}

    registered = 0
    for node_id, node_cls in declared.items():
        if _REGISTERED.get(node_id) == label:
            continue
        if node_id in taken:
            clashes.append(node_id)
            continue
        classes[node_id] = node_cls
        taken.add(node_id)
        _REGISTERED[node_id] = label
        if node_id in named:
            labels[node_id] = named[node_id]
        _set_owner(node_cls, node_id, label)
        registered += 1
    return registered


def _report_clashes(label: str, clashes: list[str]) -> None:
    """Report the ids one extension declared and something else already provides."""
    shown = ", ".join(repr(node_id) for node_id in clashes[:CLASH_SAMPLE])
    if len(clashes) > CLASH_SAMPLE:
        shown = f"{shown} and {len(clashes) - CLASH_SAMPLE} more"
    logger.warning(
        "the view extension %s declares %s node id(s) that are registered elsewhere "
        "already (%s), so those nodes were not loaded. One node id can only come from one "
        "place: uninstall whichever of the two provides it twice, or ask its author for "
        "different ids",
        label, len(clashes), shown,
    )


def _set_owner(node_cls, node_id: str, label: str) -> None:
    """Name this pack as where a registered class came from, which ComfyUI Manager reads."""
    try:
        node_cls.RELATIVE_PYTHON_MODULE = OWNER
    except Exception as error:
        logger.debug(
            "%r from %s could not be given a source module (%s), so a manager listing it "
            "will not name the pack it came from",
            node_id, label, error,
        )
