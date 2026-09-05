"""Holding a run still until someone resumes it.

``POST /was/interface/api/pause`` carries ``{"node_id", "action", "value"}``, where ``action``
is ``resume`` or ``cancel`` and ``value`` is whatever the node asked the user to edit.
``GET /was/interface/api/pause`` answers which nodes are waiting.
"""

from __future__ import annotations

import threading
import time

from .. import log

__all__ = [
    "ROUTE",
    "TICK",
    "RESUMED",
    "CANCELLED",
    "TIMED_OUT",
    "waiting",
    "wait_for_resume",
    "register_routes",
]

logger = log.get_logger("interface.pause")

#: The one route the browser resumes through.
ROUTE = "/was/interface/api/pause"

#: Seconds between checks while a node is held.
TICK = 0.1

#: How a hold ended.
RESUMED = "resumed"
CANCELLED = "cancelled"
TIMED_OUT = "timed out"

#: Node id -> what it is waiting for, while it waits. Read on the server's thread and
#: written on the worker's, so every touch takes the lock beside it.
_holds: dict[str, dict] = {}
_lock = threading.Lock()

_registered = False


def waiting() -> list[dict]:
    """Every node holding a run still.

    Returns:
        One entry per held node, each ``{"node_id", "message", "waited"}``.
    """
    now = time.monotonic()
    with _lock:
        current = list(_holds.items())
    return [
        {"node_id": node_id, "message": hold.get("message", ""),
         "kind": hold.get("kind", "none"), "content": hold.get("content", ""),
         "timeout": hold.get("timeout", 0.0),
         "waited": round(now - hold["started"], 1)}
        for node_id, hold in current
    ]


def _announce(node_id: str, message: str, timeout: float, kind: str = "none") -> None:
    """Tell the browser a node is waiting."""
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "was-pause",
            {"node_id": node_id, "message": message, "timeout": timeout, "kind": kind},
        )
    except Exception as error:
        logger.debug("a paused node could not be announced (%s)", error)


def _released(node_id: str, action: str) -> None:
    """Tell the browser a node is no longer waiting."""
    try:
        from server import PromptServer

        PromptServer.instance.send_sync("was-pause-done", {"node_id": node_id, "action": action})
    except Exception as error:
        logger.debug("a resumed node could not be announced (%s)", error)


def wait_for_resume(
    node_id: str, timeout: float = 0.0, message: str = "",
    kind: str = "none", content: str = "",
) -> tuple[str, str]:
    """Hold the run until the browser resumes it, cancels it, or the wait runs out.

    Args:
        node_id: The node holding the run, which is what the browser resumes by.
        timeout: Seconds to wait, or 0 to wait with no limit.
        message: Text drawn beside the resume control.
        kind: What is on offer to edit: ``"none"``, ``"text"`` or ``"canvas"``.
        content: What to edit, which the browser reads back from the route.

    Returns:
        How the hold ended, and the value the browser sent back, empty where it sent none.

    Raises:
        InterruptProcessingException: The run was cancelled, from the browser or from
            ComfyUI's own cancel.
    """
    import comfy.model_management

    key = str(node_id)
    with _lock:
        _holds[key] = {"started": time.monotonic(), "message": message,
                       "kind": kind, "content": content, "timeout": timeout,
                       "action": None, "value": ""}
    _announce(key, message, timeout, kind)
    told = False
    try:
        while True:
            # This clears ComfyUI's interrupt flag as it raises, which is what stops the flag
            # carrying into the next node. Testing the flag and raising by hand would leave it
            # set and cancel whatever ran next.
            comfy.model_management.throw_exception_if_processing_interrupted()
            with _lock:
                hold = dict(_holds.get(key) or {}) or None
            if hold is None:
                return RESUMED, ""
            action = hold.get("action")
            if action == CANCELLED:
                _released(key, CANCELLED)
                told = True
                raise comfy.model_management.InterruptProcessingException()
            if action == RESUMED:
                _released(key, RESUMED)
                told = True
                return RESUMED, hold.get("value") or ""
            if timeout and (time.monotonic() - hold["started"]) >= timeout:
                logger.info("%s waited %.0fs and carried on", key, timeout)
                _released(key, TIMED_OUT)
                told = True
                return TIMED_OUT, ""
            time.sleep(TICK)
    except comfy.model_management.InterruptProcessingException:
        logger.info("%s was cancelled, so the hold ended and the run stopped", key)
        raise
    finally:
        with _lock:
            _holds.pop(key, None)
        # Whatever ended the hold, the browser hears about it once. A panel left holding its
        # controls is the one failure the person watching cannot tell from a working one.
        if not told:
            _released(key, CANCELLED)


def release(node_id: str, action: str, value: str = "") -> bool:
    """Let a held node carry on.

    Args:
        node_id: The node to release.
        action: ``"resumed"`` or ``"cancelled"``.
        value: What the user edited, for a node that asked for one.

    Returns:
        True when a node was waiting under that id.
    """
    with _lock:
        hold = _holds.get(str(node_id))
        if hold is None:
            return False
        hold["value"] = value
        hold["action"] = action
    return True


def register_routes() -> bool:
    """Register the route the browser resumes a held run through.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a Pause node waits out its timeout.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        from .channel import NO_STORE

        @PromptServer.instance.routes.get(ROUTE)
        async def get_pause(request):
            return web.json_response({"waiting": waiting()}, headers=NO_STORE)

        @PromptServer.instance.routes.post(ROUTE)
        async def post_pause(request):
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"released": False, "error": "unreadable body"},
                                         status=400, headers=NO_STORE)
            action = CANCELLED if body.get("action") == "cancel" else RESUMED
            released = release(body.get("node_id", ""), action, str(body.get("value") or ""))
            return web.json_response({"released": released, "action": action},
                                     headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a Pause node cannot be resumed from the "
            "browser and waits out its timeout",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is releasing held runs", ROUTE)
    return True
