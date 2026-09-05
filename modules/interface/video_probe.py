"""What a video measures, as JSON a node interface draws its timeline against.

``GET /was/interface/api/video_probe?file=<name>`` answers the rate, the frame count, the
duration, the size and the bit depth of one file in ComfyUI's input folder.
"""

from __future__ import annotations

from .. import log
from .channel import NO_STORE

__all__ = ["ROUTE", "probe_payload", "register_routes"]

logger = log.get_logger("interface.video_probe")

#: The one route serving the measurement.
ROUTE = "/was/interface/api/video_probe"

_registered = False


def probe_payload(name: str) -> dict:
    """What the named video measures.

    Never raises.

    Args:
        name: The file as the widget holds it, inside ComfyUI's input folder.

    Returns:
        ``{"fps", "frame_count", "duration", "width", "height", "bit_depth", "read"}``.
        ``read`` is False
        where nothing could be measured.
    """
    # A file that cannot be read answers zeroes rather than an error, which leaves the player
    # on its own measurement rather than on nothing at all.
    empty = {
        "fps": 0.0, "frame_count": 0, "duration": 0.0, "width": 0, "height": 0,
        "bit_depth": 0, "read": False,
    }
    if not name:
        return empty
    try:
        from ..media import reader

        found = reader.probe(reader.input_path(name))
    except Exception as error:
        logger.debug("%s could not measure %r (%s)", ROUTE, name, error)
        return empty
    return {
        "fps": float(found.fps),
        "frame_count": int(found.frame_count),
        "duration": float(found.duration),
        "width": int(found.width),
        "height": int(found.height),
        "bit_depth": int(found.bit_depth),
        "read": True,
    }


def register_routes() -> bool:
    """Register the route serving what a video measures.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a player asking for the measurement gets a
        failed request and counts frames from what it can present.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_video_probe(request):
            return web.json_response(
                probe_payload(request.query.get("file", "")), headers=NO_STORE
            )

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a video player asking what a clip measures "
            "gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the rate and frame count of input videos", ROUTE)
    return True
