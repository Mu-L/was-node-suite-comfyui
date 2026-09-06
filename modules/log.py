"""Logging for the pack, with optional rich formatting.

:func:`get_logger` returns a logger under the pack's root name. Level and colour come from
the ``logging`` block of the config.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "was_node_suite"
PREFIX = "[WAS Node Suite] "

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

console = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger for a submodule, e.g. ``get_logger("image.filters")``."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def rich_handler():
    """Build the console and handler rich output is rendered through.

    Returns:
        ``(Console, Handler)``. The record payload is taken literally: rich markup is not
        parsed, so ``Console`` link markup has no effect, while the ``WAS Node Suite``
        prefix keeps its colour as a ``Text`` span and ``ReprHighlighter`` still styles
        the message.
    """
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.text import Text

    class WASRichHandler(RichHandler):
        """A ``RichHandler`` that prefixes every record and never raises out of ``emit``."""

        def emit(self, record: logging.LogRecord) -> None:
            # Rich guards only its own console.print; format() and render_message() run
            # bare, so a record rich dislikes would escape logger.error() itself. Such a
            # record takes the route the stdlib gives a broken StreamHandler.
            try:
                super().emit(record)
            except Exception:
                self.handleError(record)

        def render_message(self, record: logging.LogRecord, message: str):
            return Text.assemble(
                ("WAS Node Suite", "bold cyan"), " ", super().render_message(record, message)
            )

    console = Console(force_terminal=True, no_color=False)
    # markup=False is load bearing: rich parses the formatted record, text the pack did not
    # author, so "cannot open [/opt/ffmpeg]" raises MarkupError and "[rembg]" is dropped as
    # an open tag. ReprHighlighter is unaffected, it only maps regex spans onto styles.
    handler = WASRichHandler(
        console=console,
        markup=False,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return console, handler


def configure(level: str = "info", rich: bool = True) -> None:
    """Point the pack's logger at a single handler. Safe to call more than once.

    Args:
        level: One of the keys of :data:`LEVELS`; anything else reads as ``"info"``.
        rich: Render through a rich handler, falling back to a plain stream handler when
            rich is unavailable.
    """
    global console

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LEVELS.get(str(level).lower(), logging.INFO))
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    console = None
    handler = None
    if rich:
        try:
            console, handler = rich_handler()
        except Exception:
            console = None
            handler = None
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(PREFIX + "%(message)s"))
    logger.addHandler(handler)
