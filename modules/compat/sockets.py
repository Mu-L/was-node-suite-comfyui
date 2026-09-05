"""What a node's own sockets are worth while it runs.

:func:`require_input` raises when a required socket arrived as ``None``, and
:func:`input_source` answers which of a widget and a link filled an input on this run.
"""

from __future__ import annotations

from typing import Any

from .. import log

__all__ = ["LINK", "WIDGET", "input_source", "require_input"]

logger = log.get_logger("compat.sockets")

#: An input another node's output is wired into on this run.
LINK = "link"

#: An input carrying what its own widget holds on this run.
WIDGET = "widget"


def _article(word: str) -> str:
    """``an`` before a word starting with a vowel letter, ``a`` before anything else.

    Args:
        word: The phrase the article precedes.

    Returns:
        ``a`` or ``an``.
    """
    return "an" if word[:1].lower() in "aeiou" else "a"


def require_input(
    value: Any,
    node: str,
    socket: str,
    thing: str,
    source: str,
    source_output: str | None = None,
) -> Any:
    """Hand back a socket's value, or raise naming what to connect to it.

    Args:
        value: Whatever arrived on the socket.
        node: The node's display name, spelled as its title bar spells it.
        socket: The input's name, spelled as the schema spells it.
        thing: What the socket wants, in a user's words: ``detector``, ``VAE``, ``latent``.
        source: The node to add, or a phrase naming the choices.
        source_output: Name of that node's output. Defaults to ``socket``.

    Returns:
        ``value``, unchanged.

    Raises:
        ValueError: ``value`` is ``None``.
    """
    if value is None:
        raise ValueError(
            f"{node} has no {thing} on its {socket} input. Add {_article(source)} {source} "
            f"and wire its {source_output or socket} output into this node."
        )
    return value


def input_source(name: str, node_id: str | None = None) -> str | None:
    """Which of a widget and a link filled one input of the running node.

    Never raises. The prompt is read, never changed.

    Args:
        name: The input's name, spelled as the schema spells it.
        node_id: The node's graph id. Left out, the id of the node ComfyUI is executing is
            read from its execution context.

    Returns:
        :data:`LINK` when the running prompt fills that input from another node's output,
        :data:`WIDGET` when it carries the value the widget holds, or None when the server,
        the prompt, the node or the input could not be read. A body called directly rather
        than through a prompt answers None, and so does an input no prompt declares.
    """
    try:
        from comfy_execution.utils import get_executing_context
        from server import PromptServer

        context = get_executing_context()
        key = str(node_id) if node_id is not None else getattr(context, "node_id", None)
        prompt_id = getattr(context, "prompt_id", None)
        if not key or not prompt_id:
            return None
        # The queue holds the prompt as it was queued, which is the one this run was given.
        # A node inside a subgraph is keyed there by the same colon joined path the
        # execution context carries.
        for item in list(PromptServer.instance.prompt_queue.currently_running.values()):
            if item[1] != prompt_id:
                continue
            value = item[2][key]["inputs"][name]
            # A link reaches the backend as [source node id, output slot] and a widget value
            # as the number, string or boolean the widget holds.
            return LINK if isinstance(value, (list, tuple)) else WIDGET
    except Exception as error:
        logger.debug("the source of input %s could not be read (%s)", name, error)
    return None
