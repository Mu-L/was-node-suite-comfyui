"""Helpers for ``LIST`` sockets and for outputs declared ``is_output_list=True``.

:func:`require_values` and :func:`block_if_empty` are the two ways a node answers for an
empty list slot.
"""

from __future__ import annotations

from typing import Any

from comfy_execution.graph_utils import ExecutionBlocker

__all__ = ["as_list", "block_if_empty", "require_values"]


def as_list(value: Any) -> list:
    """Read a ``LIST`` socket's value as a python list.

    Args:
        value: Whatever arrived on the socket. A list or tuple is copied; ``None`` gives an
            empty list; anything else is wrapped as a single item.

    Returns:
        A new list, safe to mutate.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def require_values(values: list, message: str) -> list:
    """Hand back the values, or raise ``message`` when there are none.

    For a node whose list output is the whole point of it.

    Args:
        values: What the list slot would emit.
        message: The exception text.

    Returns:
        ``values``, unchanged.

    Raises:
        ValueError: ``values`` is empty.
    """
    if not values:
        raise ValueError(message)
    return values


def block_if_empty(values: list, message: str):
    """Hand back the values, or a blocker carrying ``message`` when there are none.

    Args:
        values: What the list slot would emit.
        message: Shown as ``Execution Blocked: <message>`` on every node reading the slot.

    Returns:
        ``values`` when it holds anything, otherwise an ``ExecutionBlocker``. The return
        goes straight into the ``io.NodeOutput`` slot it belongs to.
    """
    return values if values else ExecutionBlocker(message)
