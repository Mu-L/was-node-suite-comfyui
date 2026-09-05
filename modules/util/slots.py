"""Reading a repeated slot mapping back in the order its sockets are drawn.

A ``**kwargs`` capture and an ``io.Autogrow`` group both reach ``execute()`` as a mapping
keyed by slot name, which carries no order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = ["connected_in_order"]


def connected_in_order(values: Mapping | None, names: Iterable[str]) -> list[str]:
    """The slot names holding a value, in the order the sockets are declared.

    Args:
        values: The slot mapping, or ``None`` where nothing is connected at all.
        names: Every slot name declared, in drawn order.

    Returns:
        The names holding something, in ``names`` order rather than alphabetical order. A key
        absent from ``names`` is dropped.
    """
    position = {name: index for index, name in enumerate(names)}
    filled = [
        name for name, value in (values or {}).items()
        if value is not None and name in position
    ]
    return sorted(filled, key=lambda name: position[name])
