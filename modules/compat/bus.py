"""The ``BUS`` payload, and the named extras a dynamic bus carries with it.

A bus is a five-tuple in :data:`BUS_MEMBERS` order. :class:`DynamicBus` subclasses
``tuple`` and carries an ``extras`` mapping alongside.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BUS_MEMBERS", "DynamicBus", "extras_of", "members_of"]

#: The five members every bus carries, in wire order.
BUS_MEMBERS = ("model", "clip", "vae", "positive", "negative")


class DynamicBus(tuple):
    """A five-member bus that also carries named extras."""

    def __new__(cls, members, extras: dict[str, Any] | None = None):
        """Build the bus.

        Args:
            members: The five values, in :data:`BUS_MEMBERS` order. Padded with ``None``
                and truncated to five, so a shorter or longer sequence cannot produce a bus
                that unpacks to the wrong number of names.
            extras: Named values to carry alongside. Copied, so a later edit to the caller's
                dictionary does not reach a bus already on the wire.
        """
        values = list(members)[: len(BUS_MEMBERS)]
        values += [None] * (len(BUS_MEMBERS) - len(values))
        instance = super().__new__(cls, values)
        instance.extras = dict(extras or {})
        return instance

    def __repr__(self) -> str:
        return f"DynamicBus({tuple.__repr__(self)}, extras={sorted(self.extras)})"


def extras_of(bus) -> dict[str, Any]:
    """The extra values a bus carries.

    Args:
        bus: A bus from either bus node, or ``None``.

    Returns:
        The extras, or an empty mapping for a plain five-tuple bus and for ``None``. The
        result is a copy: mutating it does not change the bus it came from.
    """
    return dict(getattr(bus, "extras", None) or {})


def members_of(bus) -> list[Any]:
    """The five standard members of a bus.

    Args:
        bus: A bus from either bus node, or ``None``.

    Returns:
        Five values in :data:`BUS_MEMBERS` order. A bus that is ``None``, or that holds
        fewer than five values, is padded with ``None`` rather than raising.
    """
    values = list(bus) if bus else []
    values = values[: len(BUS_MEMBERS)]
    return values + [None] * (len(BUS_MEMBERS) - len(values))
