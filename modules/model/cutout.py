"""The cutout models, behind one name each.

:func:`load` answers a :class:`Cutout` carrying the network beside the side it reads at.
:data:`MODELS` lists every option, across BiRefNet and BEN2.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ben2, birefnet

__all__ = ["Cutout", "MODELS", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: Widget option -> the module answering it.
MODELS = {name: "birefnet" for name in birefnet.MODELS}
MODELS["BEN2"] = "ben2"


@dataclass(frozen=True)
class Cutout:
    """A built cutout network and what a caller must know to drive it.

    Attributes:
        backend: The ``Backend`` holding the network.
        name: Widget option the network was built from.
        family: Module it came from, ``birefnet`` or ``ben2``.
        side: Square side the frame is read at.
    """

    backend: object
    name: str
    family: str
    side: int


def load(model: str = "BiRefNet General") -> Cutout:
    """Build or return the cached cutout network for one option.

    Args:
        model: A key of :data:`MODELS`.

    Returns:
        A :class:`Cutout`, its weights resting until ``backend.load()`` is called.

    Raises:
        ValueError: ``model`` names nothing this node runs.
        ModelUnavailable: The weights are absent and ``features.network`` is off.
    """
    family = MODELS.get(model)
    if family is None:
        raise ValueError(
            f"Cutout model must be one of {', '.join(MODELS)}, not {model!r}"
        )
    if family == "ben2":
        return Cutout(ben2.load(), model, family, ben2.TRAINED_SIDE)
    return Cutout(birefnet.load(model), model, family, birefnet.RESOLUTIONS[model])
