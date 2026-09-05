"""Fusing several LoRAs into one state dict before any of them is applied.

:func:`fuse_state_dicts` joins matched up and down factors along the rank axis, which is
exact rather than an approximation. Keys it cannot pair are carried through scaled.
"""

from __future__ import annotations

import torch

from ..log import get_logger

__all__ = ["DOWN_SUFFIXES", "UP_SUFFIXES", "fuse_state_dicts"]

logger = get_logger("lora.fuse")

#: Suffixes naming the two factors of a plain LoRA, in the spellings that reach a state dict.
DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight", ".lora_down")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".lora_up")


def _module_of(key: str) -> tuple[str, str] | None:
    """Split a factor key into the module it belongs to and which factor it is.

    Args:
        key: A state dict key.

    Returns:
        ``(module, "down")`` or ``(module, "up")``, and None for a key that is neither.
    """
    for suffix in DOWN_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "down"
    for suffix in UP_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "up"
    return None


def _scale_of(state: dict, module: str, rank: int) -> float:
    """The multiplier a module's alpha asks for.

    Args:
        state: The LoRA's state dict.
        module: Module prefix, as :func:`_module_of` answers it.
        rank: Rank of that module's factors.

    Returns:
        ``alpha / rank``, and 1.0 where no alpha is recorded, which is how a LoRA saved
        without one is already scaled.
    """
    alpha = state.get(f"{module}.alpha")
    if alpha is None or not rank:
        return 1.0
    try:
        return float(alpha.item() if hasattr(alpha, "item") else alpha) / float(rank)
    except (TypeError, ValueError):
        return 1.0


def fuse_state_dicts(loaded: list[tuple[dict, float]]) -> tuple[dict, int, int]:
    """Combine several weighted LoRA state dicts into one.

    Args:
        loaded: ``(state_dict, weight)`` per LoRA, in the order they were listed.

    Returns:
        ``(fused, paired, carried)``: the combined state dict, how many modules were fused
        by rank concatenation, and how many keys were carried through as something other
        than a plain up and down pair. The result is applied at strength 1.0, since every
        weight is already folded into it.
    """
    modules: dict[str, dict[str, list]] = {}
    carried: dict[str, torch.Tensor] = {}

    for state, weight in loaded:
        seen: set[str] = set()
        for key, value in state.items():
            split = _module_of(key)
            if split is None:
                continue
            module, side = split
            seen.add(module)
            modules.setdefault(module, {"down": [], "up": []})
            modules[module][side].append((value, weight, state, module))

        for key, value in state.items():
            split = _module_of(key)
            if split is not None or key.endswith(".alpha"):
                continue
            module = key.rsplit(".", 1)[0]
            if module in seen:
                continue
            # Not a plain pair: a LoHa factor, a bias, a norm. Scaled and kept, so the LoRA
            # still contributes rather than being dropped from the fuse. Summed where two
            # of them carry the same key, since one overwriting the other would silently
            # drop whichever was listed first.
            if not value.is_floating_point():
                carried.setdefault(key, value)
                continue
            scaled = value.to(torch.float32) * weight
            carried[key] = carried[key] + scaled if key in carried else scaled

    fused: dict[str, torch.Tensor] = dict(carried)
    paired = 0
    for module, sides in modules.items():
        downs, ups = sides["down"], sides["up"]
        if len(downs) != len(ups) or not downs:
            for value, weight, _state, _module in downs + ups:
                logger.debug("%s has no matching pair and was left out of the fuse", module)
            continue

        down_parts, up_parts = [], []
        for (down, weight, state, name), (up, _w, _s, _n) in zip(downs, ups):
            rank = int(down.shape[0])
            scale = _scale_of(state, name, rank) * float(weight)
            down_parts.append(down.to(torch.float32))
            up_parts.append(up.to(torch.float32) * scale)

        try:
            down_cat = torch.cat(down_parts, dim=0)
            up_cat = torch.cat(up_parts, dim=1)
        except RuntimeError as error:
            logger.debug("%s could not be concatenated (%s), left out of the fuse", module, error)
            continue

        # Kept in float32 rather than the source dtype: the concatenated factors carry every
        # row's contribution, and rounding them back to a half float loses precision that
        # applying the rows one at a time never spends.
        fused[f"{module}.lora_down.weight"] = down_cat
        fused[f"{module}.lora_up.weight"] = up_cat
        # The rank is the sum of the parts and every scale is already folded into up, so an
        # alpha equal to the new rank leaves the applied delta exactly as it was summed.
        fused[f"{module}.alpha"] = torch.tensor(float(down_cat.shape[0]), dtype=torch.float32)
        paired += 1

    return fused, paired, len(carried)
