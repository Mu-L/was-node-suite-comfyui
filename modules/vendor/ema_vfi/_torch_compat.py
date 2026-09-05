"""The three ``timm.models.layers`` names EMA-VFI imports, built from torch alone.

timm is not installed in the environments this pack targets and is not worth a dependency for
three helpers. None of them holds a parameter, so swapping them in leaves every key in the
released checkpoint exactly where it was.
"""

from __future__ import annotations

import collections.abc

from torch import nn

__all__ = ["DropPath", "to_2tuple", "trunc_normal_"]

# torch's own initialiser takes the same arguments in the same order as timm's, so the name is
# simply bound rather than reimplemented.
trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(value):
    """``value`` as a pair, passing an existing pair straight through."""
    if isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
        return tuple(value)
    return (value, value)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop whole samples from the batch, the residual-branch form of stochastic depth.

    Args:
        x: The branch's output.
        drop_prob: Chance of dropping a sample.
        training: Whether to drop at all. Inference keeps everything.
        scale_by_keep: Whether survivors are scaled up to hold the expected value.

    Returns:
        ``x`` unchanged during inference, otherwise with whole samples zeroed.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # One value per sample, broadcast over whatever axes the tensor happens to have.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    kept = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        kept.div_(keep_prob)
    return x * kept


class DropPath(nn.Module):
    """Stochastic depth over a residual branch, holding no parameters of its own."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x):
        """Drop samples while training, pass everything through otherwise."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        """The drop chance, so a printed model shows it."""
        return f"drop_prob={self.drop_prob:0.3f}"
