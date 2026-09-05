"""The Swin v1 transformer backbone, at the width the caller's checkpoint carries.

:class:`SwinTransformer` reads ``(batch, 3, height, width)`` and answers the patch
embedding beside one feature map per stage.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    "DEPTHS",
    "EMBED_DIM_BASE",
    "EMBED_DIM_LARGE",
    "HEADS_BASE",
    "HEADS_LARGE",
    "MLP_RATIO",
    "PATCH_SIZE",
    "SwinTransformer",
    "WINDOW_SIZE",
]

#: Side of the square window attention is computed inside, in patch positions.
WINDOW_SIZE = 12

#: Blocks each of the four stages stacks.
DEPTHS = (2, 2, 18, 2)

#: Patch embedding width of the base and of the large variant.
EMBED_DIM_BASE = 128
EMBED_DIM_LARGE = 192

#: Attention heads each stage splits its channels across, base then large.
HEADS_BASE = (4, 8, 16, 32)
HEADS_LARGE = (6, 12, 24, 48)

#: Side of one patch, in pixels, and the stride the first stage reads at.
PATCH_SIZE = 4

#: Multiplier from a block's channels to the width of its feed-forward layer.
MLP_RATIO = 4


def _partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """Cut a frame into square windows, stacked along the batch axis.

    Args:
        x: ``(batch, height, width, channels)`` tensor, both sides a multiple of ``window``.
        window: Side of one square window.

    Returns:
        A ``(batch * windows, window, window, channels)`` tensor.
    """
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def _merge(windows: torch.Tensor, window: int, height: int, width: int) -> torch.Tensor:
    """Lay square windows back out as a frame.

    Args:
        windows: ``(batch * windows, window, window, channels)`` tensor.
        window: Side of one square window.
        height: Height of the frame the windows were cut from.
        width: Width of the frame the windows were cut from.

    Returns:
        A ``(batch, height, width, channels)`` tensor.
    """
    batch = windows.shape[0] // (height * width // window // window)
    x = windows.view(batch, height // window, width // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class FeedForward(nn.Module):
    """Two linear layers around a GELU, widened by :data:`MLP_RATIO` between them."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix each token across its channels.

        Args:
            x: ``(batch, tokens, dim)`` tensor.

        Returns:
            A ``(batch, tokens, dim)`` tensor.
        """
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Self-attention inside one square window, biased by the relative position of cells."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        span = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(torch.zeros(span * span, num_heads))
        rows, columns = torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        )
        cells = torch.stack((rows, columns)).flatten(1)
        offsets = (cells[:, :, None] - cells[:, None, :]).permute(1, 2, 0).contiguous()
        offsets[:, :, 0] += window_size - 1
        offsets[:, :, 1] += window_size - 1
        offsets[:, :, 0] *= span
        # Row of the bias table every ordered pair of cells in a window reads.
        self.register_buffer("relative_position_index", offsets.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attend cell against cell within each window.

        Args:
            x: ``(windows, tokens, dim)`` tensor, one row per cell of one window.
            mask: ``(windows per frame, tokens, tokens)`` additive mask, or None.

        Returns:
            A ``(windows, tokens, dim)`` tensor.
        """
        total, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(total, tokens, 3, self.num_heads, channels // self.num_heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        scores = scores + bias.view(tokens, tokens, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            count = mask.shape[0]
            scores = scores.view(total // count, count, self.num_heads, tokens, tokens)
            scores = scores + mask.unsqueeze(1).unsqueeze(0)
            scores = scores.view(-1, self.num_heads, tokens, tokens)
        attended = scores.softmax(dim=-1) @ value
        return self.proj(attended.transpose(1, 2).reshape(total, tokens, channels))


class Block(nn.Module):
    """One transformer block: residual window attention, then a residual feed-forward."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * MLP_RATIO))

    def forward(
        self, x: torch.Tensor, height: int, width: int, mask: torch.Tensor
    ) -> torch.Tensor:
        """Attend within each window, rolling the frame first where the block is shifted.

        Args:
            x: ``(batch, height * width, dim)`` tensor.
            height: Rows the tokens lay out as.
            width: Columns the tokens lay out as.
            mask: ``(windows, tokens, tokens)`` additive mask for the rolled frame.

        Returns:
            A ``(batch, height * width, dim)`` tensor.
        """
        batch, _, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        x = functional.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attended = self.attn(windows, mask if self.shift_size > 0 else None)
        attended = attended.view(-1, self.window_size, self.window_size, channels)
        x = _merge(attended, self.window_size, padded_height, padded_width)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        if pad_right > 0 or pad_bottom > 0:
            x = x[:, :height, :width, :].contiguous()
        x = shortcut + x.view(batch, height * width, channels)
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """A halving: the four cells of each 2x2 square joined by channel and projected."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Halve both sides and double the channels.

        Args:
            x: ``(batch, height * width, dim)`` tensor.
            height: Rows the tokens lay out as.
            width: Columns the tokens lay out as.

        Returns:
            A ``(batch, ceil(height / 2) * ceil(width / 2), dim * 2)`` tensor.
        """
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)
        if height % 2 or width % 2:
            x = functional.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        quads = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1
        )
        return self.reduction(self.norm(quads.view(batch, -1, 4 * channels)))


class Stage(nn.Module):
    """One scale: blocks alternating plain and shifted windows, then a halving."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, merge: bool):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            Block(dim, num_heads, window_size, 0 if index % 2 == 0 else window_size // 2)
            for index in range(depth)
        )
        self.downsample = PatchMerging(dim) if merge else None

    def forward(
        self, x: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Run every block at this scale and halve what the next scale reads.

        Args:
            x: ``(batch, height * width, dim)`` tensor.
            height: Rows the tokens lay out as.
            width: Columns the tokens lay out as.

        Returns:
            ``(out, down, height, width)``: this scale's tokens, the tokens the next scale
            reads, and the rows and the columns those lay out as.
        """
        mask = self._mask(height, width, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, height, width, mask)
        if self.downsample is None:
            return x, x, height, width
        return x, self.downsample(x, height, width), (height + 1) // 2, (width + 1) // 2

    def _mask(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Answer the additive mask hiding the pairs of cells a roll brought together.

        Args:
            height: Rows the tokens lay out as.
            width: Columns the tokens lay out as.
            device: Where to build the mask.
            dtype: Type to answer in.

        Returns:
            A ``(windows, tokens, tokens)`` tensor, -100 on each pair to be hidden.
        """
        window = self.window_size
        # Round both sides up to a whole window.
        padded_height = -(-height // window) * window
        padded_width = -(-width // window) * window
        regions = torch.zeros((1, padded_height, padded_width, 1), device=device)
        spans = (
            slice(0, -window),
            slice(-window, -self.shift_size),
            slice(-self.shift_size, None),
        )
        index = 0
        for rows in spans:
            for columns in spans:
                regions[:, rows, columns, :] = index
                index += 1
        cells = _partition(regions, window).view(-1, window * window)
        mask = cells.unsqueeze(1) - cells.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0).to(dtype)


class PatchEmbed(nn.Module):
    """A strided convolution cutting a frame into patches, normalised across its channels."""

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed each patch of a frame, padding both sides up to a whole patch first.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            A ``(batch, embed_dim, ceil(height / patch), ceil(width / patch))`` tensor.
        """
        height, width = x.shape[2], x.shape[3]
        pad_right = (self.patch_size - width % self.patch_size) % self.patch_size
        pad_bottom = (self.patch_size - height % self.patch_size) % self.patch_size
        x = self.proj(functional.pad(x, (0, pad_right, 0, pad_bottom)))
        batch, channels, rows, columns = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(batch, channels, rows, columns)


class SwinTransformer(nn.Module):
    """The Swin v1 backbone: a patch embedding and four stages of windowed attention."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM_BASE,
        depths: tuple[int, ...] = DEPTHS,
        num_heads: tuple[int, ...] = HEADS_BASE,
        window_size: int = WINDOW_SIZE,
        in_channels: int = 3,
        patch_size: int = PATCH_SIZE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        #: Channels each stage answers, finest first.
        self.num_features = [embed_dim * 2 ** index for index in range(self.num_layers)]
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        self.layers = nn.ModuleList(
            Stage(
                self.num_features[index],
                depths[index],
                num_heads[index],
                window_size,
                index < self.num_layers - 1,
            )
            for index in range(self.num_layers)
        )
        for index in range(self.num_layers):
            self.add_module(f"norm{index}", nn.LayerNorm(self.num_features[index]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Read a frame and answer the feature map every scale saw it as.

        Args:
            x: ``(batch, in_channels, height, width)`` tensor.

        Returns:
            The patch embedding at a quarter of each side, then one map per stage, at a
            quarter, an eighth, a sixteenth and a thirty-second of each side, doubling
            their channels as they go.
        """
        x = self.patch_embed(x)
        maps = [x.contiguous()]
        height, width = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        for index, stage in enumerate(self.layers):
            out, x, next_height, next_width = stage(x, height, width)
            out = getattr(self, f"norm{index}")(out)
            out = out.view(-1, height, width, self.num_features[index])
            maps.append(out.permute(0, 3, 1, 2).contiguous())
            height, width = next_height, next_width
        return tuple(maps)
