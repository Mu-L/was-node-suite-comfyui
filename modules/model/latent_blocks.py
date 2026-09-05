"""Building blocks shared by the latent diffusion UNet and autoencoder.

Parameter names follow the published checkpoints. Tensors are
``(batch, channels, height, width)`` unless a docstring says otherwise.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

__all__ = [
    "HEAD_CHANNELS",
    "MAX_PERIOD",
    "NORM_GROUPS",
    "Attention",
    "Downsample",
    "FeedForward",
    "GatedProjection",
    "ResnetBlock",
    "SpatialAttention",
    "SpatialTransformer",
    "TimestepEmbedding",
    "TransformerBlock",
    "Upsample",
    "group_norm",
    "timestep_embedding",
]

#: Groups every normalisation in these networks splits its channels into.
NORM_GROUPS = 32

#: Channels one attention head reads.
HEAD_CHANNELS = 64

#: Period of the slowest sinusoid in a timestep embedding.
MAX_PERIOD = 10000


def group_norm(channels: int, eps: float) -> nn.GroupNorm:
    """A group normalisation over :data:`NORM_GROUPS` groups.

    Args:
        channels: Channels arriving, a multiple of :data:`NORM_GROUPS`.
        eps: Value added to the variance before the square root.

    Returns:
        The layer.
    """
    return nn.GroupNorm(num_groups=NORM_GROUPS, num_channels=channels, eps=eps, affine=True)


def timestep_embedding(steps: torch.Tensor, channels: int) -> torch.Tensor:
    """Sinusoids encoding a diffusion timestep, cosines first.

    Args:
        steps: ``(batch,)`` timesteps, whole or fractional.
        channels: Width of the embedding, an even number.

    Returns:
        A ``(batch, channels)`` float32 tensor.
    """
    half = channels // 2
    exponent = -math.log(MAX_PERIOD) * torch.arange(
        start=0, end=half, dtype=torch.float32, device=steps.device
    )
    frequencies = torch.exp(exponent / half)
    angles = steps[:, None].float() * frequencies[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class TimestepEmbedding(nn.Module):
    """The two layer projection a timestep embedding passes through.

    Attributes:
        linear_1: Projection onto the wide embedding.
        linear_2: Projection the residual blocks read.
    """

    def __init__(self, in_channels: int, embed_channels: int):
        """Build the projection.

        Args:
            in_channels: Width of the sinusoid embedding arriving.
            embed_channels: Width leaving, which is what every residual block reads.
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embed_channels)
        self.linear_2 = nn.Linear(embed_channels, embed_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the projection of ``x``."""
        return self.linear_2(functional.silu(self.linear_1(x)))


class Downsample(nn.Module):
    """A strided convolution halving both sides.

    Attributes:
        conv: The stride 2 convolution.
        padding: Padding the convolution applies, 0 for a pad of one sample below and right.
    """

    def __init__(self, channels: int, out_channels: int | None = None, padding: int = 1):
        """Build the convolution.

        Args:
            channels: Channels arriving.
            out_channels: Channels leaving, or ``None`` for ``channels``.
            padding: Symmetric padding the convolution applies. 0 pads one sample below and
                to the right instead.
        """
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(
            channels, out_channels or channels, kernel_size=3, stride=2, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer ``x`` at half the height and width."""
        if self.padding == 0:
            x = functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    """A nearest neighbour doubling followed by a convolution.

    Attributes:
        conv: The 3x3 convolution applied after the resize.
    """

    def __init__(self, channels: int, out_channels: int | None = None):
        """Build the convolution.

        Args:
            channels: Channels arriving.
            out_channels: Channels leaving, or ``None`` for ``channels``.
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, out_channels or channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        """Answer ``x`` resized then convolved.

        Args:
            x: The activation.
            size: Height and width to resize to, or ``None`` to double both sides.

        Returns:
            The convolved result.
        """
        if size is None:
            x = functional.interpolate(x, scale_factor=2.0, mode="nearest")
        else:
            x = functional.interpolate(x, size=size, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Two convolutions around a normalisation, with an optional timestep term.

    Attributes:
        time_emb_proj: Projection of the timestep embedding onto the output channels, or
            ``None`` for a block that takes no timestep.
        conv_shortcut: 1x1 convolution carrying the input when the channel count changes,
            or ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_channels: int | None = None,
        eps: float = 1e-5,
    ):
        """Build the block.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels leaving.
            embed_channels: Width of the timestep embedding, or ``None`` for no timestep.
            eps: Value added to the variance in both normalisations.
        """
        super().__init__()
        self.norm1 = group_norm(in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = (
            nn.Linear(embed_channels, out_channels) if embed_channels else None
        )
        self.norm2 = group_norm(out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Answer the block applied to ``x``.

        Args:
            x: ``(batch, in_channels, height, width)``.
            embedding: ``(batch, embed_channels)`` timestep embedding, or ``None``.

        Returns:
            ``(batch, out_channels, height, width)``.
        """
        hidden = self.conv1(functional.silu(self.norm1(x)))
        if self.time_emb_proj is not None and embedding is not None:
            hidden = hidden + self.time_emb_proj(functional.silu(embedding))[:, :, None, None]
        hidden = self.conv2(functional.silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden


class Attention(nn.Module):
    """Multi-head attention over a sequence, optionally reading a second sequence.

    Attributes:
        heads: Number of attention heads.
        to_out: The output projection, as a list whose second entry is a placeholder.
    """

    def __init__(self, channels: int, context_channels: int | None = None, heads: int = 8):
        """Build the projections.

        Args:
            channels: Width of the queries and of the result.
            context_channels: Width of the keys and values, or ``None`` for ``channels``.
            heads: Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        context_channels = channels if context_channels is None else context_channels
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_channels, channels, bias=False)
        self.to_v = nn.Linear(context_channels, channels, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Identity()])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Answer attention over ``x``.

        Args:
            x: ``(batch, tokens, channels)``.
            context: ``(batch, context_tokens, context_channels)``, or ``None`` to attend
                to ``x`` itself.

        Returns:
            ``(batch, tokens, channels)``.
        """
        source = x if context is None else context
        batch = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(source)
        value = self.to_v(source)
        width = key.shape[-1] // self.heads
        query = query.view(batch, -1, self.heads, width).transpose(1, 2)
        key = key.view(batch, -1, self.heads, width).transpose(1, 2)
        value = value.view(batch, -1, self.heads, width).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, -1, self.heads * width)
        return self.to_out[0](attended.to(query.dtype))


class SpatialAttention(nn.Module):
    """Single head attention over every sample of a feature map, added to its input.

    Attributes:
        group_norm: Normalisation applied before the projections.
        to_out: The output projection, as a list whose second entry is a placeholder.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        """Build the projections.

        Args:
            channels: Channels arriving, which is also the head width.
            eps: Value added to the variance in the normalisation.
        """
        super().__init__()
        self.group_norm = group_norm(channels, eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer ``x`` plus attention over its samples."""
        batch, channels, height, width = x.shape
        tokens = x.view(batch, channels, height * width).transpose(1, 2)
        tokens = self.group_norm(tokens.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(tokens).view(batch, -1, 1, channels).transpose(1, 2)
        key = self.to_k(tokens).view(batch, -1, 1, channels).transpose(1, 2)
        value = self.to_v(tokens).view(batch, -1, 1, channels).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, -1, channels)
        attended = self.to_out[0](attended.to(query.dtype))
        return attended.transpose(-1, -2).reshape(batch, channels, height, width) + x


class GatedProjection(nn.Module):
    """A projection to twice a width, half of it gating the other half.

    Attributes:
        proj: The projection onto both halves.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Build the projection.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels leaving, which is half the projection width.
        """
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the gated half of the projection of ``x``."""
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * functional.gelu(gate)


class FeedForward(nn.Module):
    """A gated linear unit widening by four, then a projection back.

    Attributes:
        net: The gated projection, a placeholder, and the projection back.
    """

    def __init__(self, channels: int):
        """Build the pair.

        Args:
            channels: Channels arriving and leaving.
        """
        super().__init__()
        inner = channels * 4
        self.net = nn.ModuleList(
            [GatedProjection(channels, inner), nn.Identity(), nn.Linear(inner, channels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the gated projection of ``x``."""
        return self.net[2](self.net[0](x))


class TransformerBlock(nn.Module):
    """Self attention, cross attention and a feed forward, each on a residual.

    Attributes:
        attn1: Self attention over the feature map.
        attn2: Cross attention onto the prompt embedding.
    """

    def __init__(self, channels: int, context_channels: int, heads: int):
        """Build the three stages.

        Args:
            channels: Channels arriving and leaving.
            context_channels: Width of the embedding the cross attention reads.
            heads: Number of attention heads in both attentions.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = Attention(channels, heads=heads)
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = Attention(channels, context_channels, heads=heads)
        self.norm3 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Answer the block applied to ``x`` against ``context``.

        Args:
            x: ``(batch, tokens, channels)``.
            context: ``(batch, context_tokens, context_channels)``.

        Returns:
            ``(batch, tokens, channels)``.
        """
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    """Transformer blocks run over a feature map read as a sequence of samples.

    Attributes:
        transformer_blocks: The blocks, run in order.
    """

    def __init__(self, channels: int, context_channels: int, depth: int = 1):
        """Build the transformer.

        Args:
            channels: Channels arriving and leaving.
            context_channels: Width of the embedding the cross attention reads.
            depth: Number of transformer blocks.
        """
        super().__init__()
        heads = channels // HEAD_CHANNELS
        self.norm = group_norm(channels, 1e-6)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(channels, context_channels, heads) for _ in range(depth)
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Answer ``x`` plus the transformer applied to it.

        Args:
            x: ``(batch, channels, height, width)``.
            context: ``(batch, context_tokens, context_channels)``.

        Returns:
            ``(batch, channels, height, width)``.
        """
        residual = x
        batch, channels, height, width = x.shape
        tokens = self.norm(x).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        tokens = self.proj_in(tokens)
        for block in self.transformer_blocks:
            tokens = block(tokens, context)
        tokens = self.proj_out(tokens)
        tokens = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return tokens.contiguous() + residual
