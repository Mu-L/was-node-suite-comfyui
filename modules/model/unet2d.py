"""The cross attention UNet of latent diffusion, as a torch module.

:class:`UNet2DCondition` takes a latent, a timestep and a prompt embedding, and answers a
residual of the same size. Parameter names follow the published checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .latent_blocks import (
    Downsample,
    ResnetBlock,
    SpatialTransformer,
    TimestepEmbedding,
    Upsample,
    group_norm,
    timestep_embedding,
)

__all__ = [
    "BLOCK_CHANNELS",
    "CONTEXT_CHANNELS",
    "CrossAttnDownBlock",
    "CrossAttnUpBlock",
    "DownBlock",
    "LAYERS_PER_BLOCK",
    "MidBlock",
    "NORM_EPS",
    "UNet2DCondition",
    "UpBlock",
]

#: Channels each resolution of the published checkpoints works at, largest map first.
BLOCK_CHANNELS = (320, 640, 1280, 1280)

#: Residual blocks per resolution.
LAYERS_PER_BLOCK = 2

#: Width of the prompt embedding the cross attention reads.
CONTEXT_CHANNELS = 1024

#: Value added to the variance in every normalisation outside the transformers.
NORM_EPS = 1e-5


class DownBlock(nn.Module):
    """Residual blocks at one resolution, with no attention.

    Attributes:
        resnets: The residual blocks, run in order.
        downsamplers: The halving convolution, or ``None`` at the deepest resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_channels: int,
        layers: int,
        downsample: bool,
    ):
        """Build the block.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels every residual block leaves.
            embed_channels: Width of the timestep embedding.
            layers: Number of residual blocks.
            downsample: Whether a halving convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_channels if index == 0 else out_channels,
                out_channels,
                embed_channels,
                NORM_EPS,
            )
            for index in range(layers)
        )
        self.downsamplers = (
            nn.ModuleList([Downsample(out_channels)]) if downsample else None
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """Answer the block applied to ``x``, and every activation the up path reads.

        Args:
            x: ``(batch, in_channels, height, width)``.
            embedding: ``(batch, embed_channels)`` timestep embedding.

        Returns:
            The activation leaving, and the tuple of skip activations.
        """
        skips = ()
        for resnet in self.resnets:
            x = resnet(x, embedding)
            skips = skips + (x,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
            skips = skips + (x,)
        return x, skips


class CrossAttnDownBlock(nn.Module):
    """Residual blocks paired with transformers at one resolution.

    Attributes:
        resnets: The residual blocks.
        attentions: The transformer after each residual block.
        downsamplers: The halving convolution, or ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_channels: int,
        layers: int,
        downsample: bool,
    ):
        """Build the block.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels every residual block leaves.
            embed_channels: Width of the timestep embedding.
            layers: Number of residual and transformer pairs.
            downsample: Whether a halving convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_channels if index == 0 else out_channels,
                out_channels,
                embed_channels,
                NORM_EPS,
            )
            for index in range(layers)
        )
        self.attentions = nn.ModuleList(
            SpatialTransformer(out_channels, CONTEXT_CHANNELS) for _ in range(layers)
        )
        self.downsamplers = (
            nn.ModuleList([Downsample(out_channels)]) if downsample else None
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, context: torch.Tensor):
        """Answer the block applied to ``x``, and every activation the up path reads.

        Args:
            x: ``(batch, in_channels, height, width)``.
            embedding: ``(batch, embed_channels)`` timestep embedding.
            context: ``(batch, context_tokens, context_channels)`` prompt embedding.

        Returns:
            The activation leaving, and the tuple of skip activations.
        """
        skips = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            x = attention(resnet(x, embedding), context)
            skips = skips + (x,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
            skips = skips + (x,)
        return x, skips


class MidBlock(nn.Module):
    """A residual block, a transformer and a second residual block.

    Attributes:
        resnets: The two residual blocks.
        attentions: The transformer between them.
    """

    def __init__(self, channels: int, embed_channels: int):
        """Build the block.

        Args:
            channels: Channels arriving and leaving.
            embed_channels: Width of the timestep embedding.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(channels, channels, embed_channels, NORM_EPS) for _ in range(2)
        )
        self.attentions = nn.ModuleList([SpatialTransformer(channels, CONTEXT_CHANNELS)])

    def forward(
        self, x: torch.Tensor, embedding: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Answer the block applied to ``x``."""
        x = self.resnets[0](x, embedding)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attention(x, context), embedding)
        return x


class UpBlock(nn.Module):
    """Residual blocks reading skip activations at one resolution, with no attention.

    Attributes:
        resnets: The residual blocks, run in order.
        upsamplers: The doubling convolution, or ``None`` at the finest resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        embed_channels: int,
        layers: int,
        upsample: bool,
    ):
        """Build the block.

        Args:
            in_channels: Channels arriving from the block below.
            out_channels: Channels every residual block leaves.
            skip_channels: Channels of the skip activation the last residual block reads.
            embed_channels: Width of the timestep embedding.
            layers: Number of residual blocks.
            upsample: Whether a doubling convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                (in_channels if index == 0 else out_channels)
                + (skip_channels if index == layers - 1 else out_channels),
                out_channels,
                embed_channels,
                NORM_EPS,
            )
            for index in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def forward(
        self,
        x: torch.Tensor,
        skips: tuple,
        embedding: torch.Tensor,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Answer the block applied to ``x`` against ``skips``.

        Args:
            x: ``(batch, in_channels, height, width)``.
            skips: Skip activations, consumed last first.
            embedding: ``(batch, embed_channels)`` timestep embedding.
            size: Height and width to resize to, or ``None`` to double both sides.

        Returns:
            ``(batch, out_channels, height, width)`` before any resize.
        """
        for resnet in self.resnets:
            x = resnet(torch.cat([x, skips[-1]], dim=1), embedding)
            skips = skips[:-1]
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x, size)
        return x


class CrossAttnUpBlock(nn.Module):
    """Residual blocks reading skip activations, each followed by a transformer.

    Attributes:
        resnets: The residual blocks.
        attentions: The transformer after each residual block.
        upsamplers: The doubling convolution, or ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        embed_channels: int,
        layers: int,
        upsample: bool,
    ):
        """Build the block.

        Args:
            in_channels: Channels arriving from the block below.
            out_channels: Channels every residual block leaves.
            skip_channels: Channels of the skip activation the last residual block reads.
            embed_channels: Width of the timestep embedding.
            layers: Number of residual and transformer pairs.
            upsample: Whether a doubling convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                (in_channels if index == 0 else out_channels)
                + (skip_channels if index == layers - 1 else out_channels),
                out_channels,
                embed_channels,
                NORM_EPS,
            )
            for index in range(layers)
        )
        self.attentions = nn.ModuleList(
            SpatialTransformer(out_channels, CONTEXT_CHANNELS) for _ in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def forward(
        self,
        x: torch.Tensor,
        skips: tuple,
        embedding: torch.Tensor,
        context: torch.Tensor,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Answer the block applied to ``x`` against ``skips``.

        Args:
            x: ``(batch, in_channels, height, width)``.
            skips: Skip activations, consumed last first.
            embedding: ``(batch, embed_channels)`` timestep embedding.
            context: ``(batch, context_tokens, context_channels)`` prompt embedding.
            size: Height and width to resize to, or ``None`` to double both sides.

        Returns:
            ``(batch, out_channels, height, width)`` before any resize.
        """
        for resnet, attention in zip(self.resnets, self.attentions):
            x = attention(resnet(torch.cat([x, skips[-1]], dim=1), embedding), context)
            skips = skips[:-1]
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x, size)
        return x


class UNet2DCondition(nn.Module):
    """The cross attention UNet of latent diffusion.

    Attributes:
        in_channels: Channels the latent arrives with.
        out_channels: Channels the residual leaves with.
        down_blocks: The encoder, finest resolution first.
        mid_block: The bottleneck.
        up_blocks: The decoder, coarsest resolution first.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_channels: tuple[int, ...] = BLOCK_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
        attention_blocks: int = 3,
    ):
        """Build the network.

        Args:
            in_channels: Channels the latent arrives with.
            out_channels: Channels the residual leaves with.
            block_channels: Channels each resolution works at, finest first.
            layers_per_block: Residual blocks per resolution.
            attention_blocks: How many of the finest resolutions carry transformers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_channels = tuple(block_channels)
        embed_channels = block_channels[0] * 4
        self.embed_channels = embed_channels
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(block_channels[0], embed_channels)

        self.down_blocks = nn.ModuleList()
        arriving = block_channels[0]
        for index, channels in enumerate(block_channels):
            last = index == len(block_channels) - 1
            builder = CrossAttnDownBlock if index < attention_blocks else DownBlock
            self.down_blocks.append(
                builder(arriving, channels, embed_channels, layers_per_block, not last)
            )
            arriving = channels

        self.mid_block = MidBlock(block_channels[-1], embed_channels)

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_channels))
        arriving = reversed_channels[0]
        for index, channels in enumerate(reversed_channels):
            last = index == len(reversed_channels) - 1
            skip = reversed_channels[min(index + 1, len(reversed_channels) - 1)]
            builder = (
                UpBlock
                if index < len(reversed_channels) - attention_blocks
                else CrossAttnUpBlock
            )
            self.up_blocks.append(
                builder(
                    arriving,
                    channels,
                    skip,
                    embed_channels,
                    layers_per_block + 1,
                    not last,
                )
            )
            arriving = channels

        self.conv_norm_out = group_norm(block_channels[0], NORM_EPS)
        self.conv_out = nn.Conv2d(block_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor | int, context: torch.Tensor
    ) -> torch.Tensor:
        """Answer the residual the network reads out of ``x``.

        Args:
            x: ``(batch, in_channels, height, width)`` latent.
            timestep: A scalar timestep, or one per batch entry.
            context: ``(batch, context_tokens, context_channels)`` prompt embedding.

        Returns:
            ``(batch, out_channels, height, width)``.
        """
        steps = timestep
        if not torch.is_tensor(steps):
            steps = torch.tensor([steps], dtype=torch.int64, device=x.device)
        elif steps.ndim == 0:
            steps = steps[None].to(x.device)
        steps = steps.expand(x.shape[0])
        embedding = self.time_embedding(
            timestep_embedding(steps, self.block_channels[0]).to(dtype=x.dtype)
        )

        x = self.conv_in(x)
        skips = (x,)
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlock):
                x, produced = block(x, embedding, context)
            else:
                x, produced = block(x, embedding)
            skips = skips + produced

        x = self.mid_block(x, embedding, context)

        for index, block in enumerate(self.up_blocks):
            taken = skips[-len(block.resnets):]
            skips = skips[: -len(block.resnets)]
            size = None if index == len(self.up_blocks) - 1 else skips[-1].shape[2:]
            if isinstance(block, CrossAttnUpBlock):
                x = block(x, taken, embedding, context, size)
            else:
                x = block(x, taken, embedding, size)

        return self.conv_out(functional.silu(self.conv_norm_out(x)))
