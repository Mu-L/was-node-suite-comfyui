"""The latent diffusion autoencoder, as a torch module.

:class:`AutoencoderKL` carries pictures on a minus 1 to 1 scale into a four channel latent an
eighth of each side, and back. Parameter names follow the published checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .latent_blocks import (
    Downsample,
    ResnetBlock,
    SpatialAttention,
    Upsample,
    group_norm,
)

__all__ = [
    "BLOCK_CHANNELS",
    "AutoencoderKL",
    "Decoder",
    "DownEncoderBlock",
    "Encoder",
    "LATENT_CHANNELS",
    "LAYERS_PER_BLOCK",
    "MidBlock",
    "NORM_EPS",
    "SCALING_FACTOR",
    "UpDecoderBlock",
]

#: Channels each resolution of the published checkpoints works at, largest map first.
BLOCK_CHANNELS = (128, 256, 512, 512)

#: Residual blocks per resolution in the encoder. The decoder runs one more.
LAYERS_PER_BLOCK = 2

#: Channels the latent carries.
LATENT_CHANNELS = 4

#: Value added to the variance in every normalisation.
NORM_EPS = 1e-6

#: What a latent is multiplied by on the way out of :meth:`AutoencoderKL.encode`.
SCALING_FACTOR = 0.18215


class DownEncoderBlock(nn.Module):
    """Residual blocks at one encoder resolution.

    Attributes:
        resnets: The residual blocks, run in order.
        downsamplers: The halving convolution, or ``None`` at the deepest resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, layers: int, downsample: bool):
        """Build the block.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels every residual block leaves.
            layers: Number of residual blocks.
            downsample: Whether a halving convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_channels if index == 0 else out_channels, out_channels, None, NORM_EPS
            )
            for index in range(layers)
        )
        self.downsamplers = (
            nn.ModuleList([Downsample(out_channels, padding=0)]) if downsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the block applied to ``x``."""
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
        return x


class UpDecoderBlock(nn.Module):
    """Residual blocks at one decoder resolution.

    Attributes:
        resnets: The residual blocks, run in order.
        upsamplers: The doubling convolution, or ``None`` at the finest resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, layers: int, upsample: bool):
        """Build the block.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels every residual block leaves.
            layers: Number of residual blocks.
            upsample: Whether a doubling convolution follows them.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(
                in_channels if index == 0 else out_channels, out_channels, None, NORM_EPS
            )
            for index in range(layers)
        )
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the block applied to ``x``."""
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class MidBlock(nn.Module):
    """A residual block, spatial attention and a second residual block.

    Attributes:
        resnets: The two residual blocks.
        attentions: The attention between them.
    """

    def __init__(self, channels: int):
        """Build the block.

        Args:
            channels: Channels arriving and leaving.
        """
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(channels, channels, None, NORM_EPS) for _ in range(2)
        )
        self.attentions = nn.ModuleList([SpatialAttention(channels, NORM_EPS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the block applied to ``x``."""
        x = self.resnets[0](x)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attention(x))
        return x


class Encoder(nn.Module):
    """The half carrying a picture down to twice the latent channels.

    Attributes:
        down_blocks: The resolutions, finest first.
        mid_block: The bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = LATENT_CHANNELS,
        block_channels: tuple[int, ...] = BLOCK_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
    ):
        """Build the encoder.

        Args:
            in_channels: Channels the picture arrives with.
            out_channels: Channels of the latent, doubled on the way out for the variance.
            block_channels: Channels each resolution works at, finest first.
            layers_per_block: Residual blocks per resolution.
        """
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        arriving = block_channels[0]
        for index, channels in enumerate(block_channels):
            last = index == len(block_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock(arriving, channels, layers_per_block, not last)
            )
            arriving = channels
        self.mid_block = MidBlock(block_channels[-1])
        self.conv_norm_out = group_norm(block_channels[-1], NORM_EPS)
        self.conv_out = nn.Conv2d(
            block_channels[-1], out_channels * 2, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the mean and log variance of the latent for ``x``, stacked on channels."""
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(functional.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """The half carrying a latent back up to a picture.

    Attributes:
        up_blocks: The resolutions, coarsest first.
        mid_block: The bottleneck.
    """

    def __init__(
        self,
        in_channels: int = LATENT_CHANNELS,
        out_channels: int = 3,
        block_channels: tuple[int, ...] = BLOCK_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
    ):
        """Build the decoder.

        Args:
            in_channels: Channels the latent arrives with.
            out_channels: Channels the picture leaves with.
            block_channels: Channels each resolution works at, finest first.
            layers_per_block: One fewer than the residual blocks per resolution.
        """
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_channels[-1], kernel_size=3, padding=1)
        self.mid_block = MidBlock(block_channels[-1])
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(block_channels))
        arriving = reversed_channels[0]
        for index, channels in enumerate(reversed_channels):
            last = index == len(reversed_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock(arriving, channels, layers_per_block + 1, not last)
            )
            arriving = channels
        self.conv_norm_out = group_norm(block_channels[0], NORM_EPS)
        self.conv_out = nn.Conv2d(block_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the picture the decoder reads out of the latent ``x``."""
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(functional.silu(self.conv_norm_out(x)))


class AutoencoderKL(nn.Module):
    """The encoder and decoder pair, with the 1x1 convolutions either side of the latent.

    Attributes:
        encoder: The picture to latent half.
        decoder: The latent to picture half.
        quant_conv: The 1x1 convolution the encoder's output passes through.
        post_quant_conv: The 1x1 convolution the decoder's input passes through.
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        block_channels: tuple[int, ...] = BLOCK_CHANNELS,
        layers_per_block: int = LAYERS_PER_BLOCK,
    ):
        """Build the autoencoder.

        Args:
            latent_channels: Channels the latent carries.
            block_channels: Channels each resolution works at, finest first.
            layers_per_block: Residual blocks per encoder resolution.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(3, latent_channels, block_channels, layers_per_block)
        self.decoder = Decoder(latent_channels, 3, block_channels, layers_per_block)
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the mean latent for a picture.

        Args:
            x: ``(batch, 3, height, width)`` on a minus 1 to 1 scale.

        Returns:
            ``(batch, latent_channels, height / 8, width / 8)``, unscaled.
        """
        moments = self.quant_conv(self.encoder(x))
        return moments.chunk(2, dim=1)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Answer the picture a latent decodes to.

        Args:
            latent: ``(batch, latent_channels, height, width)``, unscaled.

        Returns:
            ``(batch, 3, height * 8, width * 8)`` on a minus 1 to 1 scale, unclipped.
        """
        return self.decoder(self.post_quant_conv(latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer ``x`` encoded and decoded again."""
        return self.decode(self.encode(x))
