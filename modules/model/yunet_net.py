"""The YuNet face detector as a torch module.

The network answers twelve tensors, four per stride: a class score, an objectness score, a
box and five keypoints. Decoding them into faces is :mod:`modules.model.yunet`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["STRIDES", "ConvDPUnit", "YuNet"]

#: The strides the three feature maps are taken at, largest map first.
STRIDES = (8, 16, 32)

#: Channels each head answers per anchor: one class score, one objectness score, a box, and
#: five keypoints as x and y.
HEAD_CHANNELS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}


class ConvDPUnit(nn.Module):
    """A pointwise convolution followed by a depthwise one.

    Attributes:
        pointwise: The 1x1 convolution that changes the channel count.
        depthwise: The 3x3 convolution applied to each channel on its own.
    """

    def __init__(self, in_channels: int, out_channels: int, activation: bool = True):
        """Build the pair.

        Args:
            in_channels: Channels arriving.
            out_channels: Channels leaving, and the number of depthwise groups.
            activation: Whether a ReLU follows the depthwise convolution.
        """
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Answer the pair applied to ``x``."""
        x = self.depthwise(self.pointwise(x))
        return F.relu(x) if self.activation else x


class YuNet(nn.Module):
    """The detector's backbone, feature pyramid and four heads.

    Attributes:
        stem: The strided convolution the image enters through.
        heads: One :class:`ConvDPUnit` per head per stride, keyed ``<head>_<stride>``.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        self.block0 = ConvDPUnit(16, 16)
        self.block1 = nn.ModuleList(
            [ConvDPUnit(16, 16), ConvDPUnit(16, 32), ConvDPUnit(32, 32), ConvDPUnit(32, 64)]
        )
        self.block2 = nn.ModuleList([ConvDPUnit(64, 64), ConvDPUnit(64, 64)])
        self.block3 = nn.ModuleList([ConvDPUnit(64, 64), ConvDPUnit(64, 64)])
        self.block4 = nn.ModuleList(
            [ConvDPUnit(64, 64), ConvDPUnit(64, 64), ConvDPUnit(64, 64)]
        )

        self.merge16 = ConvDPUnit(64, 64)
        self.merge8 = ConvDPUnit(64, 64)

        self.heads = nn.ModuleDict(
            {
                f"{name}_{stride}": ConvDPUnit(64, channels, activation=False)
                for name, channels in HEAD_CHANNELS.items()
                for stride in STRIDES
            }
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the detector.

        Args:
            image: ``(batch, 3, height, width)``, both sides a multiple of 32.

        Returns:
            ``{"<head>_<stride>": tensor}``, each ``(batch, anchors, channels)``. The class
            and objectness scores have passed through a sigmoid; the box and the keypoints
            are the raw offsets the decoder expects.
        """
        x = F.relu(self.stem(image))
        x = self.block0(x)

        x = F.max_pool2d(x, 2, 2)
        for unit in self.block1:
            x = unit(x)

        x = F.max_pool2d(x, 2, 2)
        for unit in self.block2:
            x = unit(x)
        features8 = x

        x = F.max_pool2d(x, 2, 2)
        for unit in self.block3:
            x = unit(x)
        features16 = x

        x = F.max_pool2d(x, 2, 2)
        for unit in self.block4:
            x = unit(x)
        features32 = x

        merged = features16 + F.interpolate(features32, scale_factor=2, mode="nearest")
        features16 = self.merge16(merged)
        merged = features8 + F.interpolate(features16, scale_factor=2, mode="nearest")
        features8 = self.merge8(merged)

        by_stride = {8: features8, 16: features16, 32: features32}
        answered = {}
        for name, channels in HEAD_CHANNELS.items():
            for stride in STRIDES:
                key = f"{name}_{stride}"
                out = self.heads[key](by_stride[stride])
                out = out.permute(0, 2, 3, 1).reshape(out.shape[0], -1, channels)
                answered[key] = torch.sigmoid(out) if name in ("cls", "obj") else out
        return answered
