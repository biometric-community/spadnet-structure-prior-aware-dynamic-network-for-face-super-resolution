"""Shared conv / upsample blocks (upstream model/common.py → project layout)."""

from __future__ import annotations

import math

import torch.nn as nn


def default_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=bias,
    )


class Upsampler(nn.Sequential):
    """PixelShuffle upsampler (Sec. IV-B, RCAB + pixelshuffle [67])."""

    def __init__(
        self,
        conv,
        scale: int,
        n_feats: int,
        bn: bool = False,
        act: bool = False,
        bias: bool = True,
    ) -> None:
        modules: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feats))
                if act:
                    modules.append(nn.ReLU(True))
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feats))
            if act:
                modules.append(nn.ReLU(True))
        else:
            raise NotImplementedError(f"Unsupported scale {scale}")
        super().__init__(*modules)


class DownBlock(nn.Module):
    """Space-to-depth downsample at network head (Eq. 1 input path)."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        s = self.scale
        x = x.view(n, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(n, c * (s**2), h // s, w // s)
