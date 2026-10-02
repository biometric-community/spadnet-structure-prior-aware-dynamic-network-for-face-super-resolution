"""Residual Channel Attention Block (RCAB) used as BasicBlock (paper Sec. IV-B, [66])."""

from __future__ import annotations

import torch
import torch.nn as nn


class CALayer(nn.Module):
    """Channel attention (squeeze-excitation style)."""

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block — paper BasicBlock."""

    def __init__(
        self,
        n_feats: int,
        kernel_size: int = 3,
        reduction: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        body: list[nn.Module] = []
        for i in range(2):
            body.append(
                nn.Conv2d(
                    n_feats,
                    n_feats,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            if i == 0:
                body.append(nn.ReLU(inplace=True))
        body.append(CALayer(n_feats, reduction=reduction))
        self.body = nn.Sequential(*body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x
