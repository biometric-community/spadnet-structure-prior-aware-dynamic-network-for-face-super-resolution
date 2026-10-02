"""Pixel-adaptive convolution implementing LSAC (Eqs. 8–10).

Refactored from upstream ``model/pac.py`` (NVIDIA PacConv, CC BY-NC-SA 4.0)
to a modern native PyTorch path that avoids deprecated ``torch._thnn``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _pair(x: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def gaussian_kernel_from_guidance(
    guidance: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    """Eqs. 8–9: disparity to center → spatially varying Gaussian weights.

    Returns kernel of shape ``(N, 1, kH, kW, H_out, W_out)``.
    """
    bs, ch, _, _ = guidance.shape
    cols = F.unfold(guidance, kernel_size, dilation, padding, stride)
    kH, kW = kernel_size
    out_h = (guidance.shape[2] + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_w = (guidance.shape[3] + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    cols = cols.view(bs, ch, kH, kW, out_h, out_w)
    cy, cx = kH // 2, kW // 2
    center = cols[:, :, cy : cy + 1, cx : cx + 1, :, :]
    diff_sq = (cols - center).pow(2).sum(dim=1, keepdim=True)
    return torch.exp(-0.5 * diff_sq)


def pac_conv2d_native(
    input_2d: torch.Tensor,
    kernel: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    """Apply PacConv: unfold × adaptive kernel × shared filter bank (Eq. 10)."""
    bs, ch = input_2d.shape[:2]
    kH, kW = weight.shape[-2:]
    cols = F.unfold(input_2d, (kH, kW), dilation, padding, stride)
    out_h = (input_2d.shape[2] + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_w = (input_2d.shape[3] + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    in_mul_k = cols.view(bs, ch, kH, kW, out_h, out_w) * kernel
    output = torch.einsum("ijklmn,ojkl->iomn", in_mul_k, weight)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


class PacConv2d(nn.Module):
    """Local structure-adaptive convolution (LSAC) via pixel-adaptive filters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n = self.weight.size(1)
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_2d: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        kernel = gaussian_kernel_from_guidance(
            guidance, self.kernel_size, self.stride, self.padding, self.dilation
        )
        return pac_conv2d_native(
            input_2d,
            kernel,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
