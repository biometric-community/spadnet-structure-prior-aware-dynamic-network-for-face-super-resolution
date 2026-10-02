"""Pixel L1 + heatmap L1 loss (Sec. III-C, Eqs. 12–13)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SPADNetLoss(nn.Module):
    """L = L1(ISR, IHR) + LHeatmap(HSR, HHR)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        heatmap_pred: torch.Tensor,
        heatmap_gt: torch.Tensor,
    ) -> torch.Tensor:
        return self.l1(sr, hr) + self.l1(heatmap_pred, heatmap_gt)
