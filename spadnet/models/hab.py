"""Structure-Aware Block (SAB) / Heatmap Attention Block (Sec. III-B, Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .pac_conv import PacConv2d


class HeatmapAttentionBlock(nn.Module):
    """HAB with LSAC (PacConv) + GSAC global bias (upstream HeatmapAttentionBlock35G2)."""

    def __init__(self, n_feats: int = 64) -> None:
        super().__init__()
        self.conv1_sr = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1)
        self.conv1_heatmap = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1)
        self.conv2_heatmap = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1)
        self.pacconv1 = PacConv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(n_feats * 2, n_feats, 3, padding=1, stride=1)
        self.relu = nn.ReLU(True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, heatmap_feat: torch.Tensor) -> torch.Tensor:
        sr1 = self.conv1_sr(x)
        b, c, h, w = heatmap_feat.shape
        hm1 = self.conv1_heatmap(heatmap_feat)
        f1 = self.pacconv1(sr1, hm1)
        hm1 = self.pool(hm1)
        hm2 = self.conv2_heatmap(hm1)
        global_bias = torch.sigmoid(hm2)
        bias = global_bias.view(b, c).unsqueeze(1).repeat(1, h * w, 1)
        bias = bias.reshape(b, c, h, w) * sr1
        f = torch.cat((f1, bias), dim=1)
        f = self.relu(self.conv3(f))
        return x + self.conv2(f)
