"""SPADNet: dual-branch SR + heatmap network (Sec. III-A, Fig. 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DownBlock, Upsampler, default_conv
from .hab import HeatmapAttentionBlock
from .rcab import RCAB


@dataclass
class SPADNetConfig:
    scale: int = 8
    n_feats: int = 64
    number: int = 6
    hab_num: int = 5


def _basic_block(n_feats: int, number: int) -> nn.Sequential:
    return nn.Sequential(*[RCAB(n_feats) for _ in range(number)])


def _hab_block(n_feats: int, hab_num: int) -> nn.Sequential:
    return nn.Sequential(*[HeatmapAttentionBlock(n_feats) for _ in range(hab_num)])


class SPADNet(nn.Module):
    """Structure Prior-Aware Dynamic Network (HAPFSR architecture)."""

    def __init__(self, cfg: SPADNetConfig | dict) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = SPADNetConfig(
                scale=int(cfg.get("scale", 8)),
                n_feats=int(cfg.get("n_feats", 64)),
                number=int(cfg.get("number", 6)),
                hab_num=int(cfg.get("hab_num", 5)),
            )
        self.cfg = cfg
        n_feats = cfg.n_feats

        self.head = nn.Sequential(
            DownBlock(cfg.scale),
            nn.Conv2d(3 * cfg.scale**2, n_feats, 3, 1, 1),
        )

        self.sr_branch1 = _basic_block(n_feats, cfg.number)
        self.sr_up1 = Upsampler(default_conv, 2, n_feats)
        self.sr_branch2 = _basic_block(n_feats, cfg.number)
        self.sr_up2 = Upsampler(default_conv, 2, n_feats)
        self.sr_branch3 = _basic_block(n_feats, cfg.number)
        self.sr_up3 = Upsampler(default_conv, 2, n_feats)
        self.sr_out = nn.Conv2d(n_feats, 3, 3, padding=1, stride=1)

        self.hm_branch1 = _hab_block(n_feats, cfg.hab_num)
        self.hm_branch2 = _hab_block(n_feats, cfg.hab_num)
        self.hm_branch3 = _hab_block(n_feats, cfg.hab_num)

        self.heatmap_branch1 = _basic_block(n_feats, cfg.number)
        self.heatmap_up1 = Upsampler(default_conv, 2, n_feats)
        self.heatmap_branch2 = _basic_block(n_feats, cfg.number)
        self.heatmap_up2 = Upsampler(default_conv, 2, n_feats)
        self.heatmap_branch3 = _basic_block(n_feats, cfg.number)
        self.heatmap_up3 = Upsampler(default_conv, 2, n_feats)
        self.heatmap_out = nn.Conv2d(n_feats, 5, 3, padding=1, stride=1)

    def _interleave_hab(
        self,
        sr_feat: torch.Tensor,
        hm_feat: torch.Tensor,
        sr_blocks: nn.Sequential,
        hm_blocks: nn.Sequential,
        hab_blocks: nn.Sequential,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_rcab = len(sr_blocks)
        for i in range(n_rcab * 2 - 1):
            if i % 2 == 0:
                idx = i // 2
                sr_feat = sr_blocks[idx](sr_feat)
                hm_feat = hm_blocks[idx](hm_feat)
            else:
                sr_feat = hab_blocks[i // 2](sr_feat, hm_feat)
        return sr_feat, hm_feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.head(x)
        sr_feat = feature
        hm_feat = feature

        sr_feat, hm_feat = self._interleave_hab(
            sr_feat, hm_feat, self.sr_branch1, self.heatmap_branch1, self.hm_branch1
        )
        sr_feat = self.sr_up1(sr_feat) + F.interpolate(feature, scale_factor=2, mode="bicubic")
        hm_feat = self.heatmap_up1(hm_feat)

        sr_feat, hm_feat = self._interleave_hab(
            sr_feat, hm_feat, self.sr_branch2, self.heatmap_branch2, self.hm_branch2
        )
        sr_feat = self.sr_up2(sr_feat) + F.interpolate(feature, scale_factor=4, mode="bicubic")
        hm_feat = self.heatmap_up2(hm_feat)

        sr_feat, hm_feat = self._interleave_hab(
            sr_feat, hm_feat, self.sr_branch3, self.heatmap_branch3, self.hm_branch3
        )
        sr_feat = self.sr_up3(sr_feat) + F.interpolate(feature, scale_factor=8, mode="bicubic")
        hm_feat = self.heatmap_up3(hm_feat)

        return self.sr_out(sr_feat), self.heatmap_out(hm_feat)


def build_spadnet(cfg: dict) -> SPADNet:
    model_cfg = cfg.get("model", cfg)
    return SPADNet(model_cfg)
