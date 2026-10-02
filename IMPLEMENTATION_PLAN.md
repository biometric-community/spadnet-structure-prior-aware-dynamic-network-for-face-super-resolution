# Implementation plan — SPADNet

- **Paper:** `papers/spadnet-structure-prior-aware-dynamic-network-for-face-super-resolution/`
- **Project:** `projects/papers/spadnet-structure-prior-aware-dynamic-network-for-face-super-resolution/`
- **Package:** `spadnet`
- **Upstream:** https://github.com/wcy-cs/SPADNet (`2f185eb`)

## Method summary

Two-branch face SR network (SRB + HEB) with Structure-Aware Blocks (SAB / HAB) that apply:

1. **LSAC** — local structure-adaptive (pixel-adaptive) convolution guided by heatmap features
2. **GSAC** — global structure-aware bias from pooled heatmap features

Loss: `L1(SR, HR) + L1(HSR, HHR)` (Eqs. 12–13). Optimizer: Adam 1e-4, β=(0.9, 0.999), LR halved every 10 epochs.

## Figure inventory (Results)

| Paper fig | Type | Metrics / axes | Our artifact |
|-----------|------|----------------|--------------|
| Fig. 3–5 | Visual SR grid | qualitative | `outputs/predictions/*_sr.png` (predict) |
| Table I | PSNR/SSIM bars | scale × dataset | `outputs/figures/metrics_bar_*.svg` |
| Fig. 7 | Face recognition accuracy | rank/IDR (optional) | skipped (LightFace not wired) — D5 |
| Train curves | loss / val PSNR | epoch | `outputs/figures/train_val_curves.svg` |

## Dataset size gate

Required roots: CelebA + Helen under `projects/datasets/`.

| Root | Role |
|------|------|
| `../../datasets/celeba/extracted/celeba` | primary train/eval |
| `../../datasets/helen` | Helen protocol / transfer |

Measured via `scripts/check_dataset_size.sh` → `outputs/logs/dataset_size.json`.

| Measurement | Value |
|-------------|-------|
| CelebA bytes | 2 910 419 069 (~2.71 GiB) |
| Helen bytes | 728 647 924 (~0.68 GiB) |
| **Total** | **3 639 066 993 (~3.39 GiB)** |
| Gate (< 5 GiB) | **full_train** — start `scripts/train_full.sh` |

## Module map

| Paper | Code |
|-------|------|
| SPADNet / HAPFSR | `spadnet/models/spadnet.py` |
| SAB / HAB (LSAC+GSAC) | `spadnet/models/hab.py` + `pac_conv.py` |
| RCAB BasicBlock | `spadnet/models/rcab.py` |
| Heatmaps HHR | `spadnet/data/heatmap.py` |
| CelebA / Helen loaders | `spadnet/data/celeba.py`, `helen.py` |
| Loss | `spadnet/losses/spadnet_loss.py` |
| PSNR/SSIM-Y | `spadnet/metrics/psnr_ssim.py` |

## Train protocols

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | smoke (capped samples, few steps) |
| `configs/full.yaml` | full CelebA splits, 50 epochs (D3 vs upstream 2000) |
