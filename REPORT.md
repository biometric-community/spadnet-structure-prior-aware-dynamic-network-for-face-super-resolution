# SPADNet Results Report

## Summary

| Metric | Ours (smoke) | Paper Table I |
|--------|------|------|
| PSNR (dB) | 22.473 | see PDF (prose cites ×4≈31.95 dB) |
| SSIM | 0.5620 | see PDF Table I |
| N test | 128 | CelebA 1000 / Helen 50 |

_Scale ×8; dataset `celeba`. Table I numeric grid not OCR'd; use paper PDF for full baseline grid._

## Setup

- Dataset: `celeba`
- Scale: ×8
- Protocol: `smoke`

## Figures

![train_val_curves](outputs/figures/train_val_curves.svg)

![metrics_bar_celeba_x8](outputs/figures/metrics_bar_celeba_x8.svg)

## Regenerate

```bash
bash scripts/report.sh
```
