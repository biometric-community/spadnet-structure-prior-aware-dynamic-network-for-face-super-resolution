# SPADNet Results Report (full protocol)

## Summary

| Run | PSNR (dB) | SSIM | N | Protocol |
|-----|-----------|------|---|----------|
| Eval (smoke) | 22.473 | 0.5620 | 128 | `smoke` |
| Full train val (epoch 6) | 26.862 | 0.7787 | — | `full` in progress |
| Paper Table I (×8, prose) | see PDF (×4≈31.95 dB) | see PDF | 1000 test | author protocol |

_Scale ×8; dataset `celeba`. Table I numeric grid not OCR'd; use paper PDF for full baseline grid. Full protocol: 50 epochs (D3); refresh after completion._

## Setup

- Dataset: CelebA (`../../datasets/celeba/extracted/celeba`) + Helen (`../../datasets/helen`)
- Scale: ×8 (LR → HR 128×128)
- Size gate: **3.3891 GiB** total → `full_train` (`outputs/logs/dataset_size.json`)
- Protocol: `full`
- Upstream refactor: https://github.com/wcy-cs/SPADNet → package `spadnet`

## Figures

_Train curves through epoch 6: val PSNR=26.862 dB, SSIM=0.7787._

![train_val_curves](outputs/figures/train_val_curves.svg)

![metrics_bar_celeba_x8](outputs/figures/metrics_bar_celeba_x8.svg)

## Regenerate

```bash
bash scripts/report.sh
# After full train finishes:
bash scripts/eval.sh --config configs/full.yaml --checkpoint outputs/checkpoints/full/latest.pt
bash scripts/report.sh --config configs/full.yaml
```
