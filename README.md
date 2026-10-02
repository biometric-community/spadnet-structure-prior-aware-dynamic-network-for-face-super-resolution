# SPADNet: Structure Prior-Aware Dynamic Network for Face Super-Resolution

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch reproduction of Wang et al., *SPADNet* (IEEE TBIOM 2024): structure prior-aware dynamic kernels (LSAC + GSAC) for face super-resolution.

**Repository:** [biometric-community/spadnet-structure-prior-aware-dynamic-network-for-face-super-resolution](https://github.com/biometric-community/spadnet-structure-prior-aware-dynamic-network-for-face-super-resolution)

Monorepo path (submodule): `projects/papers/spadnet-structure-prior-aware-dynamic-network-for-face-super-resolution/`

## Features

- Dual-branch SPADNet (SRB + HEB) with Structure-Aware Blocks
- CelebA and Helen loaders under `projects/datasets/`
- Train / eval / predict / report CLIs with YAML configs
- Fidelity audit vs paper + upstream `wcy-cs/SPADNet`

## Setup

Uses the monorepo shared `.venv` (no project-local venv):

```bash
bash scripts/setup_env.sh
```

## Data

Point configs at real corpora (never bundled here):

- CelebA: `../../datasets/celeba/extracted/celeba/` (`img_align_celeba`, partition + landmarks)
- Helen: `../../datasets/helen/` (`extracted/SmithCVPR2013_dataset_resized/{images,points}`)

```bash
bash scripts/check_dataset_size.sh \
  ../../datasets/celeba/extracted/celeba \
  ../../datasets/helen
```

## Train / Eval / Predict / Report

```bash
# Smoke (capped samples)
bash scripts/train.sh
bash scripts/eval.sh --checkpoint outputs/checkpoints/latest.pt
bash scripts/predict.sh --checkpoint outputs/checkpoints/latest.pt
bash scripts/report.sh

# Full protocol when size gate < 5 GiB
bash scripts/train_full.sh
```

See `REPORT.md` for metrics and figures regenerated from real logs only.

## Layout

```
configs/          default.yaml (smoke), full.yaml
spadnet/          package (models, data, losses, metrics, train/eval/predict/report)
scripts/          setup_env, train, train_full, eval, predict, report, check_dataset_size
```

## Fidelity & deviations

- `FIDELITY_AUDIT.md` — five-pass audit + statistics
- `DEVIATIONS.md` — justified `Dn` (landmarks, splits, epoch budget, …)
- `SOURCE_CODE.md` — upstream refactor map

## Citation

```bibtex
@article{wang2024spadnet,
  title={Structure Prior-Aware Dynamic Network for Face Super-Resolution},
  author={Wang, Chenyang and Jiang, Junjun and Jiang, Kui and Liu, Xianming},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={6},
  number={3},
  year={2024},
  doi={10.1109/TBIOM.2024.3382870}
}
```

## License

Code and documentation: CC BY 4.0 (see `LICENSE`). Datasets and the original paper retain their own terms.
