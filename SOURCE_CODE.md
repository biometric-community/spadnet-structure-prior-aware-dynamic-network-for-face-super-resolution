# Source code — SPADNet

## Upstream

| Field | Value |
|-------|-------|
| URL | https://github.com/wcy-cs/SPADNet |
| Commit inspected | `2f185eb3a8b81aa30957b41e8b49b2b5de86bde4` |
| Framework | PyTorch (legacy 1.1.0 noted in README) |
| License | Not stated in upstream README; PacConv file header is **CC BY-NC-SA 4.0** (NVIDIA) |
| Local mirror | `upstream/SPADNet/` (gitignored; refactor source only) |

## Mapping (upstream → project)

| Upstream | Project | Notes |
|----------|---------|-------|
| `model/unet.py` `HAPFSR` | `spadnet/models/spadnet.py` `SPADNet` | Same dual-branch interleave of RCAB + HAB |
| `model/hab.py` `HeatmapAttentionBlock35G2` | `spadnet/models/hab.py` `HeatmapAttentionBlock` | LSAC (PacConv) + GSAC global bias |
| `model/pac.py` | `spadnet/models/pac_conv.py` | Native PyTorch rewrite; drops deprecated `torch._thnn` |
| `model/common.py` RCAB / Upsampler / DownBlock | `spadnet/models/{rcab,common}.py` | |
| `data/dataset_landmark.py` | `spadnet/data/{celeba,helen,heatmap}.py` | Public CelebA/Helen instead of author `HR/` + `landmark.pkl` |
| `main_landmark.py` | `spadnet/train.py` | L1+heatmap L1; Adam; MultiStepLR |
| `util.py` metrics | `spadnet/metrics/psnr_ssim.py` | Y-channel PSNR/SSIM with border crop |
| `test.py` | `spadnet/eval.py`, `predict.py` | |

## Conflict rule

PDF / `analysis.md` overrides upstream when they disagree. Upstream bugfixes kept when clearly correcting paper typos (`upstream_bugfix` in DEVIATIONS if any).

## What we did not vendor as-is

- Full NVIDIA PacConv CUDA/`_thnn` backend
- TensorBoardX training loop / Baidu pretrained weights
- Author-preprocessed CelebA pickle directories
