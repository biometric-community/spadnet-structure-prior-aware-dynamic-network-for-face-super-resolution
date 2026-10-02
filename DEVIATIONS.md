# Deviations — SPADNet

Intentional mismatches vs Wang et al. (TBIOM 2024) / https://github.com/wcy-cs/SPADNet.

| ID | Topic | Paper / upstream | Ours | Rationale |
|----|-------|------------------|------|-----------|
| D1 | Landmark GT | OpenFace **68** landmarks → 5 component heatmaps | CelebA **5**-point `list_landmarks_align_celeba.txt` (+ synthetic jaw ellipse); Helen **194**-pt grouped into 5 channels | Author `landmark.pkl` / OpenFace dumps not shipped; channel count (5) preserved |
| D2 | CelebA splits | DIC-style 168 854 / 100 / 1000 non-overlapping | Official CelebA `list_eval_partition.txt` (train/val/test) | Public partition; counts differ — documented |
| D3 | Epoch budget | Upstream default `--epochs 2000`; paper does not state epoch count (only LR half every 10 epochs) | `configs/full.yaml`: **50** epochs on full CelebA train | Practical wall-clock; same Adam/LR schedule form |
| D4 | LR tensor size | Author folders store bicubic LR; head uses space-to-depth on **HR-sized** input | Bicubic downsample then upsample to 128×128 before the network | Matches upstream `DownBlock(scale)` geometry |
| D5 | Face recognition Fig. 7 | LightFace recognition accuracy | Not implemented | Extra recognizer stack out of default FSR scope |
| D6 | RaFD / race-age Table III | Additional demographic eval | Missing dataset → not run | Documented blocker; no fake metrics |
| D7 | PacConv backend | Upstream NVIDIA PacConv (`torch._thnn`) | Native unfold/einsum `pac_conv.py` | Modern PyTorch; same LSAC math (Eqs. 8–10) |
| D8 | Upstream β2 | `betas=(0.9, 0.99)` in `main_landmark.py` | Paper **0.999** in configs | PDF overrides upstream |
| D9 | Helen 194 grouping | Paper OpenFace 68 index bands | Approximate Helen contour/eye/nose/mouth bands | Helen has 194 pts; D1 covers landmark source |
| D10 | Stage-3 HAB loop | Upstream `unet.py` uses `len(self.SR_branch2)` for stage 3 | Uses `len(sr_blocks)` for the stage-3 Sequential | `upstream_bugfix` — stage 3 must index branch3 |

## Not deviations

- Dual-branch SRB+HEB with interleaved HAB (Fig. 1)
- LSAC + GSAC inside HAB (Fig. 2)
- Loss L1(SR)+L1(heatmap)
- Adam 1e-4, LR ×0.5 every 10 epochs
- Metrics PSNR/SSIM on Y with border crop
- Helen 2005 train / 50 test + rotation/flip aug
