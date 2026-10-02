"""Evaluate SPADNet PSNR/SSIM on test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spadnet.data import build_dataloader
from spadnet.metrics.psnr_ssim import calc_psnr_ssim
from spadnet.models import build_spadnet


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    req = cfg.get("device", "cuda")
    if req == "cuda" and torch.cuda.is_available():
        gpu = cfg.get("cuda_device")
        if gpu is not None:
            return torch.device(f"cuda:{int(gpu)}")
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate SPADNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "latest.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_spadnet(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    loader = build_dataloader(cfg, args.split)
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr, _ = model(lr)
            for i in range(sr.size(0)):
                p, s = calc_psnr_ssim(sr[i].clamp(0, 1), hr[i])
                psnr_sum += p
                ssim_sum += s
                n += 1

    results = {
        "split": args.split,
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
        "n": n,
        "checkpoint": str(ckpt_path),
        "dataset": cfg["data"].get("dataset", "celeba"),
        "scale": cfg["model"]["scale"],
        "protocol": cfg["train"].get("protocol", "subset"),
    }
    out = Path(cfg["paths"]["log_dir"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
