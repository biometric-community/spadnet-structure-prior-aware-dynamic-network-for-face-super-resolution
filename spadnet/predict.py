"""Run SPADNet inference on a split and save SR images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.utils import save_image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spadnet.data import build_dataloader
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
    parser = argparse.ArgumentParser(description="Predict with SPADNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=16)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoint_dir"]) / "latest.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_spadnet(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = build_dataloader(cfg, args.split)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            sr, _ = model(lr)
            for i, name in enumerate(batch["name"]):
                save_image(sr[i].clamp(0, 1), out_dir / f"{name}_sr.png")
                save_image(lr[i], out_dir / f"{name}_lr.png")
                saved += 1
                if saved >= args.max_images:
                    print(f"Saved {saved} predictions to {out_dir}")
                    return
    print(f"Saved {saved} predictions to {out_dir}")


if __name__ == "__main__":
    main()
