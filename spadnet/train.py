"""Train SPADNet (Eqs. 12–13, Sec. IV-B)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spadnet.data import build_dataloader
from spadnet.losses import SPADNetLoss
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


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr, _ = model(lr)
        for i in range(sr.size(0)):
            p, s = calc_psnr_ssim(sr[i].clamp(0, 1), hr[i])
            psnr_sum += p
            ssim_sum += s
            n += 1
    model.train()
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1), "n": n}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train SPADNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    model = build_spadnet(cfg).to(device)
    criterion = SPADNetLoss()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=(float(cfg["train"].get("beta1", 0.9)), float(cfg["train"].get("beta2", 0.999))),
        eps=1e-8,
    )

    steps_per_epoch = max(len(train_loader), 1)
    decay_every = int(cfg["train"].get("lr_decay_epochs", 10))
    milestones = [steps_per_epoch * decay_every * (i + 1) for i in range(int(cfg["train"].get("lr_decay_steps", 5)))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)

    epochs = int(cfg["train"]["epochs"])
    max_steps = args.max_steps or cfg["train"].get("max_steps")
    log_every = int(cfg["train"].get("log_every", 50))
    history = {"epochs": [], "train_loss": [], "val_psnr": [], "val_ssim": []}
    global_step = 0
    start_epoch = 0

    resume_path = args.resume
    if resume_path is None and (ckpt_dir / "latest.pt").is_file():
        # Prefer explicit --resume; otherwise continue full runs from latest.pt when present.
        if cfg["train"].get("protocol") == "full":
            resume_path = str(ckpt_dir / "latest.pt")
    if resume_path:
        ckpt_blob = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt_blob["model"])
        if "optimizer" in ckpt_blob:
            opt.load_state_dict(ckpt_blob["optimizer"])
        start_epoch = int(ckpt_blob.get("epoch", -1)) + 1
        hist_path = log_dir / "train_history.json"
        if hist_path.is_file():
            loaded = json.loads(hist_path.read_text())
            # Guard against corrupted scalar-shaped history (must be per-epoch lists).
            if isinstance(loaded.get("epochs"), list):
                history = {
                    "epochs": list(loaded.get("epochs", [])),
                    "train_loss": list(loaded.get("train_loss", [])),
                    "val_psnr": list(loaded.get("val_psnr", [])),
                    "val_ssim": list(loaded.get("val_ssim", [])),
                }
            else:
                print(f"Warning: ignoring non-list train_history at {hist_path}")
        global_step = start_epoch * steps_per_epoch
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed from {resume_path} at epoch={start_epoch} step={global_step}")

    print(
        f"SPADNet train device={device} train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} scale=x{cfg['model']['scale']}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        count = 0
        for batch in train_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            hm_gt = batch["heatmap"].to(device)
            sr, hm = model(lr)
            loss = criterion(sr, hr, hm, hm_gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            global_step += 1
            running += loss.item()
            count += 1
            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e}")
            if max_steps is not None and global_step >= int(max_steps):
                break

        val_m = validate(model, val_loader, device)
        avg_loss = running / max(count, 1)
        history["epochs"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_psnr"].append(val_m["psnr"])
        history["val_ssim"].append(val_m["ssim"])

        ckpt = ckpt_dir / f"spadnet_epoch{epoch:04d}.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt)
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt_dir / "latest.pt")

        summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_loss,
            "val_psnr": val_m["psnr"],
            "val_ssim": val_m["ssim"],
            "protocol": cfg["train"].get("protocol", "subset"),
            "dataset": cfg["data"].get("dataset", "celeba"),
            "scale": cfg["model"]["scale"],
        }
        with open(log_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        with open(log_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
            f.write("\n")

        print(
            f"Epoch {epoch} ({time.time()-t0:.1f}s) loss={avg_loss:.4f} "
            f"val_psnr={val_m['psnr']:.3f} val_ssim={val_m['ssim']:.4f}"
        )
        if max_steps is not None and global_step >= int(max_steps):
            break


if __name__ == "__main__":
    main()
