"""Helen FSR loader (Sec. IV-A: 2005 train / 50 test, augmentation)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .celeba import resolve_data_root
from .heatmap import augment_sample, generate_component_heatmaps


def _load_helen_points(path: Path) -> np.ndarray:
    lines = path.read_text().strip().splitlines()
    pts = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float32)


def _helen_split_names(img_dir: Path, split: str) -> List[str]:
    """Paper Sec. IV-A: 2005 train / 50 test (remaining held out as val)."""
    stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if split == "train":
        return stems[:2005]
    if split == "test":
        return stems[2005:2055]
    return stems[2055:]  # remainder as val (~275)


class HelenFSRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 128,
        scale: int = 8,
        max_samples: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.root = resolve_data_root(root)
        base = self.root / "extracted" / "SmithCVPR2013_dataset_resized"
        if not base.is_dir():
            base = self.root / "SmithCVPR2013_dataset_resized"
        self.img_dir = base / "images"
        self.pts_dir = base / "points"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Helen images not found at {self.img_dir}")

        names = _helen_split_names(self.img_dir, split)
        if max_samples is not None and max_samples > 0:
            names = names[: max_samples]
        self.names = names
        self.img_size = img_size
        self.scale = scale
        self.lr_size = img_size // scale
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        stem = self.names[index]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        ow, oh = img.size
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        pts_path = self.pts_dir / f"{stem}.txt"
        if pts_path.is_file():
            pts = _load_helen_points(pts_path)
            # Scale landmarks from native resolution → 128×128 crop space.
            pts = pts.copy()
            pts[:, 0] *= self.img_size / float(ow)
            pts[:, 1] *= self.img_size / float(oh)
        else:
            pts = np.zeros((5, 2), dtype=np.float32)

        heatmap = generate_component_heatmaps(self.img_size, pts, sigma=1.0)
        lr_native = img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_img = lr_native.resize((self.img_size, self.img_size), Image.BICUBIC)
        lr = np.asarray(lr_img, dtype=np.float32)
        hr = np.asarray(img, dtype=np.float32)
        if self.augment:
            lr, hr, heatmap = augment_sample(lr, hr, heatmap)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            return t.permute(2, 0, 1) if t.ndim == 3 else t

        return {
            "lr": to_tensor(lr),
            "hr": to_tensor(hr),
            "heatmap": torch.from_numpy(np.ascontiguousarray(heatmap)).float(),
            "name": stem,
        }


def build_helen_loader(cfg: dict, split: str):
    from torch.utils.data import DataLoader

    data = cfg["data"]
    max_key = {"train": "max_train_samples", "val": "max_val_samples", "test": "max_test_samples"}[split]
    max_samples = data.get(max_key)
    ds = HelenFSRDataset(
        root=data.get("helen_root", "../../datasets/helen"),
        split=split if split != "val" else "val",
        img_size=int(cfg["model"]["img_size"]),
        scale=int(cfg["model"]["scale"]),
        max_samples=int(max_samples) if max_samples is not None else None,
        augment=bool(data.get("augment_helen", True)),
    )
    bs = int(cfg["train"]["batch_size"] if split == "train" else cfg["eval"].get("batch_size", 1))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(split == "train"),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        drop_last=(split == "train"),
        pin_memory=True,
    )
