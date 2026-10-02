"""CelebA FSR loader (Sec. IV-A; DIC-style splits via list_eval_partition)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .heatmap import augment_sample, generate_component_heatmaps


def load_partition(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_celeba_landmarks(path: Path) -> Dict[str, np.ndarray]:
    landmarks: Dict[str, np.ndarray] = {}
    with open(path) as f:
        next(f)
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            name = parts[0]
            coords = np.array([float(x) for x in parts[1:11]], dtype=np.float32).reshape(5, 2)
            landmarks[name] = coords
    return landmarks


def resolve_data_root(root: str | Path) -> Path:
    p = Path(root)
    if p.is_absolute() and p.exists():
        return p
    proj = Path(__file__).resolve().parents[2]
    cand = (proj / p).resolve()
    if cand.exists():
        return cand
    repo = proj
    while repo != repo.parent:
        if (repo / "docs" / "PAPERS.md").exists():
            break
        repo = repo.parent
    stripped = Path(*[x for x in p.parts if x != ".."])
    alt = (repo / "projects" / stripped).resolve()
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Dataset root not found: {root} (resolved {cand})")


class CelebAFSRDataset(Dataset):
    """128×128 HR, bicubic LR at ``scale``, 5-channel component heatmaps."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 128,
        scale: int = 8,
        max_samples: Optional[int] = None,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        self.root = resolve_data_root(root)
        self.img_dir = self.root / "img_align_celeba"
        part_path = self.root / "list_eval_partition.txt"
        lm_path = self.root / "list_landmarks_align_celeba.txt"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"CelebA images missing at {self.img_dir}")
        if not part_path.is_file() or not lm_path.is_file():
            raise FileNotFoundError(f"CelebA annotations missing under {self.root}")

        split_id = {"train": 0, "val": 1, "test": 2}[split]
        partition = load_partition(part_path)
        landmarks = load_celeba_landmarks(lm_path)
        names = sorted(n for n, sid in partition.items() if sid == split_id and n in landmarks)
        if max_samples is not None and max_samples > 0:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(names), size=min(max_samples, len(names)), replace=False)
            names = [names[i] for i in sorted(idx.tolist())]
        self.names = names
        self.landmarks = landmarks
        self.img_size = img_size
        self.scale = scale
        self.lr_size = img_size // scale
        self.split = split
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        img = Image.open(self.img_dir / name).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize(
            (self.img_size, self.img_size), Image.BICUBIC
        )

        pts = self.landmarks[name].copy()
        pts[:, 0] = (pts[:, 0] - left) * (self.img_size / side)
        pts[:, 1] = (pts[:, 1] - top) * (self.img_size / side)

        heatmap = generate_component_heatmaps(self.img_size, pts, sigma=1.0)
        # Native LR then bicubic upsample to HR size (space-to-depth head expects 128×128).
        lr_native = img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_img = lr_native.resize((self.img_size, self.img_size), Image.BICUBIC)

        lr = np.asarray(lr_img, dtype=np.float32)
        hr = np.asarray(img, dtype=np.float32)
        if self.augment:
            lr, hr, heatmap = augment_sample(lr, hr, heatmap)

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            if t.ndim == 3:
                return t.permute(2, 0, 1)
            return t

        return {
            "lr": to_tensor(lr),
            "hr": to_tensor(hr),
            "heatmap": torch.from_numpy(np.ascontiguousarray(heatmap)).float(),
            "name": name,
        }


def build_celeba_loader(cfg: dict, split: str):
    from torch.utils.data import DataLoader

    data = cfg["data"]
    max_key = {"train": "max_train_samples", "val": "max_val_samples", "test": "max_test_samples"}[split]
    max_samples = data.get(max_key)
    ds = CelebAFSRDataset(
        root=data["root"],
        split=split,
        img_size=int(cfg["model"]["img_size"]),
        scale=int(cfg["model"]["scale"]),
        max_samples=int(max_samples) if max_samples is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        augment=bool(data.get("augment", False)),
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
