"""Facial component heatmaps HHR (Sec. IV-B, 5 groups from landmarks)."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _gaussian_2d(size: int, center: tuple[float, float], sigma: float = 1.0) -> np.ndarray:
    x_range = np.arange(size, dtype=np.float32)
    y_range = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x_range, y_range)
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)


def generate_component_heatmaps(
    size: int,
    landmarks: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return ``(5, H, W)`` heatmaps: left eye, right eye, nose, mouth, jawline."""
    heatmaps = np.zeros((5, size, size), dtype=np.float32)
    if landmarks.shape[0] >= 5:
        # CelebA-style 5 points: leye, reye, nose, lmouth, rmouth
        heatmaps[0] = _gaussian_2d(size, (landmarks[0, 0], landmarks[0, 1]), sigma)
        heatmaps[1] = _gaussian_2d(size, (landmarks[1, 0], landmarks[1, 1]), sigma)
        heatmaps[2] = _gaussian_2d(size, (landmarks[2, 0], landmarks[2, 1]), sigma)
        mouth = (landmarks[3] + landmarks[4]) / 2.0
        heatmaps[3] = _gaussian_2d(size, (mouth[0], mouth[1]), sigma * 1.2)
        cx = landmarks[:, 0].mean()
        cy = landmarks[:, 1].mean()
        rx = max(8.0, (landmarks[:, 0].max() - landmarks[:, 0].min()) * 0.95)
        ry = max(10.0, (landmarks[:, 1].max() - landmarks[:, 1].min()) * 1.25)
        yy, xx = np.mgrid[0:size, 0:size]
        jaw = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        heatmaps[4] = jaw.astype(np.float32)
        return heatmaps

    # Dense landmarks (Helen 194-pt or OpenFace-style 68-pt).
    # Channel order matches paper / upstream: left eye, right eye, nose, mouth, jawline.
    pts = landmarks.astype(np.float32)
    n = len(pts)
    if n >= 68 and n < 100:
        # OpenFace 68 (upstream dataset_landmark.py grouping)
        groups: list[Iterable[int]] = [
            range(36, 42),  # left eye
            range(42, 48),  # right eye
            range(27, 36),  # nose
            range(48, 68),  # mouth
            range(0, 27),  # jaw / silhouette
        ]
    else:
        # Helen 194-pt approximate facial-component bands
        groups = [
            range(114, min(134, n)),  # left eye
            range(134, min(154, n)),  # right eye
            range(41, min(58, n)),  # nose
            range(58, min(114, n)),  # mouth
            range(0, min(41, n)),  # jawline
        ]
    for gi, idxs in enumerate(groups):
        for i in idxs:
            if i < n:
                heatmaps[gi] = np.maximum(
                    heatmaps[gi],
                    _gaussian_2d(size, (pts[i, 0], pts[i, 1]), sigma),
                )
    return heatmaps


def augment_sample(
    lr: np.ndarray,
    hr: np.ndarray,
    heatmap: np.ndarray,
    hflip: bool = True,
    rot: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helen / train augmentation (90/180/270° + horizontal flip)."""
    if hflip and np.random.random() > 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()
        heatmap = heatmap[:, :, ::-1].copy()
    if rot:
        r = np.random.random()
        k = 0
        if r > 0.75:
            k = 1
        elif r > 0.5:
            k = 2
        elif r > 0.25:
            k = 3
        if k:
            lr = np.rot90(lr, k=k, axes=(0, 1)).copy()
            hr = np.rot90(hr, k=k, axes=(0, 1)).copy()
            heatmap = np.rot90(heatmap, k=k, axes=(1, 2)).copy()
    return lr, hr, heatmap
