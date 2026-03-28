"""
data_utils.py — Data Loading & Preprocessing
===============================================
Shared data pipeline for the quark/gluon jet image dataset.

Supports HDF5 format (N, 125, 125, 3) channels-last → (N, 3, 125, 125).
Active repo convention treats the detector channels as:
  channel 0 = Tracks
  channel 1 = ECAL
  channel 2 = HCAL
Labels: y=0 quark, y=1 gluon.

Provides:
  - HDF5 dataset loading with lazy truncation
  - PyTorch Dataset wrapper with global normalization
  - Deterministic train/val/test splitting
  - Image → point cloud conversion for graph-based tasks
"""

import os
import logging
from typing import Tuple, Optional

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .config import DATA_DIR, IMAGE_SIZE, N_CHANNELS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# HDF5 Dataset Loading
# ─────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: str = DATA_DIR,
    max_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the quark/gluon jet image dataset from HDF5.

    Searches for the canonical HDF5 file in `data_dir` and one level up
    (for flexible directory layouts, e.g. Colab vs. local).

    Args:
        data_dir:   Directory to search for the dataset.
        max_events: Optional cap on number of events (for rapid prototyping).

    Returns:
        X: float32 array of shape (N, 3, 125, 125) — channels-first.
        y: int64 array of shape (N,) — 0=quark, 1=gluon.

    Raises:
        FileNotFoundError: If no HDF5 file is found in search paths.
    """
    h5_name = "quark-gluon_data-set_n139306.hdf5"
    search_paths = [
        os.path.join(data_dir, h5_name),
        os.path.join(os.path.dirname(os.path.abspath(data_dir)), h5_name),
        # Also check project root (common in Colab setups)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), h5_name),
    ]

    h5_path = None
    for path in search_paths:
        if os.path.exists(path):
            h5_path = path
            break

    if h5_path is None:
        raise FileNotFoundError(
            f"Dataset '{h5_name}' not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
        )

    logger.info("Loading dataset from: %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        n_total = f["X_jets"].shape[0]
        limit = min(max_events, n_total) if max_events else n_total

        X = f["X_jets"][:limit].astype(np.float32)   # (N, 125, 125, 3)
        y = f["y"][:limit].astype(np.int64)

    # Channels-last → channels-first for PyTorch Conv2d
    X = np.transpose(X, (0, 3, 1, 2))  # (N, 3, 125, 125)

    n_quark = int((y == 0).sum())
    n_gluon = int((y == 1).sum())
    logger.info(
        "Loaded %s events — shape %s | quark=%s, gluon=%s",
        f"{limit:,}", X.shape, f"{n_quark:,}", f"{n_gluon:,}"
    )
    return X, y


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

def compute_channel_scales(
    X: np.ndarray,
    use_log1p: bool = True,
) -> np.ndarray:
    """
    Compute per-channel normalisation scales from a reference split.

    Scaling is intentionally channel-wise so Tracks/ECAL/HCAL
    dominating one another.
    """
    X_proc = np.log1p(X) if use_log1p else X
    scales = X_proc.max(axis=(0, 2, 3)).astype(np.float32)
    return np.maximum(scales, 1e-8)


def normalize_channels(
    X: np.ndarray,
    channel_scales: np.ndarray,
    use_log1p: bool = True,
) -> np.ndarray:
    """Apply optional log1p and per-channel [0, 1] scaling."""
    X_proc = np.log1p(X) if use_log1p else X
    scales = np.asarray(channel_scales, dtype=np.float32).reshape(1, -1, 1, 1)
    return (X_proc / scales).astype(np.float32)


class JetImageDataset(Dataset):
    """
    PyTorch Dataset for jet images with optional log1p and per-channel scaling.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channel_scales: Optional[np.ndarray] = None,
        use_log1p: bool = True,
    ) -> None:
        if channel_scales is None:
            channel_scales = compute_channel_scales(X, use_log1p=use_log1p)
        self.channel_scales = np.asarray(channel_scales, dtype=np.float32)
        self.use_log1p = use_log1p
        self.X = torch.from_numpy(
            normalize_channels(X, self.channel_scales, use_log1p=self.use_log1p)
        ).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# Train / Val / Test Splitting
# ─────────────────────────────────────────────────────────────

def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create deterministic index splits for train/val/test.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    logger.info(
        "Split — train: %s | val: %s | test: %s",
        f"{len(train_idx):,}", f"{len(val_idx):,}", f"{len(test_idx):,}"
    )
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────
# Image → Point Cloud Conversion (for GNN Tasks)
# ─────────────────────────────────────────────────────────────

def image_to_pointcloud(
    img: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert a single (3, 125, 125) jet image to a point cloud representation.

    Each pixel position (η_i, φ_j) where at least one channel exceeds the
    threshold becomes a "particle" node with 5D feature vector.

    The (η, φ) coordinates are the detector pseudorapidity and azimuthal
    angle, mapped from pixel indices and normalized to [-1, 1].

    Args:
        img:       Single jet image of shape (3, 125, 125).
        threshold: Minimum total energy to consider a pixel active.

    Returns:
        points: array of shape (M, 5) where columns are
                [η_norm, φ_norm, E_Tracks, E_ECAL, E_HCAL].
                If no active pixels, returns a single zero-padded node.
    """
    assert img.shape == (N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), \
        f"Expected ({N_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}), got {img.shape}"

    ch_tracks, ch_ecal, ch_hcal = img[0], img[1], img[2]

    # Active pixel mask: any channel with energy above threshold
    total_energy = np.abs(ch_tracks) + np.abs(ch_ecal) + np.abs(ch_hcal)
    mask = total_energy > threshold
    eta_idx, phi_idx = np.where(mask)

    if len(eta_idx) == 0:
        # Return single zero-padded node to prevent empty graphs
        return np.zeros((1, 5), dtype=np.float32)

    # Normalize pixel coordinates to [-1, 1] range
    eta_norm = (eta_idx / (IMAGE_SIZE - 1)) * 2.0 - 1.0
    phi_norm = (phi_idx / (IMAGE_SIZE - 1)) * 2.0 - 1.0

    e_tracks = ch_tracks[eta_idx, phi_idx]
    e_ecal = ch_ecal[eta_idx, phi_idx]
    e_hcal = ch_hcal[eta_idx, phi_idx]

    points = np.column_stack(
        [eta_norm, phi_norm, e_tracks, e_ecal, e_hcal]
    ).astype(np.float32)

    return points
