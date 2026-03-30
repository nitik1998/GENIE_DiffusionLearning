"""
task1_autoencoder.py
====================
Task 1 autoencoder pipeline for jet image reconstruction,
kept modular inside the repo and using the requested channel order:
Tracks, ECAL, HCAL.
"""

import os
import sys
import argparse
import copy
import json
import shutil
import time
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Resolve imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    setup_logging, set_seed, get_device, ensure_dirs,
    OUTPUT_DIR, get_auto_batch_size,
    DataConfig,
)
from src.data_utils import load_dataset
from src.metrics import reconstruction_summary, sparse_reconstruction_metrics
from src.models.autoencoder import JetVAE
from src.experiment_tracker import log_experiment, save_run_metrics

logger = setup_logging("task1")

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 0
TASK1_CHANNEL_NAMES = ("Tracks", "ECAL", "HCAL")

EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "task1_autoencoder": {
        "preprocess_mode": "detector_reference",
        "recon_loss": "mse",
        "recon_mix_alpha": 0.7,
        "beta_max": 1e-2,
        "kl_warmup_epochs": 100,
        "nonzero_weight": 4.0,
        "use_transpose_decoder": True,
        "latent_dim": 256,
        "lr": 1e-3,
        "scheduler_patience": 5,
        "epochs_override": 200,
        "batch_size_override": 0,
        "variational": True,
        "boost_channel": 0,
        "boost_factor": 1.5,
        "decoder_batchnorm": True,
        "output_bias_init": 0.0,
        "eval_thresholds": [0.05],
        "eval_use_mean": False,
        "optimizer": "adam",
        "training_recipe": "reference_vae",
    },
}


def normalize_preprocess_mode(preprocess_mode: str) -> str:
    alias_map = {
        "detector_reference": "detector_reference",
        "global_logmax": "global_logmax",
        "robust_log_channelwise": "robust_log_channelwise",
    }
    if preprocess_mode not in alias_map:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
    return alias_map[preprocess_mode]





class Task1Dataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        preprocessor_params: List[Dict[str, float]],
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = torch.from_numpy(y[self.indices]).long()
        # Pre-compute ALL preprocessing once → __getitem__ becomes a pure tensor lookup
        raw = X[self.indices]  # shape: [N, C, H, W]
        self.raw_viz = raw[:5].copy()  # Save a tiny subset for plotting
        processed = apply_task1_preprocessor(raw, preprocessor_params)  # numpy
        self.data = torch.from_numpy(processed).float().contiguous(memory_format=torch.channels_last)  # H100-optimized layout
        del raw, processed  # Free memory immediately

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────


def score_metrics(metrics: Dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("active_iou", -1.0)),
        float(metrics.get("active_recall", -1.0)),
        -float(metrics.get("background_false_activation", 1e9)),
        -float(metrics.get("nonzero_mse", 1e9)),
        float(metrics.get("ssim", -1.0)),
    )


def fit_task1_preprocessor(
    X: np.ndarray,
    preprocess_mode: str,
    boost_channel: int = 0,
    boost_factor: float = 1.5,
) -> List[Dict[str, float]]:
    """
    Fit a Task 1 normalizer using the training split only.

    The detector-reference recipe follows the original Evaluation Test
    Reference preprocessing pipeline:
      - one special channel gets log/percentile scaling with a top-end boost
      - the remaining channels use mean/std scaling mapped into [0, 1]
    """
    preprocess_mode = normalize_preprocess_mode(preprocess_mode)
    params: List[Dict[str, float]] = []

    if preprocess_mode == "global_logmax":
        X_proc = np.log1p(X)
        scale = float(np.max(X_proc))
        for _ in range(X.shape[1]):
            params.append({"type": "global_logmax", "scale": max(scale, 1e-8)})
        return params

    for ch in range(X.shape[1]):
        channel = X[:, ch]

        if preprocess_mode == "detector_reference":
            special_channel = int(boost_channel)
            if ch == special_channel:
                transformed = np.log1p(channel * 10.0 + 1e-8) / 3.0
                transformed_nonzero = transformed[channel > 0]
                if transformed_nonzero.size:
                    p_low, p_high = np.percentile(transformed_nonzero, [1.0, 99.9])
                else:
                    p_low, p_high = 0.0, 1.0
                params.append({
                    "type": "detector_reference_log",
                    "p_low": float(p_low),
                    "scale": max(float(p_high - p_low), 1e-8),
                    "boost_threshold": 0.75,
                    "boost_factor": float(boost_factor),
                    "special_channel": special_channel,
                })
            else:
                std = float(np.std(channel))
                if std > 1e-10:
                    params.append({
                        "type": "detector_reference_standard",
                        "mean": float(np.mean(channel)),
                        "std_scale": max(3.0 * std, 1e-8),
                    })
                else:
                    params.append({"type": "detector_reference_identity"})
        elif preprocess_mode == "robust_log_channelwise":
            transformed = np.log1p(channel)
            transformed_nonzero = transformed[channel > 0]
            if transformed_nonzero.size:
                p_low, p_high = np.percentile(transformed_nonzero, [1.0, 99.9])
            else:
                p_low, p_high = 0.0, 1.0
            params.append(
                {
                    "type": "robust_log_channelwise",
                    "p_low": float(p_low),
                    "scale": max(float(p_high - p_low), 1e-8),
                }
            )
        else:
            raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")

    return params


def apply_task1_preprocessor(
    X: np.ndarray,
    params: List[Dict[str, float]],
) -> np.ndarray:
    X_cf = np.asarray(X, dtype=np.float32)
    out = np.zeros_like(X_cf, dtype=np.float32)

    for ch, spec in enumerate(params):
        channel = X_cf[:, ch]
        kind = spec["type"]

        if kind == "global_logmax":
            out[:, ch] = np.log1p(channel) / spec["scale"]
            continue

        if kind == "detector_reference_log":
            transformed = np.log1p(channel * 10.0 + 1e-8) / 3.0
            normalized = (transformed - spec["p_low"]) / spec["scale"]
            high_mask = normalized > spec["boost_threshold"]
            normalized[high_mask] = spec["boost_threshold"] + (
                normalized[high_mask] - spec["boost_threshold"]
            ) * spec["boost_factor"]
        elif kind == "detector_reference_standard":
            normalized = (channel - spec["mean"]) / spec["std_scale"]
            normalized = np.clip(normalized, -1.0, 1.0) * 0.5 + 0.5
        elif kind == "detector_reference_identity":
            normalized = channel
        elif kind == "robust_log_channelwise":
            transformed = np.log1p(channel)
            normalized = (transformed - spec["p_low"]) / spec["scale"]
        else:
            raise ValueError(f"Unsupported preprocessor type: {kind}")

        out[:, ch] = np.clip(normalized, 0.0, 1.0)

    return np.nan_to_num(out, copy=False)


def compute_kl_weight(
    epoch: int,
    total_epochs: int,
    beta_max: float,
    warmup_epochs: int | None = None,
) -> float:
    if warmup_epochs is None:
        warmup_epochs = max(1, min(10, total_epochs))
    else:
        warmup_epochs = max(1, int(warmup_epochs))
    return beta_max * min(epoch / warmup_epochs, 1.0)


def reference_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference VAE loss:
      - weighted MSE summed across the full batch
      - KL divergence summed across the full batch
      - total = recon + beta * KL
    """
    importance_weights = 1.0 + x * 4.0
    diff_squared = (recon_x - x) ** 2
    recon = torch.sum(diff_squared * importance_weights)

    mu_f, lv_f = mu.float(), logvar.float()
    kld = -0.5 * torch.sum(1 + lv_f - mu_f.pow(2) - lv_f.exp())
    if torch.isnan(kld) or torch.isinf(kld):
        print("Warning: KL divergence is NaN or Inf, using zero instead", flush=True)
        kld = torch.zeros((), device=recon_x.device, dtype=recon_x.dtype)

    total = recon + beta * kld
    return total, recon, kld


@torch.no_grad()
def reconstruct_with_mode(model: nn.Module, imgs: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
    if hasattr(model, "reconstruct"):
        return model.reconstruct(imgs, use_mean=use_mean)
    mu, _, _ = model.encode(imgs)
    return model.decode(mu)




def train_epoch_reference_style(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    scaler: torch.amp.GradScaler = None,
) -> tuple[float, float, float, int]:
    model.train()
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    batch_count = 0

    use_amp = device.type == "cuda" and scaler is not None
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            recons, mu, logvar = model(imgs)
            loss, recon, kld = reference_vae_loss(recons, imgs, mu, logvar, beta=beta)
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epoch_loss += loss.item()
        epoch_recon_loss += recon.item()
        epoch_kl_loss += kld.item()
        batch_count += 1

    return epoch_loss, epoch_recon_loss, epoch_kl_loss, batch_count


@torch.no_grad()
def eval_epoch_reference_style(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float, int]:
    model.eval()
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    batch_count = 0

    use_amp = device.type == "cuda"
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=use_amp):
            recons, mu, logvar = model(imgs)
            loss, recon, kld = reference_vae_loss(recons, imgs, mu, logvar, beta=beta)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            epoch_loss += loss.item()
            epoch_recon_loss += recon.item()
            epoch_kl_loss += kld.item()
            batch_count += 1

    return epoch_loss, epoch_recon_loss, epoch_kl_loss, batch_count


@torch.no_grad()
def collect_loader_reconstructions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_orig, all_recon = [], []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        recons = reconstruct_with_mode(model, imgs, use_mean=use_mean)
        all_orig.append(imgs.cpu())
        all_recon.append(recons.cpu())
    return torch.cat(all_orig), torch.cat(all_recon)


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_reconstructions(
    model: nn.Module,
    dataset: Task1Dataset,
    device: torch.device,
    out_dir: str,
    n_show: int = 8,
    use_mean: bool = True,
) -> None:
    """Side-by-side comparison: original vs reconstructed for each channel."""
    model.eval()
    n_show = min(n_show, len(dataset))
    imgs = torch.stack([dataset[i][0] for i in range(n_show)]).to(device)

    with torch.no_grad():
        recons = reconstruct_with_mode(model, imgs, use_mean=use_mean)
        recons = recons.cpu().numpy()
    originals = imgs.cpu().numpy()

    fig, axes = plt.subplots(n_show, 6, figsize=(20, n_show * 2.5), squeeze=False)
    fig.suptitle(
        "Task 1 — Autoencoder Reconstruction (Original | Reconstructed)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for row in range(n_show):
        for ch in range(3):
            # Use per-channel adaptive vmin = median so background floor appears dark
            orig_ch = originals[row, ch]
            recon_ch = recons[row, ch]
            bg = float(np.median(orig_ch))
            vmax = max(float(orig_ch.max()), float(recon_ch.max()), bg + 1e-6)
            axes[row, ch].imshow(orig_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 3].imshow(recon_ch, cmap="hot", vmin=bg, vmax=vmax)

            if row == 0:
                axes[row, ch].set_title(f"Original {TASK1_CHANNEL_NAMES[ch]}", fontsize=10)
                axes[row, ch + 3].set_title(f"Reconstructed {TASK1_CHANNEL_NAMES[ch]}", fontsize=10)

            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "original_vs_reconstructed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reconstructions → %s", path)


def _get_preview_arrays(
    model: nn.Module,
    dataset: Task1Dataset,
    device: torch.device,
    n_show: int = 8,
    use_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_show = min(n_show, len(dataset))
    imgs = torch.stack([dataset[i][0] for i in range(n_show)]).to(device)
    with torch.no_grad():
        recons = reconstruct_with_mode(model, imgs, use_mean=use_mean)
    return imgs.cpu().numpy(), recons.cpu().numpy()


def plot_reconstructions_with_error(
    model: nn.Module,
    dataset: Task1Dataset,
    device: torch.device,
    out_dir: str,
    n_show: int = 6,
    use_mean: bool = True,
) -> None:
    originals, recons = _get_preview_arrays(model, dataset, device, n_show=n_show, use_mean=use_mean)
    n_rows = originals.shape[0]
    errors = np.abs(originals - recons)

    fig, axes = plt.subplots(n_rows, 9, figsize=(28, n_rows * 2.4), squeeze=False)
    fig.suptitle(
        "Task 1 — Original | Reconstructed | Absolute Error",
        fontsize=14, fontweight="bold", y=1.01,
    )
    for row in range(n_rows):
        for ch in range(3):
            orig_ch = originals[row, ch]
            recon_ch = recons[row, ch]
            bg = float(np.median(orig_ch))
            vmax = max(float(orig_ch.max()), float(recon_ch.max()), bg + 1e-6)
            evmax = float(np.percentile(errors[row, ch], 99.5)) if np.any(errors[row, ch] > 0) else 1e-6
            axes[row, ch].imshow(orig_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 3].imshow(recon_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 6].imshow(errors[row, ch], cmap="magma", vmin=0, vmax=evmax)
            if row == 0:
                axes[row, ch].set_title(f"Original {TASK1_CHANNEL_NAMES[ch]}", fontsize=9)
                axes[row, ch + 3].set_title(f"Reconstructed {TASK1_CHANNEL_NAMES[ch]}", fontsize=9)
                axes[row, ch + 6].set_title(f"Abs Error {TASK1_CHANNEL_NAMES[ch]}", fontsize=9)
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")
            axes[row, ch + 6].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "original_vs_reconstructed_vs_abs_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reconstruction/error comparison → %s", path)


def plot_sparse_diagnostics(
    model: nn.Module,
    dataset: Task1Dataset,
    device: torch.device,
    out_dir: str,
    n_show: int = 6,
    threshold: float = 0.05,
    use_mean: bool = True,
) -> None:
    originals, recons = _get_preview_arrays(model, dataset, device, n_show=n_show, use_mean=use_mean)
    n_rows = originals.shape[0]
    # Per-channel background-aware mask: signal = pixels above the median background floor
    true_mask = np.zeros_like(originals, dtype=bool)
    for ch in range(3):
        ch_median = float(np.median(originals[:, ch]))
        true_mask[:, ch] = originals[:, ch] > (ch_median + 0.01)
    pred_mask = recons > threshold

    fig, axes = plt.subplots(n_rows, 6, figsize=(18, n_rows * 2.2), squeeze=False)
    fig.suptitle("Task 1 — Sparse Diagnostics", fontsize=14, fontweight="bold", y=1.01)
    for row in range(n_rows):
        for ch in range(3):
            axes[row, ch].imshow(true_mask[row, ch], cmap="gray", vmin=0, vmax=1)
            axes[row, ch + 3].imshow(pred_mask[row, ch], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, ch].set_title(f"True Mask {TASK1_CHANNEL_NAMES[ch]}", fontsize=9)
                axes[row, ch + 3].set_title(f"Pred Mask {TASK1_CHANNEL_NAMES[ch]}", fontsize=9)
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")
    plt.tight_layout()
    mask_path = os.path.join(out_dir, "true_mask_vs_pred_mask.png")
    plt.savefig(mask_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved mask diagnostics → %s", mask_path)

    fig, axes = plt.subplots(n_rows, 3, figsize=(11, n_rows * 2.2), squeeze=False)
    fig.suptitle("Task 1 — Sparse Reconstruction Diagnostics", fontsize=14, fontweight="bold", y=1.01)
    for row in range(n_rows):
        combined_orig = originals[row].sum(axis=0)
        combined_recon = recons[row].sum(axis=0)
        combined_error = np.abs(combined_orig - combined_recon)
        bg = float(np.median(combined_orig))
        vmax = max(float(combined_orig.max()), float(combined_recon.max()), bg + 1e-6)
        evmax = float(np.percentile(combined_error, 99.5)) if np.any(combined_error > 0) else 1e-6
        axes[row, 0].imshow(combined_orig, cmap="hot", vmin=bg, vmax=vmax)
        axes[row, 1].imshow(combined_recon, cmap="hot", vmin=bg, vmax=vmax)
        axes[row, 2].imshow(combined_error, cmap="magma", vmin=0, vmax=max(evmax, 1e-6))
        if row == 0:
            axes[row, 0].set_title("Combined Original", fontsize=9)
            axes[row, 1].set_title("Combined Recon", fontsize=9)
            axes[row, 2].set_title("Combined Abs Error", fontsize=9)
        for col in range(3):
            axes[row, col].axis("off")
    plt.tight_layout()
    sparse_path = os.path.join(out_dir, "sparse_diagnostics.png")
    plt.savefig(sparse_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sparse diagnostics → %s", sparse_path)


def plot_normalized_inputs(
    dataset: Task1Dataset,
    out_dir: str,
    n_show: int = 3,
) -> None:
    """Save a side-by-side view: raw detector images vs. normalized model inputs."""
    n_show = min(n_show, len(dataset))
    norm_imgs = torch.stack([dataset[i][0] for i in range(n_show)]).numpy()
    raw_imgs = dataset.raw_viz[:n_show]

    fig, axes = plt.subplots(n_show, 6, figsize=(22, n_show * 3), squeeze=False)
    fig.suptitle(
        "Task 1 — Raw Detector Images vs. Normalized Model Inputs",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for row in range(n_show):
        for ch in range(3):
            # Columns 0-2: Raw data (vmax = per-channel max)
            raw_ch = raw_imgs[row, ch]
            vmax_raw = max(float(raw_ch.max()), 1e-6)
            axes[row, ch].imshow(raw_ch, cmap="hot", vmin=0, vmax=vmax_raw)
            axes[row, ch].axis("off")
            if row == 0:
                axes[row, ch].set_title(f"Raw {TASK1_CHANNEL_NAMES[ch]}", fontsize=10)

            # Columns 3-5: Normalized data
            # Use median as vmin so the background floor (≈0.5 for mean-centered channels)
            # appears dark, letting active signal stand out clearly.
            norm_ch = norm_imgs[row, ch]
            bg = float(np.median(norm_ch))
            axes[row, ch + 3].imshow(norm_ch, cmap="hot", vmin=bg, vmax=max(float(norm_ch.max()), bg + 1e-6))
            axes[row, ch + 3].axis("off")
            if row == 0:
                axes[row, ch + 3].set_title(f"Normalized {TASK1_CHANNEL_NAMES[ch]}", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "normalized_input_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved normalized inputs → %s", path)


def save_task1_reconstruction_samples(
    original_batch: torch.Tensor,
    reconstructed_batch: torch.Tensor,
    sample_indices: List[int],
    out_dir: str,
    filename_prefix: str = "",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    channel_names = list(TASK1_CHANNEL_NAMES)

    for idx in sample_indices:
        if idx >= original_batch.shape[0]:
            continue
        original = original_batch[idx].detach().cpu().numpy()
        recon = reconstructed_batch[idx].detach().cpu().numpy()
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        fig.suptitle(f"Jet Event {idx} - Original vs. Reconstructed", fontsize=16)
        for ch in range(3):
            vmin = min(float(np.min(original[ch])), float(np.min(recon[ch])))
            vmax = max(float(np.max(original[ch])), float(np.max(recon[ch])))
            im1 = axes[ch, 0].imshow(original[ch], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[ch, 0].set_title(f"Original - {channel_names[ch]}")
            axes[ch, 0].axis("off")
            fig.colorbar(im1, ax=axes[ch, 0], fraction=0.046, pad=0.04)

            im2 = axes[ch, 1].imshow(recon[ch], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[ch, 1].set_title(f"Reconstructed - {channel_names[ch]}")
            axes[ch, 1].axis("off")
            fig.colorbar(im2, ax=axes[ch, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        sample_path = os.path.join(out_dir, f"{filename_prefix}vae_reconstruction_sample_{idx}.png")
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        plt.close()





def plot_training_curves_reference_style(
    train_losses: List[float],
    val_losses: List[float],
    recon_losses: List[float],
    kl_losses: List[float],
    out_dir: str,
) -> None:
    train_losses = [x for x in train_losses if not (np.isnan(x) or np.isinf(x))]
    val_losses = [x for x in val_losses if not (np.isnan(x) or np.isinf(x))]
    recon_losses = [x for x in recon_losses if not (np.isnan(x) or np.isinf(x))]
    kl_losses = [x for x in kl_losses if not (np.isnan(x) or np.isinf(x))]

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_losses, "b-", label="Train Loss")
    if val_losses:
        plt.plot(val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax1 = plt.subplot(1, 2, 2)
    if recon_losses:
        ax1.semilogy(recon_losses, "g-", linewidth=2, label="Reconstruction Loss")
        ax1.set_ylabel("Reconstruction Loss", color="g")
        ax1.tick_params(axis="y", labelcolor="g")
    if kl_losses:
        ax2 = ax1.twinx()
        ax2.semilogy(kl_losses, "m-", linewidth=2, label="KL Divergence Loss")
        ax2.set_ylabel("KL Divergence Loss", color="m")
        ax2.tick_params(axis="y", labelcolor="m")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Component Losses (log scale)")
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if kl_losses else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reference-style training curves → %s", path)


def make_reference_split_indices(y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    test_idx = np.array(val_idx, copy=True)
    logger.info(
        "Reference split — train: %s | val/test: %s",
        f"{len(train_idx):,}",
        f"{len(val_idx):,}",
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def plot_threshold_sweep(
    threshold_metrics: List[Dict[str, float]],
    out_dir: str,
) -> None:
    thresholds = [m["pred_threshold"] for m in threshold_metrics]
    precision = [m["active_precision"] for m in threshold_metrics]
    recall = [m["active_recall"] for m in threshold_metrics]
    iou = [m["active_iou"] for m in threshold_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, marker="o", lw=2, label="Precision")
    ax.plot(thresholds, recall, marker="o", lw=2, label="Recall")
    ax.plot(thresholds, iou, marker="o", lw=2, label="IoU")
    ax.set_xlabel("Prediction Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Task 1 — Sparse Support Threshold Sweep", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, "threshold_sweep.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold sweep → %s", path)


def plot_sparse_vs_full_metrics(metrics: Dict[str, float], out_dir: str) -> None:
    labels = ["Full MSE", "Nonzero MSE", "Background False Activation"]
    values = [
        float(metrics.get("mse_overall", 0.0)),
        float(metrics.get("nonzero_mse", 0.0)),
        float(metrics.get("background_false_activation", 0.0)),
    ]
    colors = ["#2563EB", "#DC2626", "#059669"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.set_title("Task 1 — Full vs Sparse Reconstruction Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)
    path = os.path.join(out_dir, "sparse_vs_full_metric_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sparse/full comparison → %s", path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def resolve_experiment_settings(args: argparse.Namespace, exp_name: str) -> Dict[str, Any]:
    base = {
        "preprocess_mode": args.preprocess_mode,
        "recon_loss": args.recon_loss,
        "recon_mix_alpha": args.recon_mix_alpha,
        "beta_max": args.beta,
        "nonzero_weight": args.nonzero_weight,
        "use_transpose_decoder": args.use_transpose_decoder,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "scheduler_patience": args.scheduler_patience,
        "variational": args.variational,
        "boost_channel": args.boost_channel,
        "boost_factor": args.boost_factor,
        "decoder_batchnorm": args.decoder_batchnorm,
        "output_bias_init": args.output_bias_init,
        "eval_thresholds": [0.05],
        "eval_use_mean": True,
        "optimizer": "adamw",
    }
    matched_preset = None
    if exp_name in EXPERIMENT_PRESETS:
        matched_preset = exp_name
    else:
        for preset_name in sorted(EXPERIMENT_PRESETS.keys(), key=len, reverse=True):
            if exp_name.startswith(f"{preset_name}_"):
                matched_preset = preset_name
                break
    if matched_preset is not None:
        base.update(EXPERIMENT_PRESETS[matched_preset])
    return base


def build_datasets(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    settings: Dict[str, Any],
) -> tuple[Task1Dataset, Task1Dataset, Task1Dataset, List[Dict[str, float]]]:
    preprocessor_params = fit_task1_preprocessor(
        X[train_idx],
        preprocess_mode=settings["preprocess_mode"],
        boost_channel=settings["boost_channel"],
        boost_factor=settings["boost_factor"],
    )
    train_ds = Task1Dataset(X, y, train_idx, preprocessor_params)
    val_ds = Task1Dataset(X, y, val_idx, preprocessor_params)
    test_ds = Task1Dataset(X, y, test_idx, preprocessor_params)
    return train_ds, val_ds, test_ds, preprocessor_params


def save_checkpoint(
    model: nn.Module,
    out_dir: str,
    run_params: Dict[str, Any],
    metrics: Dict[str, float],
    preprocessor_params: List[Dict[str, float]],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "params": run_params,
        "metrics": metrics,
        "preprocessor_params": preprocessor_params,
    }
    ckpt_path = os.path.join(out_dir, "model_checkpoint.pt")
    torch.save(payload, ckpt_path)
    torch.save(payload, os.path.join(out_dir, "checkpoint.pt"))
    logger.info("Saved checkpoint → %s", ckpt_path)


def save_config(out_dir: str, run_params: Dict[str, Any]) -> None:
    path = os.path.join(out_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)


def save_summary(out_dir: str, exp_name: str, run_params: Dict[str, Any], metrics: Dict[str, float]) -> None:
    lines = [f"# {exp_name}", "", "## Positioning", ""]
    lines.extend([
        "- Reference VAE baseline for sparse detector reconstruction.",
        f"- Channel order: {', '.join(TASK1_CHANNEL_NAMES)}.",
        "- Use this as the stable Task 1 baseline before trying new ideas.",
        "",
        "## Config",
        "",
    ])
    for key, value in sorted(run_params.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Metrics", ""])
    for key, value in sorted(metrics.items()):
        lines.append(f"- `{key}`: `{value:.6f}`" if isinstance(value, (int, float)) else f"- `{key}`: `{value}`")
    content = "\n".join(lines) + "\n"
    for filename in ("summary.md", "short_summary.md"):
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def update_leaderboard(
    root_dir: str,
    exp_name: str,
    run_params: Dict[str, Any],
    metrics: Dict[str, float],
) -> List[Dict[str, Any]]:
    def row_metric(row: Dict[str, Any], score_key: str, metric_key: str, default: float) -> float:
        value = row.get(score_key, row.get(metric_key, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    entry: Dict[str, Any] = {
        "experiment": exp_name,
        "score_active_iou": float(metrics.get("active_iou", 0.0)),
        "score_active_recall": float(metrics.get("active_recall", 0.0)),
        "score_background_false_activation": float(metrics.get("background_false_activation", 1.0)),
        "score_nonzero_mse": float(metrics.get("nonzero_mse", 1.0)),
        "score_mse_overall": float(metrics.get("mse_overall", 1.0)),
        "score_psnr_db": float(metrics.get("psnr_db", 0.0)),
        "score_ssim": float(metrics.get("ssim", 0.0)),
        "score_tuple": list(score_metrics(metrics)),
        **{k: metrics[k] for k in metrics},
        "batch_size": run_params.get("batch_size"),
        "epochs": run_params.get("epochs"),
        "latent_dim": run_params.get("latent_dim"),
        "variational": run_params.get("variational"),
        "use_transpose_decoder": run_params.get("use_transpose_decoder"),
        "recon_loss": run_params.get("recon_loss"),
    }

    csv_path = os.path.join(root_dir, "leaderboard.csv")
    rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        rows = [
            row for row in rows
            if row.get("experiment")
            and os.path.isdir(os.path.join(root_dir, str(row.get("experiment"))))
        ]
        rows = [row for row in rows if row.get("experiment") != exp_name]
    rows.append(entry)
    rows.sort(
        key=lambda row: (
            row_metric(row, "score_active_iou", "active_iou", 0.0),
            row_metric(row, "score_active_recall", "active_recall", 0.0),
            -row_metric(row, "score_background_false_activation", "background_false_activation", 1.0),
            -row_metric(row, "score_nonzero_mse", "nonzero_mse", 1.0),
            row_metric(row, "score_ssim", "ssim", 0.0),
        ),
        reverse=True,
    )

    import csv
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = os.path.join(root_dir, "leaderboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Task 1 Leaderboard\n\n")
        f.write("| Rank | Experiment | active_iou | active_recall | nonzero_mse | background_false_activation | mse_overall | psnr_db | ssim |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for idx, row in enumerate(rows, start=1):
            f.write(
                f"| {idx} | {row['experiment']} | "
                f"{row_metric(row, 'score_active_iou', 'active_iou', 0.0):.6f} | "
                f"{row_metric(row, 'score_active_recall', 'active_recall', 0.0):.6f} | "
                f"{row_metric(row, 'score_nonzero_mse', 'nonzero_mse', 1.0):.6f} | "
                f"{row_metric(row, 'score_background_false_activation', 'background_false_activation', 1.0):.6f} | "
                f"{row_metric(row, 'score_mse_overall', 'mse_overall', 1.0):.6f} | "
                f"{row_metric(row, 'score_psnr_db', 'psnr_db', 0.0):.6f} | "
                f"{row_metric(row, 'score_ssim', 'ssim', 0.0):.6f} |\n"
            )
    return rows


def update_context(root_dir: str, rows: List[Dict[str, Any]], notes: List[str]) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "context.md")
    best = rows[0] if rows else None
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Task 1 Context\n\n")
        f.write("## Root Issue\n\n")
        f.write("- The active repo Task 1 path had drifted away from the known-good reference VAE notebook and started collapsing.\n")
        f.write("- Matching the reference preprocessing, weighted MSE loss scaling, optimizer, and split behavior is the main stability requirement.\n")
        f.write("- Task 1 is now kept as a faithful notebook-equivalent baseline with the requested channel order.\n\n")
        f.write("## Final Training Path\n\n")
        f.write("- `task1_autoencoder` now follows the modularized reference VAE path.\n")
        f.write("- Keep transpose-decoder VAE training, weighted MSE reconstruction, reference preprocessing, sampled-latent evaluation, and channel-wise reconstruction evidence.\n")
        f.write("- Local and Colab runs share the same official script path and the same saved artifacts.\n\n")
        f.write("## Run Notes\n\n")
        for note in notes:
            f.write(f"- {note}\n")
        f.write("\n## Final Recommended Config\n\n")
        if best is not None:
            f.write(f"- Experiment: `{best['experiment']}`\n")
            f.write(f"- active_iou: `{float(best['score_active_iou']):.6f}`\n")
            f.write(f"- active_recall: `{float(best['score_active_recall']):.6f}`\n")
            f.write(f"- background_false_activation: `{float(best['score_background_false_activation']):.6f}`\n")
            f.write(f"- nonzero_mse: `{float(best['score_nonzero_mse']):.6f}`\n")
            f.write(f"- mse_overall: `{float(best.get('score_mse_overall', 1.0)):.6f}`\n")
            f.write(f"- psnr_db: `{float(best.get('score_psnr_db', 0.0)):.6f}`\n")
            f.write(f"- ssim: `{float(best['score_ssim']):.6f}`\n")


def save_best_artifacts(root_dir: str, best_exp_name: str, baseline_exp_name: str | None) -> None:
    best_dir = os.path.join(root_dir, best_exp_name)
    shutil.copy2(os.path.join(best_dir, "model_checkpoint.pt"), os.path.join(best_dir, "best_model.pt"))
    shutil.copy2(os.path.join(best_dir, "metrics.json"), os.path.join(best_dir, "best_metrics.json"))
    shutil.copy2(os.path.join(best_dir, "metrics.csv"), os.path.join(best_dir, "best_metrics.csv"))
    shutil.copy2(os.path.join(best_dir, "original_vs_reconstructed.png"), os.path.join(best_dir, "best_samples.png"))
    with open(os.path.join(best_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    yaml_path = os.path.join(best_dir, "best_config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        for key, value in sorted(config.items()):
            f.write(f"{key}: {value}\n")

    if baseline_exp_name:
        base_img = plt.imread(os.path.join(root_dir, baseline_exp_name, "original_vs_reconstructed.png"))
        best_img = plt.imread(os.path.join(best_dir, "original_vs_reconstructed.png"))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(base_img)
        axes[0].set_title(f"Baseline: {baseline_exp_name}")
        axes[0].axis("off")
        axes[1].imshow(best_img)
        axes[1].set_title(f"Best: {best_exp_name}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(best_dir, "baseline_vs_final_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()


def run_experiment(
    args: argparse.Namespace,
    exp_name: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> None:
    settings = resolve_experiment_settings(args, exp_name)
    out_dir = os.path.join(OUTPUT_DIR, "ae", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    if not args.force_rerun:
        metrics_path = os.path.join(out_dir, "metrics.json")
        checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
        if os.path.exists(metrics_path) and os.path.exists(checkpoint_path):
            logger.info("Skipping completed experiment %s; outputs already exist in %s", exp_name, out_dir)
            return {"exp_name": exp_name, "skipped": True}
    logger.info("Experiment: %s → %s", exp_name, out_dir)
    logger.info("Settings: %s", settings)


    train_idx, val_idx, test_idx = splits
    train_ds, val_ds, test_ds, preprocessor_params = build_datasets(
        X, y, train_idx, val_idx, test_idx, settings
    )
    # Free the original raw arrays to save ~30-50GB of RAM
    del X, y, splits
    import gc
    gc.collect()

    target_epochs = int(settings.get("epochs_override", args.epochs))
    if args.epochs != DEFAULT_EPOCHS:
        target_epochs = int(args.epochs)

    batch_size = int(settings.get("batch_size_override", args.batch_size))
    if args.batch_size != DEFAULT_BATCH_SIZE:
        batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = get_auto_batch_size(task_num=1)
        logger.info("Auto-scaled batch size to %d based on VRAM", batch_size)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_state = None
    model = None

    while True:
        try:
            pin = device.type == "cuda"
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')  # Enable TF32 on H100/A100
            # Data is pre-cached as tensors — num_workers=0 is fastest (avoids IPC)
            # and prevents Colab /dev/shm exhaustion crashes
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

            model = JetVAE(latent_dim=settings["latent_dim"]).to(device)
            if device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)  # H100 tensor core layout
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Reference VAE — %s trainable parameters", f"{n_params:,}")
            try:
                model = torch.compile(model)
                logger.info("torch.compile() enabled — expect warmup on first batch")
            except Exception as e:
                logger.warning("torch.compile() not available: %s (running uncompiled)", e)

            optimizer_name = str(settings.get("optimizer", "adamw")).lower()
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.7, patience=settings["scheduler_patience"], min_lr=1e-6
            )


            logger.info("Starting training for %d epochs (patience=%d, batch=%d)...", target_epochs, args.patience, batch_size)
            train_losses, val_losses = [], []
            recon_losses: List[float] = []
            kl_losses: List[float] = []
            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0

            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

            print(f"=== Starting training [{exp_name}] for {target_epochs} epochs ===", flush=True)
            for epoch in range(1, target_epochs + 1):
                epoch_start = time.time()
                print(f"E{epoch}/{target_epochs} starting...", end="\r", flush=True)
                beta = compute_kl_weight(
                    epoch,
                    target_epochs,
                    settings["beta_max"],
                    warmup_epochs=settings.get("kl_warmup_epochs"),
                )
                tr_total, tr_recon, tr_kl, tr_batches = train_epoch_reference_style(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    beta,
                    scaler=scaler,
                )
                n_train = len(train_ds)
                tr_loss = tr_total / n_train if tr_batches > 0 else float("inf")
                tr_recon_loss = tr_recon / n_train if tr_batches > 0 else float("inf")
                tr_kl_loss = tr_kl / n_train if tr_batches > 0 else float("inf")
                train_losses.append(tr_loss)
                recon_losses.append(tr_recon_loss)
                kl_losses.append(tr_kl_loss)

                # Validate every 3 epochs or on the last epoch to save time
                run_val = (epoch % 3 == 0) or (epoch == target_epochs) or (epoch <= 3)
                if run_val:
                    vl_total, vl_recon, vl_kl, vl_batches = eval_epoch_reference_style(
                        model,
                        val_loader,
                        device,
                        beta,
                    )
                    n_val = len(val_ds)
                    vl_loss = vl_total / n_val if vl_batches > 0 else float("inf")
                    vl_recon_loss = vl_recon / n_val if vl_batches > 0 else float("inf")
                    vl_kl_loss = vl_kl / n_val if vl_batches > 0 else float("inf")
                    scheduler.step(vl_loss)
                    val_losses.append(vl_loss)
                    if vl_loss < best_val_loss:
                        best_val_loss = vl_loss
                        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        patience_counter = 0
                        marker = " *"
                    else:
                        patience_counter += 1
                        marker = ""
                else:
                    vl_loss = val_losses[-1] if val_losses else float("inf")
                    vl_recon_loss = 0.0
                    vl_kl_loss = 0.0
                    marker = " (skip-val)"

                gap = vl_loss / (tr_loss + 1e-10)
                epoch_elapsed = time.time() - epoch_start
                print(f"E{epoch}/{target_epochs} [{epoch_elapsed:.1f}s] train={tr_loss:.2f} val={vl_loss:.2f} gap={gap:.2f}x lr={optimizer.param_groups[0]['lr']:.6f} kl_w={beta:.4f}{marker}", flush=True)

                if (epoch % 20 == 0) or epoch == target_epochs:
                    with torch.no_grad():
                        if len(val_loader) > 0:
                            sample_batch = next(iter(val_loader))[0].to(device)
                            reconstructed, _, _ = model(sample_batch)
                            sample_count = min(5, sample_batch.size(0))
                            save_task1_reconstruction_samples(
                                sample_batch,
                                reconstructed,
                                list(range(sample_count)),
                                out_dir=out_dir,
                                filename_prefix=f"epoch_{epoch}_",
                            )

                if patience_counter >= args.patience and args.patience > 0:
                    print(f"[{exp_name}] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)", flush=True)
                    break
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or device.type != "cuda" or batch_size <= 2:
                raise
            logger.warning("CUDA OOM for %s at batch size %d; retrying with batch size %d", exp_name, batch_size, batch_size // 2)
            torch.cuda.empty_cache()
            batch_size = max(2, batch_size // 2)

    if best_state is not None and model is not None:
        model.load_state_dict(best_state)
        print(f"[{exp_name}] Restored best model (Val Loss: {best_val_loss:.6f})", flush=True)

    eval_use_mean = bool(settings.get("eval_use_mean", True))
    if len(val_loader) > 0:

        with torch.no_grad():
            sample_batch = next(iter(val_loader))[0].to(device)
            reconstructed, _, _ = model(sample_batch)
            save_task1_reconstruction_samples(
                sample_batch,
                reconstructed,
                list(range(min(5, sample_batch.size(0)))),
                out_dir=out_dir,
                filename_prefix="final_",
            )
    orig_cat, recon_cat = collect_loader_reconstructions(model, test_loader, device, use_mean=eval_use_mean)
    metrics = reconstruction_summary(orig_cat, recon_cat)
    report_thresholds = sorted(set([0.02, 0.03, 0.05] + [float(t) for t in settings.get("eval_thresholds", [0.05])]))
    best_threshold = float(settings.get("eval_thresholds", [0.05])[0])
    threshold_metrics: List[Dict[str, float]] = []
    for threshold in report_thresholds:
        sweep_metrics = sparse_reconstruction_metrics(orig_cat, recon_cat, pred_threshold=float(threshold))
        threshold_metrics.append(sweep_metrics)
        if abs(float(threshold) - best_threshold) < 1e-9:
            metrics.update(sweep_metrics)
    metrics["best_val_loss"] = best_val_loss
    metrics["best_pred_threshold"] = best_threshold

    logger.info("═══ Test Set Evaluation (%s) ═══", exp_name)
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)

    run_params = {
        "epochs": target_epochs,
        "lr": settings["lr"],
        "batch_size": batch_size,
        "max_events": args.max_events,
        "seed": args.seed,
        **settings,
    }
    save_run_metrics(out_dir, metrics, run_params)

    plot_normalized_inputs(train_ds, out_dir)
    plot_training_curves_reference_style(train_losses, val_losses, recon_losses, kl_losses, out_dir)

    plot_reconstructions(model, val_ds, device, out_dir, use_mean=eval_use_mean)
    plot_reconstructions_with_error(model, val_ds, device, out_dir, use_mean=eval_use_mean)
    plot_sparse_diagnostics(model, val_ds, device, out_dir, threshold=float(best_threshold), use_mean=eval_use_mean)
    plot_threshold_sweep(threshold_metrics, out_dir)
    plot_sparse_vs_full_metrics(metrics, out_dir)
    save_checkpoint(model, out_dir, run_params, metrics, preprocessor_params)
    save_config(out_dir, run_params)
    save_summary(out_dir, exp_name, run_params, metrics)

    rows = update_leaderboard(os.path.join(OUTPUT_DIR, "ae"), exp_name, run_params, metrics)
    notes = [
        f"{exp_name}: batch_size={batch_size}, epochs={target_epochs}, latent_dim={settings['latent_dim']}, transpose_decoder={settings['use_transpose_decoder']}",
        f"{exp_name}: recon_loss={settings['recon_loss']}, weighting=1+4x, beta_max={settings['beta_max']}, optimizer={settings.get('optimizer', 'adamw')}",
        f"{exp_name}: eval_use_mean={eval_use_mean}, eval_thresholds={settings.get('eval_thresholds', [0.05])}, best_pred_threshold={best_threshold:.2f}",
    ]
    update_context(os.path.join(OUTPUT_DIR, "ae"), rows, notes)
    baseline_dir = None
    archive_root = os.path.join(OUTPUT_DIR, "ae", "archive")
    for candidate in ("task1_image_baseline_pre_reference_cleanup_20260327", "task1_image_baseline", "task1_current_baseline", "task1_final"):
        if os.path.exists(os.path.join(archive_root, candidate, "original_vs_reconstructed.png")):
            baseline_dir = os.path.join("archive", candidate)
            break
    save_best_artifacts(os.path.join(OUTPUT_DIR, "ae"), exp_name, baseline_dir)

    log_experiment("task1", exp_name, run_params, metrics, status="SUCCESS")
    logger.info("Task 1 complete. Outputs → %s", out_dir)
    return {
        "exp_name": exp_name,
        "metrics": metrics,
        "params": run_params,
    }


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    ensure_dirs()

    data_cfg = DataConfig(max_events=args.max_events)
    X, y = load_dataset(data_cfg.data_dir, data_cfg.max_events)
    splits = make_reference_split_indices(y, seed=args.seed)

    run_experiment(args, args.exp_name, X, y, splits, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 — Reference-aligned VAE autoencoder")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (0=auto scale based on VRAM)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (0=disabled)")
    parser.add_argument("--max-events", type=int, default=None, help="Limit events (None=all)")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun experiment even if outputs already exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="task1_autoencoder", help="Experiment name for output dir")
    parser.add_argument("--preprocess-mode", type=str, default="detector_reference", help="Preprocess mode: detector_reference, global_logmax, or robust_log_channelwise")
    parser.add_argument("--recon-loss", choices=["mse", "l1", "mixed"], default="l1")
    parser.add_argument("--recon-mix-alpha", type=float, default=0.7, help="Alpha for mixed loss: alpha*L1 + (1-alpha)*MSE")
    parser.add_argument("--beta", type=float, default=1e-4, help="KL weight for VAE training")
    parser.add_argument("--nonzero-weight", type=float, default=2.0, help="Extra reconstruction weight on active pixels")
    parser.add_argument("--use-transpose-decoder", action="store_true", help="Use ConvTranspose decoder")
    parser.add_argument("--variational", dest="variational", action="store_true", help="Enable KL-regularized VAE mode")
    parser.add_argument("--deterministic-ae", dest="variational", action="store_false", help="Disable KL and sampling for plain autoencoder mode")
    parser.set_defaults(variational=True)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--boost-channel", type=int, default=0, choices=[0, 1, 2], help="Which channel gets high-end activation boost in detector_reference mode")
    parser.add_argument("--boost-factor", type=float, default=1.5, help="Boost factor applied above the high-activation threshold")
    parser.add_argument("--decoder-batchnorm", dest="decoder_batchnorm", action="store_true", help="Use BatchNorm inside bilinear decoder blocks")
    parser.add_argument("--no-decoder-batchnorm", dest="decoder_batchnorm", action="store_false", help="Disable BatchNorm inside bilinear decoder blocks")
    parser.set_defaults(decoder_batchnorm=True)
    parser.add_argument("--output-bias-init", type=float, default=-2.0, help="Initial bias for final reconstruction layer")
    main(parser.parse_args())
