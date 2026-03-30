"""
task3_diffusion.py — Specific Task 3
========================================
Diffusion models for jet image generation with two experiment modes:

  1. image_ddpm     — pixel-space DDPM using U-Net (baseline)
  2. latent_diffusion — latent-space DDPM using frozen VAE + MLP denoiser (main)

Usage:
    # Latent diffusion (main experiment)
    python src/task3_diffusion.py --mode latent_diffusion --epochs 30

    # Image DDPM (baseline)
    python src/task3_diffusion.py --mode image_ddpm --epochs 30

    # Quick smoke test
    python src/task3_diffusion.py --mode latent_diffusion --max-events 200 --epochs 3

Outputs (saved to outputs/task3/<exp_name>/):
    original_vs_reconstructed.png
    original_vs_reconstructed_vs_abs_error.png
    samples_channelwise.png
    training_curves.png
    metrics.json / config.json / model_checkpoint.pt
"""

import os
import sys
import json
import copy
import time
import argparse
import shutil
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    setup_logging, set_seed, get_device, ensure_dirs, ensure_task_dirs,
    CHECKPOINT_DIR, CHANNEL_NAMES, get_auto_batch_size,
)
from src.data_utils import load_dataset, JetImageDataset, make_splits
from src.metrics import reconstruction_summary
from src.models.diffusion_unet import SimpleUNet
from src.models.diffusion_core import DDPM
from src.models.latent_denoiser import LatentDenoiser
from src.models.autoencoder import ConvAutoEncoder, DeepFalconVAE
from src.experiment_tracker import log_experiment, save_run_metrics

logger = setup_logging("task3")


# ─────────────────────────────────────────────────────────────
# Visualization (channel-wise, median-based vmin)
# ─────────────────────────────────────────────────────────────

def plot_channel_samples(
    originals: np.ndarray,
    generated: np.ndarray,
    out_dir: str,
    n_show: int = 6,
    title_prefix: str = "Task 3",
) -> None:
    """Channel-wise comparison: original vs generated for each detector channel."""
    n_show = min(n_show, len(originals), len(generated))
    fig, axes = plt.subplots(n_show, 6, figsize=(20, n_show * 2.5), squeeze=False)
    fig.suptitle(
        f"{title_prefix} — Original | Generated (per-channel)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for row in range(n_show):
        for ch in range(3):
            orig_ch = originals[row, ch]
            gen_ch = generated[row, ch]
            bg = float(np.median(orig_ch))
            vmax = max(float(orig_ch.max()), float(gen_ch.max()), bg + 1e-6)
            axes[row, ch].imshow(orig_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 3].imshow(gen_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")
            if row == 0:
                axes[row, ch].set_title(f"Original {CHANNEL_NAMES[ch]}", fontsize=10)
                axes[row, ch + 3].set_title(f"Generated {CHANNEL_NAMES[ch]}", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "original_vs_reconstructed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved channel-wise samples → %s", path)

    # Also save as samples_channelwise.png
    shutil.copy2(path, os.path.join(out_dir, "samples_channelwise.png"))


def plot_error_maps(
    originals: np.ndarray,
    generated: np.ndarray,
    out_dir: str,
    n_show: int = 6,
) -> None:
    """Original | Generated | Absolute Error per channel."""
    n_show = min(n_show, len(originals), len(generated))
    errors = np.abs(originals - generated)
    fig, axes = plt.subplots(n_show, 9, figsize=(28, n_show * 2.4), squeeze=False)
    fig.suptitle(
        "Task 3 — Original | Generated | Absolute Error",
        fontsize=14, fontweight="bold", y=1.01,
    )
    for row in range(n_show):
        for ch in range(3):
            orig_ch = originals[row, ch]
            gen_ch = generated[row, ch]
            bg = float(np.median(orig_ch))
            vmax = max(float(orig_ch.max()), float(gen_ch.max()), bg + 1e-6)
            evmax = float(np.percentile(errors[row, ch], 99.5)) if np.any(errors[row, ch] > 0) else 1e-6
            axes[row, ch].imshow(orig_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 3].imshow(gen_ch, cmap="hot", vmin=bg, vmax=vmax)
            axes[row, ch + 6].imshow(errors[row, ch], cmap="magma", vmin=0, vmax=evmax)
            if row == 0:
                axes[row, ch].set_title(f"Orig {CHANNEL_NAMES[ch]}", fontsize=9)
                axes[row, ch + 3].set_title(f"Gen {CHANNEL_NAMES[ch]}", fontsize=9)
                axes[row, ch + 6].set_title(f"Error {CHANNEL_NAMES[ch]}", fontsize=9)
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")
            axes[row, ch + 6].axis("off")
    plt.tight_layout()
    path = os.path.join(out_dir, "original_vs_reconstructed_vs_abs_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error maps → %s", path)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    out_dir: str,
    loss_label: str = "Loss",
) -> None:
    """Training convergence visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, label="Train", color="#2563EB", lw=2)
    ax.semilogy(epochs, val_losses, label="Val", color="#DC2626", lw=2, ls="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(loss_label, fontsize=12)
    ax.set_title("Task 3 — Training Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", path)


def save_config(out_dir: str, run_params: dict) -> None:
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)


def compute_nonzero_mse(real: torch.Tensor, gen: torch.Tensor) -> dict:
    """MSE computed only on nonzero pixels (active detector signal)."""
    metrics = {}
    for ch, name in enumerate(CHANNEL_NAMES):
        mask = real[:, ch] > 0.01
        if mask.any():
            metrics[f"nonzero_mse_{name}"] = float(F.mse_loss(gen[:, ch][mask], real[:, ch][mask]).item())
        else:
            metrics[f"nonzero_mse_{name}"] = 0.0
    # Overall
    mask_all = real > 0.01
    if mask_all.any():
        metrics["nonzero_mse_overall"] = float(F.mse_loss(gen[mask_all], real[mask_all]).item())
    else:
        metrics["nonzero_mse_overall"] = 0.0
    return metrics


# ─────────────────────────────────────────────────────────────
# VAE Auto-Training (for latent diffusion when no checkpoint)
# ─────────────────────────────────────────────────────────────

def train_quick_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    latent_dim: int = 256,
) -> nn.Module:
    """Train a lightweight VAE for encoding/decoding. Used when no Task 1 checkpoint is available."""
    logger.info("No VAE checkpoint found. Auto-training ConvAutoEncoder (%d epochs)...", epochs)
    vae = ConvAutoEncoder(in_channels=3).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    print(f"=== Auto-training VAE for {epochs} epochs ===", flush=True)
    for epoch in range(1, epochs + 1):
        vae.train()
        total_loss = 0.0
        n = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            recon, _, _ = vae(imgs)
            loss = F.mse_loss(recon, imgs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        tr_loss = total_loss / n
        print(f"VAE E{epoch}/{epochs} | Loss: {tr_loss:.6f}", flush=True)

    vae.eval()
    logger.info("VAE auto-training complete.")
    return vae


# ─────────────────────────────────────────────────────────────
# Mode 1: Image DDPM (baseline)
# ─────────────────────────────────────────────────────────────

def run_image_ddpm(args: argparse.Namespace) -> None:
    """Pixel-space DDPM with U-Net — baseline experiment."""
    device = get_device(args.force_cpu)
    out_dir = ensure_task_dirs("task3", args.exp_name)
    logger.info("Experiment [image_ddpm]: %s → %s", args.exp_name, out_dir)

    # Data
    X, y = load_dataset(max_events=args.max_events)
    train_idx, val_idx, test_idx = make_splits(X, y, seed=args.seed)
    train_ds = JetImageDataset(X[train_idx], y[train_idx])
    val_ds = JetImageDataset(X[val_idx], y[val_idx])
    test_ds = JetImageDataset(X[test_idx], y[test_idx])

    batch_size = args.batch_size if args.batch_size > 0 else get_auto_batch_size(task_num=3)
    pin = device.type == "cuda"
    # num_workers=0: prevents Colab /dev/shm exhaustion crash
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Model
    model = SimpleUNet(in_channels=3, base_channels=64).to(device)
    ddpm = DDPM(model, timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SimpleUNet — %s trainable params, timesteps=%d", f"{n_params:,}", args.timesteps)

    # Training
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"=== Starting training [image_ddpm] for {args.epochs} epochs ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            imgs_scaled = imgs * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = ddpm.compute_loss(imgs_scaled, loss_type="l1")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        tr_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                vl_sum += ddpm.compute_loss(imgs * 2.0 - 1.0, loss_type="l1").item()
                vl_n += 1
        vl_loss = vl_sum / vl_n

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # CPU to avoid VRAM frag
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        elapsed = time.time() - epoch_start
        remaining = elapsed * (args.epochs - epoch)
        print(f"E{epoch}/{args.epochs} [{elapsed:.1f}s] train={tr_loss:.4f} val={vl_loss:.4f} ETA={remaining/60:.1f}min{marker}", flush=True)

        if patience_counter >= args.patience and args.patience > 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # Restore best & generate
    model.load_state_dict(best_state)
    n_samples = min(args.n_samples, len(test_ds))
    logger.info("Sampling %d images...", n_samples)
    samples = ddpm.sample((n_samples, 3, 125, 125))
    samples_01 = ((samples + 1.0) / 2.0).clamp(0, 1).cpu()
    real_imgs = torch.stack([test_ds[i][0] for i in range(n_samples)])

    # Metrics
    metrics = compute_nonzero_mse(real_imgs, samples_01)
    basic = reconstruction_summary(real_imgs, samples_01)
    metrics.update(basic)

    gen_density = (samples_01 > 0.01).float().mean().item() * 100
    real_density = (real_imgs > 0.01).float().mean().item() * 100
    metrics["generated_active_density_pct"] = gen_density
    metrics["real_active_density_pct"] = real_density

    logger.info("═══ Image DDPM Results ═══")
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)

    # Save
    run_params = {
        "mode": "image_ddpm", "epochs": args.epochs, "lr": args.lr,
        "batch_size": batch_size, "timesteps": args.timesteps,
        "max_events": args.max_events, "seed": args.seed,
    }
    save_run_metrics(out_dir, metrics, run_params)
    save_config(out_dir, run_params)
    plot_training_curves(train_losses, val_losses, out_dir, "L1 Noise Prediction Loss")
    plot_channel_samples(real_imgs.numpy(), samples_01.numpy(), out_dir, n_show=n_samples, title_prefix="Task 3 — Image DDPM")
    plot_error_maps(real_imgs.numpy(), samples_01.numpy(), out_dir, n_show=min(n_samples, 6))
    torch.save({"model_state_dict": best_state, "params": run_params, "metrics": metrics},
               os.path.join(out_dir, "model_checkpoint.pt"))
    log_experiment("task3", args.exp_name, run_params, metrics, status="SUCCESS")
    logger.info("Image DDPM complete → %s", out_dir)


# ─────────────────────────────────────────────────────────────
# Mode 2: Latent Diffusion (main)
# ─────────────────────────────────────────────────────────────

def run_latent_diffusion(args: argparse.Namespace) -> None:
    """Latent-space DDPM: frozen VAE encoder/decoder + MLP denoiser."""
    device = get_device(args.force_cpu)
    out_dir = ensure_task_dirs("task3", args.exp_name)
    logger.info("Experiment [latent_diffusion]: %s → %s", args.exp_name, out_dir)

    # Data
    X, y = load_dataset(max_events=args.max_events)
    train_idx, val_idx, test_idx = make_splits(X, y, seed=args.seed)
    train_ds = JetImageDataset(X[train_idx], y[train_idx])
    val_ds = JetImageDataset(X[val_idx], y[val_idx])
    test_ds = JetImageDataset(X[test_idx], y[test_idx])

    batch_size = args.batch_size if args.batch_size > 0 else get_auto_batch_size(task_num=3)
    pin = device.type == "cuda"
    # num_workers=0: prevents Colab /dev/shm exhaustion crash
    img_loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    img_loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # ── Step 1: Get or train VAE ──
    # Auto-discover Task 1 checkpoint if not specified
    vae_ckpt = args.vae_checkpoint
    if not vae_ckpt or not os.path.exists(vae_ckpt):
        for candidate in [
            "results_from_colab/task1_autoencoder/best_model.pt",
            "results_from_colab/task1_autoencoder/model_checkpoint.pt",
            os.path.join(os.path.expanduser("~"), ".cache/genie/task1/best_model.pt"),
        ]:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full = os.path.join(repo_root, candidate)
            if os.path.exists(full):
                vae_ckpt = full
                break
    vae = None
    if vae_ckpt and os.path.exists(vae_ckpt):
        logger.info("Loading VAE from %s", vae_ckpt)
        ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Detect architecture from state keys
        if "fc_mu.weight" in state:
            vae = DeepFalconVAE(in_channels=3, latent_dim=256).to(device)
        else:
            vae = ConvAutoEncoder(in_channels=3).to(device)
        vae.load_state_dict(state)
        logger.info("Loaded VAE: %s", vae.__class__.__name__)
    else:
        # Auto-train a lightweight VAE
        train_loader_vae = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        val_loader_vae = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        vae = train_quick_vae(train_loader_vae, val_loader_vae, device, epochs=20)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    logger.info("VAE frozen (%s trainable params)", "0")

    # ── Step 2: Pre-encode dataset into latent space ──
    logger.info("Pre-encoding images into latent vectors...")
    latent_spatial_shape = None  # Track conv latent shape for ConvAutoEncoder
    def encode_dataset(loader):
        nonlocal latent_spatial_shape
        latents = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                z = vae.get_latent(imgs)  # (B, latent_dim) or (B, C, H, W)
                if z.dim() > 2:
                    if latent_spatial_shape is None:
                        latent_spatial_shape = z.shape[1:]  # e.g. (32, 8, 8)
                    z = z.view(z.size(0), -1)  # Flatten conv latent to 1D
                latents.append(z.cpu())
        return torch.cat(latents, dim=0)

    z_train = encode_dataset(img_loader_train)
    z_val = encode_dataset(img_loader_val)
    latent_dim = z_train.shape[1]
    logger.info("Latent dim: %d, train: %d, val: %d", latent_dim, len(z_train), len(z_val))

    # Free the massive raw image arrays — latents are all we need for training
    import gc
    del train_ds, val_ds, img_loader_train, img_loader_val
    gc.collect()

    # Normalize latents for stable diffusion
    z_mean = z_train.mean(dim=0)
    z_std = z_train.std(dim=0).clamp(min=1e-6)
    z_train_norm = (z_train - z_mean) / z_std
    z_val_norm = (z_val - z_mean) / z_std

    z_train_ds = TensorDataset(z_train_norm)
    z_val_ds = TensorDataset(z_val_norm)
    z_train_loader = DataLoader(z_train_ds, batch_size=batch_size, shuffle=True)
    z_val_loader = DataLoader(z_val_ds, batch_size=batch_size, shuffle=False)

    # ── Step 3: Train latent denoiser ──
    denoiser = LatentDenoiser(latent_dim=latent_dim, hidden_dim=512, time_emb_dim=128).to(device)
    ddpm = DDPM(denoiser, timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    logger.info("LatentDenoiser — %s trainable params, latent_dim=%d, timesteps=%d",
                f"{n_params:,}", latent_dim, args.timesteps)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"=== Starting training [latent_diffusion] for {args.epochs} epochs ===", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        denoiser.train()
        epoch_loss, n_batches = 0.0, 0
        for (z_batch,) in z_train_loader:
            z_batch = z_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.compute_loss(z_batch, loss_type="mse")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        tr_loss = epoch_loss / n_batches

        # Validate
        denoiser.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for (z_batch,) in z_val_loader:
                z_batch = z_batch.to(device, non_blocking=True)
                vl_sum += ddpm.compute_loss(z_batch, loss_type="mse").item()
                vl_n += 1
        vl_loss = vl_sum / vl_n

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.cpu() for k, v in denoiser.state_dict().items()}  # CPU to avoid VRAM frag
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        elapsed = time.time() - epoch_start
        remaining = elapsed * (args.epochs - epoch)
        print(f"E{epoch}/{args.epochs} [{elapsed:.1f}s] train={tr_loss:.6f} val={vl_loss:.6f} ETA={remaining/60:.1f}min{marker}", flush=True)

        if patience_counter >= args.patience and args.patience > 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # ── Step 4: Generate samples ──
    denoiser.load_state_dict(best_state)
    n_samples = min(args.n_samples, len(test_ds))
    logger.info("Sampling %d latent vectors and decoding...", n_samples)

    z_sampled_norm = ddpm.sample((n_samples, latent_dim))
    # Denormalize
    z_sampled = z_sampled_norm * z_std.to(device) + z_mean.to(device)

    # Decode through frozen VAE
    with torch.no_grad():
        z_decode = z_sampled
        if latent_spatial_shape is not None:
            z_decode = z_decode.view(z_decode.size(0), *latent_spatial_shape)
        generated = vae.decode(z_decode).clamp(0, 1).cpu()

    real_imgs = torch.stack([test_ds[i][0] for i in range(n_samples)])

    # Metrics
    metrics = compute_nonzero_mse(real_imgs, generated)
    basic = reconstruction_summary(real_imgs, generated)
    metrics.update(basic)

    gen_density = (generated > 0.01).float().mean().item() * 100
    real_density = (real_imgs > 0.01).float().mean().item() * 100
    metrics["generated_active_density_pct"] = gen_density
    metrics["real_active_density_pct"] = real_density

    logger.info("═══ Latent Diffusion Results ═══")
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)

    # Save
    run_params = {
        "mode": "latent_diffusion", "epochs": args.epochs, "lr": args.lr,
        "batch_size": batch_size, "latent_dim": latent_dim, "timesteps": args.timesteps,
        "max_events": args.max_events, "seed": args.seed,
        "vae_type": vae.__class__.__name__,
    }
    save_run_metrics(out_dir, metrics, run_params)
    save_config(out_dir, run_params)
    plot_training_curves(train_losses, val_losses, out_dir, "Latent MSE Loss")
    plot_channel_samples(real_imgs.numpy(), generated.numpy(), out_dir, n_show=n_samples, title_prefix="Task 3 — Latent Diffusion")
    plot_error_maps(real_imgs.numpy(), generated.numpy(), out_dir, n_show=min(n_samples, 6))

    # Save checkpoint with everything needed to reproduce
    torch.save({
        "denoiser_state_dict": best_state,
        "vae_state_dict": vae.state_dict(),
        "vae_class": vae.__class__.__name__,
        "z_mean": z_mean,
        "z_std": z_std,
        "params": run_params,
        "metrics": metrics,
    }, os.path.join(out_dir, "model_checkpoint.pt"))

    log_experiment("task3", args.exp_name, run_params, metrics, status="SUCCESS")
    logger.info("Latent Diffusion complete → %s", out_dir)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dirs()

    if args.mode == "image_ddpm":
        if args.exp_name == "latent_diffusion":
            args.exp_name = "image_ddpm"
        run_image_ddpm(args)
    elif args.mode == "latent_diffusion":
        if args.exp_name == "image_ddpm":
            args.exp_name = "latent_diffusion"
        run_latent_diffusion(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'image_ddpm' or 'latent_diffusion'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 — Diffusion Models for Jet Images")
    parser.add_argument("--mode", choices=["image_ddpm", "latent_diffusion"], default="latent_diffusion",
                        help="Experiment mode: image_ddpm (baseline) or latent_diffusion (main)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=auto)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (0=disabled)")
    parser.add_argument("--timesteps", type=int, default=200, help="DDPM noise schedule steps")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples for evaluation")
    parser.add_argument("--max-events", type=int, default=None, help="Limit events (None=all)")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="latent_diffusion", help="Experiment name for output dir")
    parser.add_argument("--vae-checkpoint", type=str, default=None, help="Path to pretrained VAE (optional)")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun even if outputs exist")
    main(parser.parse_args())
