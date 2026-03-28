"""
task3_diffusion.py — Specific Task 3
========================================
Denoising Diffusion Probabilistic Model (DDPM) for jet image generation.

The DDPM learns to reverse a noise-corruption process, generating realistic
jet images from pure Gaussian noise. Uses L1 loss (not L2) and [-1, 1] data
scaling to correctly handle the extreme sparsity (~98.4% zeros) of jet images.

Usage:
    # Quick local test (CPU, small subset)
    python src/task3_diffusion.py --max-events 200 --epochs 3 --force-cpu

    # Full training (GPU)
    python src/task3_diffusion.py --epochs 30 --batch-size 32

Outputs (saved to outputs/task3/<exp_name>/):
    task3_reconstructions.png — side-by-side original vs generated
    task3_loss_curve.png      — training convergence plot
    metrics.json              — quantitative evaluation metrics
"""

import os
import sys
import json
import argparse
import shutil
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    setup_logging, set_seed, get_device, ensure_dirs, ensure_task_dirs,
    CHECKPOINT_DIR, CHANNEL_NAMES, get_auto_batch_size,
)
from src.data_utils import load_dataset, JetImageDataset, make_splits
from src.metrics import reconstruction_summary
from src.models.diffusion_unet import SimpleUNet
from src.models.diffusion_core import DDPM
from src.experiment_tracker import log_experiment, save_run_metrics

logger = setup_logging("task3")


def save_config(out_dir: str, run_params: dict) -> None:
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)


def save_summary(out_dir: str, exp_name: str, run_params: dict, metrics: dict) -> None:
    lines = [
        f"# {exp_name}",
        "",
        "## Positioning",
        "",
        "- Exploratory pixel-space diffusion baseline for sparse detector images.",
        f"- Channel order: {', '.join(CHANNEL_NAMES)}.",
        "- This section is intentionally framed as exploratory because sparse detector structure is not naturally dense-image-like.",
        "",
        "## Config",
        "",
    ]
    for key, value in sorted(run_params.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Metrics", ""])
    for key, value in sorted(metrics.items()):
        lines.append(f"- `{key}`: `{value:.6f}`" if isinstance(value, (int, float)) else f"- `{key}`: `{value}`")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_samples(
    originals: np.ndarray,
    generated: np.ndarray,
    out_dir: str,
    n_show: int = 5,
) -> None:
    """Side-by-side comparison: original vs DDPM-generated for each channel."""
    n_show = min(n_show, len(originals), len(generated))
    fig, axes = plt.subplots(n_show, 6, figsize=(20, n_show * 2.5), squeeze=False)
    fig.suptitle(
        "Task 3 — Diffusion Model (Original | Generated)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for row in range(n_show):
        for ch in range(3):
            orig_img = originals[row, ch]
            gen_img = generated[row, ch]
            vmax = max(orig_img.max(), 1e-5)

            axes[row, ch].imshow(orig_img, cmap="hot", vmin=0, vmax=vmax)
            axes[row, ch + 3].imshow(gen_img, cmap="hot", vmin=0, vmax=vmax)
            axes[row, ch].axis("off")
            axes[row, ch + 3].axis("off")

            if row == 0:
                axes[row, ch].set_title(f"Original {CHANNEL_NAMES[ch]}", fontsize=10)
                axes[row, ch + 3].set_title(f"Generated {CHANNEL_NAMES[ch]}", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "task3_reconstructions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reconstructions → %s", path)
    shutil.copy2(path, os.path.join(out_dir, "original_vs_generated_or_reconstructed.png"))


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    out_dir: str,
) -> None:
    """Training convergence visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train L1", color="#2563EB", lw=2)
    ax.plot(epochs, val_losses, label="Val L1", color="#DC2626", lw=2, ls="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("L1 Noise Prediction Loss", fontsize=12)
    ax.set_title("Task 3 — DDPM Training Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "task3_loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curve → %s", path)
    shutil.copy2(path, os.path.join(out_dir, "training_curves.png"))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    ensure_dirs()
    out_dir = ensure_task_dirs("task3", args.exp_name)
    logger.info("Experiment: %s → %s", args.exp_name, out_dir)

    # ── Data Loading ──
    X, y = load_dataset(max_events=args.max_events)
    train_idx, val_idx, test_idx = make_splits(X, y, seed=args.seed)

    # Create datasets (JetImageDataset applies log1p + [0,1] normalization)
    train_ds = JetImageDataset(X[train_idx], y[train_idx])
    val_ds = JetImageDataset(X[val_idx], y[val_idx])
    test_ds = JetImageDataset(X[test_idx], y[test_idx])

    batch_size = args.batch_size if args.batch_size > 0 else get_auto_batch_size(task_num=3)
    if args.batch_size <= 0:
        logger.info("Auto-scaled batch size to %d based on VRAM", batch_size)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # ── Model ──
    model = SimpleUNet(in_channels=3, base_channels=64).to(device)
    ddpm = DDPM(model, timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SimpleUNet — %s trainable parameters", f"{n_params:,}")
    logger.info("Starting DDPM training for %d epochs (patience=%d)...", args.epochs, args.patience)

    # ── Training Loop ──
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ddpm_{args.exp_name}.pt")

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for imgs, _ in tqdm(train_loader, leave=False, desc="Training"):
            imgs = imgs.to(device, non_blocking=True)
            # Scale [0,1] → [-1,1] for DDPM (empty pixels at -1 = symmetric for Gaussian noise)
            imgs_scaled = imgs * 2.0 - 1.0

            optimizer.zero_grad(set_to_none=True)
            loss = ddpm.compute_loss(imgs_scaled, loss_type=args.loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        tr_loss = epoch_loss / n_batches

        # Validate
        model.eval()
        vl_loss_sum = 0.0
        vl_batches = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                imgs_scaled = imgs * 2.0 - 1.0
                vl_loss_sum += ddpm.compute_loss(imgs_scaled, loss_type=args.loss_type).item()
                vl_batches += 1
        vl_loss = vl_loss_sum / vl_batches

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
            torch.save(best_state, ckpt_path)
        else:
            patience_counter += 1
            marker = ""

        pbar.set_postfix(train=f"{tr_loss:.4f}", val=f"{vl_loss:.4f}")
        print(f"Epoch {epoch:03d}/{args.epochs} | Train L1: {tr_loss:.4f} | Val L1: {vl_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}{marker}")

        if patience_counter >= args.patience and args.patience > 0:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Restore best model ──
    model.load_state_dict(best_state)
    logger.info("Restored best model (Val L1: %.4f)", best_val_loss)

    # ── Generate samples & evaluate ──
    logger.info("Sampling %d images from DDPM (this may take a while)...", args.n_samples)
    n_samples = min(args.n_samples, len(test_ds))
    samples_ddpm = ddpm.sample((n_samples, 3, 125, 125))
    # Scale [-1,1] → [0,1] and clamp
    samples_01 = ((samples_ddpm + 1.0) / 2.0).clamp(0, 1).cpu()

    # Get real test images for comparison
    real_imgs = torch.stack([test_ds[i][0] for i in range(n_samples)])

    # Compute metrics
    metrics = reconstruction_summary(real_imgs, samples_01)
    logger.info("═══ Generation Quality (vs Test Set) ═══")
    for k, v in metrics.items():
        logger.info("  %s: %.6f", k, v)

    # Sparsity check
    gen_density = (samples_01 > 0.01).float().mean().item() * 100
    real_density = (real_imgs > 0.01).float().mean().item() * 100
    metrics["generated_active_density_pct"] = gen_density
    metrics["real_active_density_pct"] = real_density
    logger.info("  Generated active density: %.2f%% (real: %.2f%%)", gen_density, real_density)

    # ── Save outputs ──
    run_params = {
        "epochs": args.epochs, "lr": args.lr, "batch_size": batch_size,
        "timesteps": args.timesteps, "max_events": args.max_events,
        "seed": args.seed, "loss": args.loss_type.upper(), "data_scaling": "[-1,1]",
    }
    save_run_metrics(out_dir, metrics, run_params)
    save_config(out_dir, run_params)
    save_summary(out_dir, args.exp_name, run_params, metrics)

    # Plots
    plot_loss_curve(train_losses, val_losses, out_dir)
    plot_samples(real_imgs.numpy(), samples_01.numpy(), out_dir, n_show=n_samples)
    torch.save(
        {
            "model_state_dict": best_state,
            "params": run_params,
            "metrics": metrics,
        },
        os.path.join(out_dir, "model_checkpoint.pt"),
    )

    # Log experiment
    log_experiment("task3", args.exp_name, run_params, metrics, status="SUCCESS")

    logger.info("Task 3 complete. Outputs → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specific Task 3 — DDPM Jet Image Generation")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=auto scale based on VRAM)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (0=disabled)")
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM noise schedule steps")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to generate for evaluation")
    parser.add_argument("--max-events", type=int, default=None, help="Limit events (None=all)")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="l1_ddpm", help="Experiment name for output dir")
    parser.add_argument("--loss-type", choices=["l1", "mse"], default="l1", help="Noise prediction loss for DDPM training")
    main(parser.parse_args())
