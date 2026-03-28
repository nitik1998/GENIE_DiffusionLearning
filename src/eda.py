"""
eda.py — Exploratory Data Analysis
======================================
Visualise the quark/gluon jet dataset to understand sparsity,
channel distributions, and sample events before training.

Usage:
    python src/eda.py --max-events 1000

Outputs (saved to outputs/eda/):
    sample_events.png          — 6 sample events (quark + gluon)
    channel_distributions.png  — per-channel log-scale histograms
    sparsity_analysis.png      — active pixel density distribution
    eda_stats.json             — numerical summary
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import setup_logging, OUTPUT_DIR, CHANNEL_NAMES, IMAGE_SIZE
from src.data_utils import load_dataset

logger = setup_logging("eda")


def run_eda(args: argparse.Namespace) -> None:
    out_dir = os.path.join(OUTPUT_DIR, "eda")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_dataset(max_events=args.max_events)
    n_events = len(y)
    n_quark = int((y == 0).sum())
    n_gluon = int((y == 1).sum())

    # ── Sparsity Analysis ──
    total_pixels = IMAGE_SIZE * IMAGE_SIZE * 3
    active_per_event = (X > 0).reshape(n_events, -1).sum(axis=1)
    density_per_event = active_per_event / total_pixels * 100

    stats = {
        "n_events": n_events,
        "n_quark": n_quark,
        "n_gluon": n_gluon,
        "mean_active_pixels": float(active_per_event.mean()),
        "mean_active_density_pct": float(density_per_event.mean()),
        "median_active_density_pct": float(np.median(density_per_event)),
        "max_pixel_value": float(X.max()),
        "per_channel_max": {
            name: float(X[:, ch].max()) for ch, name in enumerate(CHANNEL_NAMES)
        },
        "per_channel_mean_nonzero": {
            name: float(X[:, ch][X[:, ch] > 0].mean()) if (X[:, ch] > 0).any() else 0.0
            for ch, name in enumerate(CHANNEL_NAMES)
        },
    }

    with open(os.path.join(out_dir, "eda_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Sparsity: %.2f%% active (mean), max pixel = %.1f",
                stats["mean_active_density_pct"], stats["max_pixel_value"])

    # ── Plot 1: Sample Events ──
    fig, axes = plt.subplots(4, 6, figsize=(20, 12), squeeze=False)
    fig.suptitle("Sample Jet Events (Top 2: Quark, Bottom 2: Gluon)", fontsize=14, fontweight="bold")

    quark_idx = np.where(y == 0)[0][:2]
    gluon_idx = np.where(y == 1)[0][:2]
    show_idx = np.concatenate([quark_idx, gluon_idx])

    for row, idx in enumerate(show_idx):
        label = "Quark" if y[idx] == 0 else "Gluon"
        for ch in range(3):
            ax = axes[row, ch]
            img = X[idx, ch]
            vmax = max(img.max(), 1e-6)
            ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{CHANNEL_NAMES[ch]}", fontsize=11)
            if ch == 0:
                ax.set_ylabel(f"{label} #{row % 2 + 1}", fontsize=10)

        # Log1p view
        for ch in range(3):
            ax = axes[row, ch + 3]
            img = np.log1p(X[idx, ch])
            vmax = max(img.max(), 1e-6)
            ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"log1p({CHANNEL_NAMES[ch]})", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_events.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Channel Distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Per-Channel Non-Zero Pixel Value Distributions (log scale)", fontsize=13, fontweight="bold")
    colors = ["#2563EB", "#DC2626", "#059669"]

    for ch, name in enumerate(CHANNEL_NAMES):
        vals = X[:, ch].flatten()
        vals = vals[vals > 0]
        if len(vals) > 0:
            axes[ch].hist(np.log1p(vals), bins=100, color=colors[ch], alpha=0.8, edgecolor="none")
        axes[ch].set_title(name, fontsize=12)
        axes[ch].set_xlabel("log1p(pixel value)")
        axes[ch].set_ylabel("Count")
        axes[ch].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "channel_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Sparsity Analysis ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(density_per_event, bins=80, color="#7C3AED", alpha=0.85, edgecolor="none")
    ax.axvline(density_per_event.mean(), color="#DC2626", ls="--", lw=2,
               label=f"Mean: {density_per_event.mean():.2f}%")
    ax.set_xlabel("Active Pixel Density (%)", fontsize=12)
    ax.set_ylabel("Number of Events", fontsize=12)
    ax.set_title("Jet Image Sparsity Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sparsity_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("EDA outputs saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA — Jet Image Dataset Analysis")
    parser.add_argument("--max-events", type=int, default=None, help="Limit events for quick analysis")
    run_eda(parser.parse_args())
