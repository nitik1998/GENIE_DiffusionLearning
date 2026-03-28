"""
metrics.py — Evaluation Metrics
==================================
Quantitative evaluation utilities for reconstruction quality and
classification performance across all three tasks.

Provides per-channel and aggregate metrics beyond simple MSE,
demonstrating rigorous evaluation methodology.
"""

import numpy as np
import torch
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────
# Reconstruction Metrics
# ─────────────────────────────────────────────────────────────

def per_channel_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    channel_names: Tuple[str, ...] = ("Tracks", "ECAL", "HCAL"),
) -> Dict[str, float]:
    """
    Compute MSE independently for each detector channel.

    This reveals whether the model struggles with specific detector
    sub-systems (e.g., calorimeter response vs. sparse tracking hits).

    Args:
        original:      Tensor of shape (B, C, H, W).
        reconstructed: Tensor of shape (B, C, H, W).
        channel_names: Human-readable names for each channel.

    Returns:
        Dictionary mapping channel name → MSE value.
    """
    assert original.shape == reconstructed.shape
    results = {}
    for i, name in enumerate(channel_names):
        mse = torch.mean((original[:, i] - reconstructed[:, i]) ** 2).item()
        results[name] = mse
    results["overall"] = torch.mean((original - reconstructed) ** 2).item()
    return results


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 · log₁₀(MAX² / MSE)

    Higher is better. Typical good reconstruction: >30 dB.
    """
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim_simple(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Simplified Structural Similarity Index (SSIM) computation.

    SSIM captures perceptual similarity by comparing luminance, contrast,
    and structure — more aligned with human perception than MSE.

    SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2) /
                 (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    Args:
        original:      Tensor of shape (B, C, H, W) in [0, 1].
        reconstructed: Tensor of shape (B, C, H, W) in [0, 1].

    Returns:
        Mean SSIM value across batch and channels.
    """
    try:
        from pytorch_msssim import ssim
        return ssim(original, reconstructed, data_range=1.0, size_average=True).item()
    except ImportError:
        # Fallback: compute a simplified global SSIM without windowing
        mu_x = original.mean(dim=(-2, -1), keepdim=True)
        mu_y = reconstructed.mean(dim=(-2, -1), keepdim=True)

        sigma_x_sq = ((original - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_y_sq = ((reconstructed - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_xy = ((original - mu_x) * (reconstructed - mu_y)).mean(
            dim=(-2, -1), keepdim=True
        )

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map.mean().item()


# ─────────────────────────────────────────────────────────────
# Summary Reporter
# ─────────────────────────────────────────────────────────────

def reconstruction_summary(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """
    Comprehensive reconstruction quality report.

    Returns dict with overall MSE, per-channel MSE, PSNR, and SSIM.
    """
    ch_mse = per_channel_mse(original, reconstructed)
    psnr = compute_psnr(original, reconstructed)
    ssim_val = compute_ssim_simple(original, reconstructed)

    return {
        **{f"mse_{k}": v for k, v in ch_mse.items()},
        "psnr_db": psnr,
        "ssim": ssim_val,
    }


def sparse_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    true_threshold: float = 0.01,
    pred_threshold: float = 0.05,
) -> Dict[str, float]:
    active_true = original > true_threshold
    active_pred = reconstructed > pred_threshold
    inactive_true = ~active_true

    if torch.any(active_true):
        nonzero_mse = torch.mean((original[active_true] - reconstructed[active_true]) ** 2).item()
    else:
        nonzero_mse = torch.mean((original - reconstructed) ** 2).item()

    tp = torch.logical_and(active_true, active_pred).sum().item()
    fp = torch.logical_and(inactive_true, active_pred).sum().item()
    fn = torch.logical_and(active_true, ~active_pred).sum().item()
    inactive_count = inactive_true.sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    background_false_activation = fp / (inactive_count + 1e-8)

    return {
        "nonzero_mse": float(nonzero_mse),
        "active_precision": float(precision),
        "active_recall": float(recall),
        "active_iou": float(iou),
        "background_false_activation": float(background_false_activation),
        "pred_threshold": float(pred_threshold),
    }
