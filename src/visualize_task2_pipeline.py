"""
visualize_task2_pipeline.py — Task 2 Full Pipeline Visualization
=================================================================
Demonstrates the complete Image → Point Cloud → Graph pipeline for 
quark/gluon jet classification using k-NN graph construction.

For each sample, creates a 3-row × 3-column figure:
  Row 1: Original 3-channel detector images (Tracks, ECAL, HCAL)
  Row 2: Point cloud extracted from non-zero pixels (η vs φ, colored by energy)
  Row 3: k-NN graph with edges overlaid on nodes

Saves individual sample figures + a combined summary figure.

Usage:
    python src/visualize_task2_pipeline.py --n-samples 5 --k 8
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_dataset, image_to_pointcloud, compute_channel_scales, normalize_channels
from src.models.gnn import build_knn_graph
from src.config import CHANNEL_NAMES, OUTPUT_DIR, setup_logging

logger = setup_logging("viz_task2")


def visualize_single_sample(
    img_raw: np.ndarray,
    label: int,
    sample_idx: int,
    out_dir: str,
    k: int = 8,
) -> None:
    """
    Create a comprehensive 3-stage visualization for one jet event.
    
    Stage 1: Original detector images (3 channels)
    Stage 2: Point cloud (non-zero pixel extraction)
    Stage 3: k-NN graph (edges connecting nearby nodes)
    """
    jet_type = "Quark" if label == 0 else "Gluon"
    
    # ── Stage 1: Extract point cloud ──
    points = image_to_pointcloud(img_raw, threshold=0.0)
    n_nodes = len(points)
    eta, phi = points[:, 0], points[:, 1]
    e_ecal, e_hcal, e_track = points[:, 2], points[:, 3], points[:, 4]
    total_energy = np.abs(e_ecal) + np.abs(e_hcal) + np.abs(e_track)
    
    # ── Stage 2: Build k-NN graph ──
    points_tensor = torch.tensor(points, dtype=torch.float32)
    graph = build_knn_graph(points_tensor, label, k=k)
    edge_index = graph.edge_index.numpy()
    n_edges = edge_index.shape[1]
    
    # ── Figure: 3 rows × 3 columns ──
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f"Sample {sample_idx} — {jet_type} Jet  |  {n_nodes} active pixels → {n_edges} edges (k={k})",
        fontsize=16, fontweight="bold", y=0.98,
    )
    
    # Row 1: Original 3-channel images
    for ch_idx in range(3):
        ax = fig.add_subplot(3, 3, ch_idx + 1)
        ch_data = img_raw[ch_idx]
        vmax = max(ch_data.max(), 1e-5)
        im = ax.imshow(ch_data, cmap="hot", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"Original — {CHANNEL_NAMES[ch_idx]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("φ pixel index")
        ax.set_ylabel("η pixel index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Point cloud scatter plots (one per channel energy)
    ch_energies = [e_ecal, e_hcal, e_track]
    ch_cmaps = ["Reds", "Greens", "Blues"]
    for ch_idx in range(3):
        ax = fig.add_subplot(3, 3, ch_idx + 4)
        e = np.abs(ch_energies[ch_idx])
        scatter = ax.scatter(
            phi, eta, c=e, cmap=ch_cmaps[ch_idx], s=8, alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(f"Point Cloud — {CHANNEL_NAMES[ch_idx]} Energy", fontsize=12, fontweight="bold")
        ax.set_xlabel("φ (normalized)")
        ax.set_ylabel("η (normalized)")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotate sparsity
        total_pixels = 125 * 125
        density = n_nodes / total_pixels * 100
        ax.text(0.02, 0.98, f"Active: {n_nodes}/{total_pixels} ({density:.1f}%)",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Row 3: Graph visualizations
    # 3a: Full graph with all edges
    ax = fig.add_subplot(3, 3, 7)
    # Draw edges
    segments = []
    for i in range(n_edges):
        src, dst = edge_index[0, i], edge_index[1, i]
        segments.append([(phi[src], eta[src]), (phi[dst], eta[dst])])
    lc = LineCollection(segments, colors="steelblue", linewidths=0.3, alpha=0.15)
    ax.add_collection(lc)
    ax.scatter(phi, eta, c=total_energy, cmap="hot", s=12, zorder=5, edgecolors="black", linewidths=0.3)
    ax.set_title(f"k-NN Graph (k={k}, {n_edges} edges)", fontsize=12, fontweight="bold")
    ax.set_xlabel("φ (normalized)")
    ax.set_ylabel("η (normalized)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    
    # 3b: Zoomed-in subregion to show edge detail
    ax = fig.add_subplot(3, 3, 8)
    # Find densest region for zoom
    center_eta = np.median(eta)
    center_phi = np.median(phi)
    zoom_range = 0.4
    zoom_mask = (np.abs(eta - center_eta) < zoom_range) & (np.abs(phi - center_phi) < zoom_range)
    zoom_indices = set(np.where(zoom_mask)[0])
    
    # Draw zoomed edges
    zoom_segments = []
    for i in range(n_edges):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in zoom_indices or dst in zoom_indices:
            zoom_segments.append([(phi[src], eta[src]), (phi[dst], eta[dst])])
    lc_zoom = LineCollection(zoom_segments, colors="steelblue", linewidths=0.6, alpha=0.4)
    ax.add_collection(lc_zoom)
    ax.scatter(phi[zoom_mask], eta[zoom_mask], c=total_energy[zoom_mask],
               cmap="hot", s=40, zorder=5, edgecolors="black", linewidths=0.5)
    ax.set_title("Zoomed Graph (median region)", fontsize=12, fontweight="bold")
    ax.set_xlabel("φ (normalized)")
    ax.set_ylabel("η (normalized)")
    ax.set_xlim(center_phi - zoom_range, center_phi + zoom_range)
    ax.set_ylim(center_eta - zoom_range, center_eta + zoom_range)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    
    # 3c: Degree distribution histogram
    ax = fig.add_subplot(3, 3, 9)
    degrees = np.zeros(n_nodes, dtype=int)
    for i in range(n_edges):
        degrees[edge_index[0, i]] += 1
    ax.hist(degrees, bins=range(0, max(degrees) + 2), color="steelblue",
            edgecolor="black", alpha=0.7, rwidth=0.85)
    ax.set_title("Node Degree Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Degree (# neighbors)")
    ax.set_ylabel("Count")
    ax.axvline(degrees.mean(), color="red", linestyle="--", label=f"Mean: {degrees.mean():.1f}")
    ax.legend()
    ax.grid(alpha=0.15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, f"sample_{sample_idx}_{jet_type.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample %d (%s) → %s", sample_idx, jet_type, path)


def create_summary_figure(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    k: int = 8,
    n_samples: int = 5,
) -> None:
    """Create a compact summary: 1 row per sample, 3 columns (Image | Point Cloud | Graph)."""
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, n_samples * 4.5), squeeze=False)
    fig.suptitle(
        "Task 2 Pipeline: Image → Point Cloud → k-NN Graph",
        fontsize=18, fontweight="bold", y=1.01,
    )
    
    col_titles = ["Secondary: summed detector image", "Point Cloud (η vs φ)", f"k-NN Graph (k={k})"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight="bold", pad=12)
    
    for row in range(n_samples):
        img = images[row]
        label = labels[row]
        jet_type = "Quark" if label == 0 else "Gluon"
        
        points = image_to_pointcloud(img, threshold=0.0)
        n_nodes = len(points)
        eta, phi = points[:, 0], points[:, 1]
        total_energy = np.abs(points[:, 2]) + np.abs(points[:, 3]) + np.abs(points[:, 4])
        
        points_tensor = torch.tensor(points, dtype=torch.float32)
        graph = build_knn_graph(points_tensor, int(label), k=k)
        edge_index = graph.edge_index.numpy()
        n_edges = edge_index.shape[1]
        
        # Col 1: Combined channel image
        ax = axes[row, 0]
        combined = img[0] + img[1] + img[2]
        vmax = max(combined.max(), 1e-5)
        im = ax.imshow(combined, cmap="hot", vmin=0, vmax=vmax)
        ax.set_ylabel(f"Sample {row} ({jet_type})", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        
        total_pixels = 125 * 125
        density = n_nodes / total_pixels * 100
        ax.text(0.02, 0.98, f"{density:.1f}% active",
                transform=ax.transAxes, fontsize=9, verticalalignment="top", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        
        # Col 2: Point cloud
        ax = axes[row, 1]
        scatter = ax.scatter(phi, eta, c=total_energy, cmap="hot", s=6, alpha=0.8, edgecolors="none")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)
        ax.text(0.02, 0.98, f"{n_nodes} nodes",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Col 3: k-NN graph
        ax = axes[row, 2]
        segments = []
        for i in range(n_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            segments.append([(phi[src], eta[src]), (phi[dst], eta[dst])])
        lc = LineCollection(segments, colors="steelblue", linewidths=0.3, alpha=0.2)
        ax.add_collection(lc)
        ax.scatter(phi, eta, c=total_energy, cmap="hot", s=8, zorder=5, edgecolors="black", linewidths=0.2)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)
        ax.text(0.02, 0.98, f"{n_edges} edges",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "task2_pipeline_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved pipeline summary → %s", path)


def create_point_cloud_summary(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
) -> None:
    """Save point-cloud-only plots with x/y/intensity colouring."""
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 1, figsize=(7, n_samples * 4.2), squeeze=False)
    fig.suptitle("Point Cloud Samples (x, y, intensity)", fontsize=16, fontweight="bold", y=1.01)

    for row in range(n_samples):
        img = images[row]
        label = labels[row]
        points = image_to_pointcloud(img, threshold=0.0)
        eta, phi = points[:, 0], points[:, 1]
        intensity = np.abs(points[:, 2]) + np.abs(points[:, 3]) + np.abs(points[:, 4])

        ax = axes[row, 0]
        sc = ax.scatter(phi, eta, c=intensity, cmap="viridis", s=10, alpha=0.85, edgecolors="none")
        ax.set_title(f"Sample {row} — {'Quark' if label == 0 else 'Gluon'}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x / phi")
        ax.set_ylabel("y / eta")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Intensity")

    plt.tight_layout()
    path = os.path.join(out_dir, "point_cloud_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved point cloud summary → %s", path)


def create_graph_summary(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    k: int = 8,
) -> None:
    """Save graph-only plots with edges drawn on top of nodes."""
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 1, figsize=(7, n_samples * 4.2), squeeze=False)
    fig.suptitle("Graph Samples (with edges)", fontsize=16, fontweight="bold", y=1.01)

    for row in range(n_samples):
        img = images[row]
        label = labels[row]
        points = image_to_pointcloud(img, threshold=0.0)
        eta, phi = points[:, 0], points[:, 1]
        intensity = np.abs(points[:, 2]) + np.abs(points[:, 3]) + np.abs(points[:, 4])
        graph = build_knn_graph(torch.tensor(points, dtype=torch.float32), int(label), k=k)
        edge_index = graph.edge_index.numpy()

        ax = axes[row, 0]
        for edge in range(edge_index.shape[1]):
            src, dst = edge_index[0, edge], edge_index[1, edge]
            ax.plot([phi[src], phi[dst]], [eta[src], eta[dst]], color="#94A3B8", alpha=0.2, lw=0.4)
        ax.scatter(phi, eta, c=intensity, cmap="viridis", s=12, alpha=0.9, edgecolors="black", linewidths=0.2)
        ax.set_title(f"Sample {row} — {'Quark' if label == 0 else 'Gluon'}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x / phi")
        ax.set_ylabel("y / eta")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    plt.tight_layout()
    path = os.path.join(out_dir, "graph_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved graph summary → %s", path)


def main():
    parser = argparse.ArgumentParser(description="Task 2 Pipeline Visualization")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--k", type=int, default=8, help="k for k-NN graph construction")
    parser.add_argument("--max-events", type=int, default=500, help="Events to load")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    out_dir = os.path.join(OUTPUT_DIR, "graphs")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    X, y = load_dataset(max_events=args.max_events)
    channel_scales = compute_channel_scales(X, use_log1p=True)
    X_norm = normalize_channels(X, channel_scales, use_log1p=True)
    
    # Pick diverse samples (mix of quark and gluon)
    np.random.seed(args.seed)
    quark_idx = np.where(y == 0)[0]
    gluon_idx = np.where(y == 1)[0]
    
    n_quark = args.n_samples // 2
    n_gluon = args.n_samples - n_quark
    
    selected = np.concatenate([
        np.random.choice(quark_idx, size=min(n_quark, len(quark_idx)), replace=False),
        np.random.choice(gluon_idx, size=min(n_gluon, len(gluon_idx)), replace=False),
    ])
    
    logger.info("Generating pipeline visualizations for %d samples (k=%d)...", len(selected), args.k)
    
    # Individual detailed figures
    for i, idx in enumerate(selected):
        visualize_single_sample(X_norm[idx], int(y[idx]), i, out_dir, k=args.k)

    # Combined summary figure
    create_summary_figure(X_norm[selected], y[selected], out_dir, k=args.k, n_samples=len(selected))
    create_point_cloud_summary(X_norm[selected], y[selected], out_dir)
    create_graph_summary(X_norm[selected], y[selected], out_dir, k=args.k)
    
    logger.info("All visualizations saved to %s", out_dir)


if __name__ == "__main__":
    main()
