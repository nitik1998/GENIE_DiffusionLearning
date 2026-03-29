"""task2_gnn.py — Common Task 2
================================
GraphSAGE classifier for quark/gluon jet classification.

Pipeline:
  1. Convert 125×125 jet images → point clouds (non-zero pixels)
  2. Build k-NN graphs in (η, φ) angular space
  3. Train GraphSAGE classifier on graph representations
  4. Evaluate with ROC-AUC and visualize results

Usage:
    # Quick local test
    python src/task2_gnn.py --max-events 100 --epochs 3 --force-cpu

    # Full training (GPU)
    python src/task2_gnn.py --epochs 15 --batch-size 64

Outputs (saved to outputs/task2/<exp_name>/):
    task2_pipeline.png    — image → point cloud → graph conversion
    task2_graph_stats.png — node/edge distribution histograms
    task2_roc.png         — ROC curve with AUC
    task2_confusion.png   — confusion matrix
    metrics.json          — classification metrics
"""

import os
import sys
import argparse
import copy
import json
import shutil
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    setup_logging, set_seed, get_device, ensure_dirs, ensure_task_dirs,
    CHECKPOINT_DIR, OUTPUT_DIR, CHANNEL_NAMES, IMAGE_SIZE,
    DataConfig, TrainConfig, GNNConfig, get_auto_batch_size,
)
from src.data_utils import (
    load_dataset,
    image_to_pointcloud,
    make_splits,
    compute_channel_scales,
    normalize_channels,
)
from src.models.gnn import GraphSAGEClassifier, EdgeConvClassifier, build_knn_graph
from src.experiment_tracker import log_experiment, save_run_metrics

logger = setup_logging("task2")


def save_config(out_dir: str, run_params: Dict[str, float | int | str]) -> None:
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2)


def save_summary(out_dir: str, exp_name: str, run_params: Dict[str, float | int | str], metrics: Dict[str, float]) -> None:
    lines = [
        f"# {exp_name}",
        "",
        "## Scientific Framing",
        "",
        "- Sparse detector hits are represented as graph nodes rather than treated as dense RGB-style pixels.",
        f"- Active channel order: {', '.join(CHANNEL_NAMES)}.",
        "- Each node stores normalized position, detector-specific intensities, and centroid-relative radius.",
        "- Each edge stores deterministic spatial relations: delta_eta, delta_phi, distance, intensity difference.",
        "",
        "## Config",
        "",
    ]
    for key, value in sorted(run_params.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Metrics", ""])
    for key, value in sorted(metrics.items()):
        lines.append(f"- `{key}`: `{value:.6f}`")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def augment_point_features(points: np.ndarray) -> np.ndarray:
    """
    Add one physics-aware geometric feature: distance to the intensity-weighted
    centroid of the active detector hits in the event.
    """
    if points.shape[0] == 0:
        return np.zeros((1, 6), dtype=np.float32)

    eta_phi = points[:, :2]
    intensities = points[:, 2:5]
    total_e = intensities.sum(axis=1)

    if float(total_e.sum()) > 1e-12:
        centroid = (eta_phi * total_e[:, None]).sum(axis=0) / total_e.sum()
    else:
        centroid = eta_phi.mean(axis=0)

    r_centroid = np.linalg.norm(eta_phi - centroid[None, :], axis=1, keepdims=True)
    return np.concatenate([points, r_centroid.astype(np.float32)], axis=1).astype(np.float32)


def pooled_graph_features(dataset: "JetGraphDataset") -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for i in range(len(dataset)):
        graph = dataset[i]
        feats.append(graph.x.numpy().mean(axis=0))
        labels.append(int(graph.y.item()))
    feats = np.stack(feats, axis=0)
    labels = np.array(labels, dtype=np.int64)
    return feats, labels


def evaluate_logistic_baseline(
    train_ds: "JetGraphDataset", val_ds: "JetGraphDataset", test_ds: "JetGraphDataset"
) -> Dict[str, float]:
    X_train, y_train = pooled_graph_features(train_ds)
    X_val, y_val = pooled_graph_features(val_ds)
    X_test, y_test = pooled_graph_features(test_ds)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    return {
        "baseline_val_auc": float(roc_auc_score(y_val, val_prob)),
        "baseline_test_auc": float(roc_auc_score(y_test, test_prob)),
        "baseline_test_accuracy": float(accuracy_score(y_test, test_pred)),
        "baseline_test_f1": float(f1_score(y_test, test_pred)),
    }


# ─────────────────────────────────────────────────────────────
# Graph Dataset
# ─────────────────────────────────────────────────────────────

class JetGraphDataset(Dataset):
    """
    Lazily builds graph representations from jet images.

    This avoids materialising the full graph set in RAM up front, which is
    important for full-dataset Colab runs.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, knn_k: int = 8, tag: str = "data"
    ) -> None:
        logger.info("Preparing %s graph dataset (n=%s, k=%d)...", tag, f"{len(y):,}", knn_k)
        self.X = X
        self.y = y
        self.knn_k = knn_k
        self.channel_scales = compute_channel_scales(X, use_log1p=True)

    @staticmethod
    def _build_one(img: np.ndarray, label: int, knn_k: int) -> Data:
        points = augment_point_features(image_to_pointcloud(img))
        pts_tensor = torch.tensor(points, dtype=torch.float32)
        return build_knn_graph(pts_tensor, label, k=knn_k)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Data:
        img = normalize_channels(
            self.X[idx : idx + 1], self.channel_scales, use_log1p=True
        )[0]
        return self._build_one(img, int(self.y[idx]), self.knn_k)


def collate_graphs(batch: List[Data]) -> Batch:
    """Collate PyG Data objects into a Batch."""
    return Batch.from_data_list(batch)


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """One training epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False, desc="Training"):
        batch = batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate and return loss, predicted logits, and true labels."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.float())
        total_loss += loss.item() * batch.num_graphs
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    return total_loss / len(loader.dataset), np.array(all_logits), np.array(all_labels)


# ─────────────────────────────────────────────────────────────
# Pipeline Visualization (Image → Point Cloud → Graph)
# ─────────────────────────────────────────────────────────────

def plot_pipeline(
    X: np.ndarray, y: np.ndarray, knn_k: int, out_dir: str, n_show: int = 2,
) -> None:
    """
    Visualise the full image → point cloud → graph conversion pipeline.

    For each sample jet, shows:
      Column 1-3: Original 3-channel image
      Column 4:   Point cloud scatter (eta vs phi, colour = total energy)
      Column 5:   k-NN graph with edges drawn
    """
    n_show = min(n_show, len(y))
    fig, axes = plt.subplots(n_show, 5, figsize=(25, n_show * 4.5), squeeze=False)
    fig.suptitle(
        "Task 2 — Pipeline: Image → Point Cloud → k-NN Graph",
        fontsize=15, fontweight="bold", y=1.02,
    )

    channel_scales = compute_channel_scales(X, use_log1p=True)
    X_norm = normalize_channels(X, channel_scales, use_log1p=True)

    for row in range(n_show):
        img = X_norm[row]  # (3, 125, 125)
        label = "Quark" if y[row] == 0 else "Gluon"

        # Columns 0-2: Original channels
        for ch in range(3):
            raw_ch = X[row, ch]
            vmax = max(raw_ch.max(), 1e-6)
            axes[row, ch].imshow(raw_ch, cmap="hot", vmin=0, vmax=vmax)
            axes[row, ch].axis("off")
            if row == 0:
                axes[row, ch].set_title(f"{CHANNEL_NAMES[ch]}", fontsize=12)
            if ch == 0:
                axes[row, ch].set_ylabel(f"{label}", fontsize=12, fontweight="bold")

        # Convert to point cloud
        points = augment_point_features(image_to_pointcloud(img))
        eta = points[:, 0]
        phi = points[:, 1]
        total_e = points[:, 2] + points[:, 3] + points[:, 4]

        # Column 3: Point cloud scatter
        ax_pc = axes[row, 3]
        sc = ax_pc.scatter(eta, phi, c=total_e, cmap="viridis", s=8, alpha=0.7, edgecolors="none")
        ax_pc.set_xlim(-1.05, 1.05)
        ax_pc.set_ylim(-1.05, 1.05)
        ax_pc.set_aspect("equal")
        ax_pc.grid(alpha=0.2)
        if row == 0:
            ax_pc.set_title(f"Point Cloud ({len(points)} pts)", fontsize=12)
        else:
            ax_pc.set_title(f"({len(points)} pts)", fontsize=10)
        ax_pc.set_xlabel("eta (normalised)")
        ax_pc.set_ylabel("phi (normalised)")
        plt.colorbar(sc, ax=ax_pc, fraction=0.046, pad=0.04, label="Energy")

        # Column 4: k-NN graph
        pts_tensor = torch.tensor(points, dtype=torch.float32)
        graph = build_knn_graph(pts_tensor, int(y[row]), k=knn_k)
        edge_index = graph.edge_index.numpy()

        ax_g = axes[row, 4]
        # Draw edges first (faint)
        for e in range(edge_index.shape[1]):
            s_idx, d_idx = edge_index[0, e], edge_index[1, e]
            ax_g.plot(
                [eta[s_idx], eta[d_idx]], [phi[s_idx], phi[d_idx]],
                color="#94A3B8", alpha=0.15, lw=0.4,
            )
        # Draw nodes on top
        ax_g.scatter(eta, phi, c=total_e, cmap="viridis", s=10, alpha=0.8, edgecolors="none", zorder=5)
        ax_g.set_xlim(-1.05, 1.05)
        ax_g.set_ylim(-1.05, 1.05)
        ax_g.set_aspect("equal")
        ax_g.grid(alpha=0.2)
        n_edges = edge_index.shape[1]
        if row == 0:
            ax_g.set_title(f"k-NN Graph (k={knn_k}, {n_edges} edges)", fontsize=12)
        else:
            ax_g.set_title(f"({n_edges} edges)", fontsize=10)
        ax_g.set_xlabel("eta (normalised)")
        ax_g.set_ylabel("phi (normalised)")

    plt.tight_layout()
    path = os.path.join(out_dir, "task2_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved pipeline visualization -> %s", path)
    shutil.copy2(path, os.path.join(out_dir, "image_to_pointcloud_graph_pipeline.png"))


def plot_graph_stats(
    X: np.ndarray, y: np.ndarray, knn_k: int, out_dir: str, n_sample: int = 500,
) -> None:
    """
    Histogram of node counts and edge counts across sample jets.
    Helps reviewers understand graph size variability.
    """
    n_sample = min(n_sample, len(y))
    channel_scales = compute_channel_scales(X, use_log1p=True)
    X_norm = normalize_channels(X, channel_scales, use_log1p=True)

    node_counts = []
    edge_counts = []
    for i in range(n_sample):
        pts = augment_point_features(image_to_pointcloud(X_norm[i]))
        pts_t = torch.tensor(pts, dtype=torch.float32)
        g = build_knn_graph(pts_t, int(y[i]), k=knn_k)
        node_counts.append(pts.shape[0])
        edge_counts.append(g.edge_index.shape[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Task 2 — Graph Construction Statistics", fontsize=13, fontweight="bold")

    ax1.hist(node_counts, bins=50, color="#2563EB", alpha=0.8, edgecolor="none")
    ax1.axvline(np.mean(node_counts), color="#DC2626", ls="--", lw=2,
                label=f"Mean: {np.mean(node_counts):.0f}")
    ax1.set_xlabel("Number of Nodes (active pixels)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Node Count Distribution", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.hist(edge_counts, bins=50, color="#059669", alpha=0.8, edgecolor="none")
    ax2.axvline(np.mean(edge_counts), color="#DC2626", ls="--", lw=2,
                label=f"Mean: {np.mean(edge_counts):.0f}")
    ax2.set_xlabel(f"Number of Edges (k={knn_k})", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Edge Count Distribution", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "task2_graph_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved graph stats -> %s", path)


def plot_graph_stats_by_class(
    X: np.ndarray, y: np.ndarray, knn_k: int, out_dir: str, n_sample: int = 500,
) -> None:
    n_sample = min(n_sample, len(y))
    channel_scales = compute_channel_scales(X, use_log1p=True)
    X_norm = normalize_channels(X, channel_scales, use_log1p=True)
    stats = {"Quark": {"nodes": [], "edges": [], "density": []}, "Gluon": {"nodes": [], "edges": [], "density": []}}

    for i in range(n_sample):
        cls = "Quark" if int(y[i]) == 0 else "Gluon"
        pts = augment_point_features(image_to_pointcloud(X_norm[i]))
        pts_t = torch.tensor(pts, dtype=torch.float32)
        g = build_knn_graph(pts_t, int(y[i]), k=knn_k)
        stats[cls]["nodes"].append(float(pts.shape[0]))
        stats[cls]["edges"].append(float(g.edge_index.shape[1]))
        stats[cls]["density"].append(float(pts.shape[0]) / float(IMAGE_SIZE * IMAGE_SIZE))

    labels = ["Nodes", "Edges", "Active Density"]
    quark_vals = [np.mean(stats["Quark"]["nodes"]), np.mean(stats["Quark"]["edges"]), np.mean(stats["Quark"]["density"])]
    gluon_vals = [np.mean(stats["Gluon"]["nodes"]), np.mean(stats["Gluon"]["edges"]), np.mean(stats["Gluon"]["density"])]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, quark_vals, width, label="Quark", color="#2563EB")
    ax.bar(x + width / 2, gluon_vals, width, label="Gluon", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Task 2 — Average Graph Statistics by Class", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, "graph_stats_by_class.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved classwise graph stats → %s", path)


def plot_classwise_example_graphs(
    X: np.ndarray,
    y: np.ndarray,
    knn_k: int,
    out_dir: str,
) -> None:
    class_indices = {
        "Quark": int(np.where(y == 0)[0][0]),
        "Gluon": int(np.where(y == 1)[0][0]),
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    fig.suptitle("Task 2 — Classwise Example Detector Graphs", fontsize=15, fontweight="bold", y=1.02)

    channel_scales = compute_channel_scales(X, use_log1p=True)
    X_norm = normalize_channels(X, channel_scales, use_log1p=True)

    for row, (cls, idx) in enumerate(class_indices.items()):
        img_raw = X[idx]
        img_norm = X_norm[idx]
        points = augment_point_features(image_to_pointcloud(img_norm))
        graph = build_knn_graph(torch.tensor(points, dtype=torch.float32), int(y[idx]), k=knn_k)
        eta, phi = points[:, 0], points[:, 1]
        total_e = points[:, 2] + points[:, 3] + points[:, 4]
        edge_index = graph.edge_index.numpy()

        for ch in range(3):
            ax = axes[row, ch]
            vmax = max(float(img_raw[ch].max()), 1e-6)
            ax.imshow(img_raw[ch], cmap="hot", vmin=0, vmax=vmax)
            ax.axis("off")
            ax.set_title(f"{cls} — {CHANNEL_NAMES[ch]}", fontsize=10)

        ax = axes[row, 3]
        for e in range(edge_index.shape[1]):
            s_idx, d_idx = edge_index[0, e], edge_index[1, e]
            ax.plot([eta[s_idx], eta[d_idx]], [phi[s_idx], phi[d_idx]], color="#94A3B8", alpha=0.12, lw=0.4)
        ax.scatter(eta, phi, c=total_e, cmap="viridis", s=10, edgecolors="none")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{cls} Graph ({len(points)} nodes)", fontsize=10)
        ax.set_xlabel("eta")
        ax.set_ylabel("phi")

    path = os.path.join(out_dir, "classwise_example_graphs.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved classwise example graphs → %s", path)


# ─────────────────────────────────────────────────────────────
# Classification Visualization
# ─────────────────────────────────────────────────────────────

def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_val: float,
    out_dir: str,
    baseline_auc: float | None = None,
) -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"Task 2 GNN (AUC = {auc_val:.4f})", color="#2563EB", lw=2.5)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    if baseline_auc is not None:
        ax.text(
            0.98, 0.08,
            f"Mean-pooled logistic baseline AUC = {baseline_auc:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1"),
        )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Task 2 — ROC: Quark vs Gluon Jet Classification", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    path = os.path.join(out_dir, "task2_roc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve → %s", path)
    shutil.copy2(path, os.path.join(out_dir, "roc_curve.png"))


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str) -> None:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Quark", "Gluon"])
    ax.set_yticklabels(["Quark", "Gluon"])
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(out_dir, "task2_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", path)
    shutil.copy2(path, os.path.join(out_dir, "confusion_matrix.png"))


def plot_training_curves(
    train_losses: List[float],
    val_aucs: List[float],
    out_dir: str,
) -> None:
    """Save Task 2 training curves for Colab and local review."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_losses, color="#2563EB", lw=2)
    axes[0].set_title("Train Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_aucs, color="#059669", lw=2)
    axes[1].set_title("Validation ROC-AUC", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", path)


@torch.no_grad()
def plot_graph_embedding_projection(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str, max_graphs: int = 400) -> None:
    model.eval()
    embeddings = []
    labels = []
    collected = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        emb = model.encode_graph(batch.x, batch.edge_index, batch.batch).cpu().numpy()
        lbl = batch.y.cpu().numpy()
        embeddings.append(emb)
        labels.append(lbl)
        collected += emb.shape[0]
        if collected >= max_graphs:
            break
    if not embeddings:
        return
    X_emb = np.concatenate(embeddings, axis=0)[:max_graphs]
    y_emb = np.concatenate(labels, axis=0)[:max_graphs]
    X_centered = X_emb - X_emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    proj = X_centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color, name in [(0, "#2563EB", "Quark"), (1, "#DC2626", "Gluon")]:
        mask = y_emb == cls
        ax.scatter(proj[mask, 0], proj[mask, 1], s=18, alpha=0.7, c=color, label=name, edgecolors="none")
    ax.set_title("Task 2 — Graph Embedding Projection", fontsize=13, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(alpha=0.3)
    ax.legend()
    path = os.path.join(out_dir, "graph_embedding_projection.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved graph embedding projection → %s", path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(args.force_cpu)
    ensure_dirs()
    out_dir = ensure_task_dirs("task2", args.exp_name)
    metrics_path = os.path.join(out_dir, "metrics.json")
    checkpoint_path = os.path.join(out_dir, "model_checkpoint.pt")
    if not args.force_rerun and os.path.exists(metrics_path) and os.path.exists(checkpoint_path):
        logger.info("Skipping completed experiment %s; outputs already exist in %s", args.exp_name, out_dir)
        return
    logger.info("Experiment: %s → %s", args.exp_name, out_dir)

    # Load and split
    X, y = load_dataset(max_events=args.max_events)
    train_idx, val_idx, test_idx = make_splits(X, y, seed=args.seed)

    # Build graph datasets
    gnn_cfg = GNNConfig(knn_k=args.knn_k)
    loaders = {}
    batch_size = args.batch_size if args.batch_size > 0 else get_auto_batch_size(task_num=2)
    if args.batch_size <= 0:
        logger.info("Auto-scaled batch size to %d based on VRAM", batch_size)

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ds = JetGraphDataset(X[idx], y[idx], knn_k=gnn_cfg.knn_k, tag=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=collate_graphs, num_workers=0,
        )

    # Model
    if args.model_type == "edgeconv":
        model = EdgeConvClassifier(
            in_channels=6,
            hidden=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        model = GraphSAGEClassifier(
            in_channels=6,
            hidden=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s — %s trainable parameters", model.__class__.__name__, f"{n_params:,}")

    # Class weighting for imbalanced data
    pos_weight = torch.tensor(
        [y[train_idx].mean() / (1 - y[train_idx].mean() + 1e-6)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    baseline_metrics = evaluate_logistic_baseline(loaders["train"].dataset, loaders["val"].dataset, loaders["test"].dataset)
    logger.info(
        "Mean-pooled logistic baseline — val AUC: %.4f | test AUC: %.4f | acc: %.4f",
        baseline_metrics["baseline_val_auc"],
        baseline_metrics["baseline_test_auc"],
        baseline_metrics["baseline_test_accuracy"],
    )

    # Training loop with early stopping
    logger.info("Starting GNN training for %d epochs (patience=%d)...", args.epochs, args.patience)
    best_auc = 0.0
    best_state = None
    patience_counter = 0
    train_losses, val_aucs = [], []
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"gnn_{args.exp_name}.pt")

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in pbar:
        tr_loss = train_epoch(model, loaders["train"], optimizer, criterion, scaler, device)
        _, va_log, va_y = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        va_prob = 1.0 / (1.0 + np.exp(-va_log))  # Sigmoid
        va_auc = roc_auc_score(va_y, va_prob)
        train_losses.append(tr_loss)
        val_aucs.append(va_auc)

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *"
            torch.save(best_state, ckpt_path)
        else:
            patience_counter += 1
            marker = ""

        pbar.set_postfix(loss=f"{tr_loss:.4f}", auc=f"{va_auc:.4f}")
        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {tr_loss:.4f} | "
            f"Val AUC: {va_auc:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}{marker}",
            flush=True,
        )

        if patience_counter >= args.patience and args.patience > 0:
            print(f"Early stopping at epoch {epoch} (no AUC improvement for {args.patience} epochs)", flush=True)
            break

    # Test evaluation
    model.load_state_dict(best_state)
    _, te_log, te_y = eval_epoch(model, loaders["test"], criterion, device)
    te_prob = 1.0 / (1.0 + np.exp(-te_log))
    te_auc = roc_auc_score(te_y, te_prob)
    te_pred = (te_prob > 0.5).astype(int)
    te_acc = accuracy_score(te_y, te_pred)
    te_f1 = f1_score(te_y, te_pred)

    logger.info("═══ Test Set Results ═══")
    logger.info("  ROC-AUC:  %.4f", te_auc)
    logger.info("  Accuracy: %.4f", te_acc)

    # Save metrics (structured)
    metrics = {"roc_auc": te_auc, "accuracy": te_acc, "f1": te_f1, **baseline_metrics}
    run_params = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": batch_size,
        "knn_k": gnn_cfg.knn_k,
        "max_events": args.max_events,
        "seed": args.seed,
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    save_run_metrics(out_dir, metrics, run_params)
    save_config(out_dir, run_params)
    save_summary(out_dir, args.exp_name, run_params, metrics)

    metrics_path = os.path.join(out_dir, "task2_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Common Task 2 — GNN Jet Classification Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Graph construction: k-NN (k={gnn_cfg.knn_k})\n")
        f.write(f"Channel order: {', '.join(CHANNEL_NAMES)}\n")
        f.write("Node features: [eta_norm, phi_norm, E_Tracks, E_ECAL, E_HCAL, r_centroid]\n")
        f.write("Edge features: [delta_eta, delta_phi, distance, delta_intensity]\n\n")
        f.write(f"Test ROC-AUC:  {te_auc:.4f}\n")
        f.write(f"Test Accuracy: {te_acc:.4f}\n")
        f.write(f"Test F1:       {te_f1:.4f}\n")
        f.write(f"Baseline ROC-AUC (mean-pooled logistic): {baseline_metrics['baseline_test_auc']:.4f}\n")
    torch.save(
        {
            "model_state_dict": best_state,
            "params": run_params,
            "metrics": metrics,
        },
        os.path.join(out_dir, "model_checkpoint.pt"),
    )

    # Pipeline visualizations (using raw data)
    logger.info("Generating pipeline visualizations...")
    plot_pipeline(X[test_idx], y[test_idx], gnn_cfg.knn_k, out_dir, n_show=2)
    plot_graph_stats(X[test_idx], y[test_idx], gnn_cfg.knn_k, out_dir,
                     n_sample=min(500, len(test_idx)))
    plot_classwise_example_graphs(X[test_idx], y[test_idx], gnn_cfg.knn_k, out_dir)
    plot_graph_stats_by_class(X[test_idx], y[test_idx], gnn_cfg.knn_k, out_dir,
                              n_sample=min(500, len(test_idx)))

    # Classification plots
    plot_training_curves(train_losses, val_aucs, out_dir)
    plot_roc(te_y, te_prob, te_auc, out_dir, baseline_auc=baseline_metrics["baseline_test_auc"])
    plot_confusion(te_y, te_pred, out_dir)
    plot_graph_embedding_projection(model, loaders["test"], device, out_dir)

    # Log experiment
    log_experiment("task2", args.exp_name, run_params, metrics, status="SUCCESS")

    logger.info("Task 2 complete. Outputs -> %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Common Task 2 — GNN Jet Classifier")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=auto scale based on VRAM)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0=disabled)")
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--model-type", choices=["edgeconv", "graphsage"], default="graphsage")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun experiment even if outputs already exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="baseline", help="Experiment name for output dir")
    main(parser.parse_args())
