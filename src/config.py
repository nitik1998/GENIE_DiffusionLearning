"""
config.py — Shared Configuration
==================================
Centralised configuration dataclasses and utility functions used across all
three evaluation tasks. Eliminates magic numbers and ensures reproducibility.

Design: immutable dataclasses enforce single-source-of-truth for every
hyper-parameter. All paths are resolved relative to project root.
"""

import os
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
# Path Resolution
# ─────────────────────────────────────────────────────────────

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = str(Path(os.environ.get("GENIE_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)).resolve())
DATA_DIR = str(Path(os.environ.get("GENIE_DATA_DIR", Path(PROJECT_ROOT) / "data")).resolve())
OUTPUT_DIR = str(Path(os.environ.get("GENIE_OUTPUT_DIR", Path(PROJECT_ROOT) / "outputs")).resolve())
CHECKPOINT_DIR = str(Path(os.environ.get("GENIE_CHECKPOINT_DIR", Path(PROJECT_ROOT) / "checkpoints")).resolve())

CHANNEL_NAMES = ("Tracks", "ECAL", "HCAL")
IMAGE_SIZE = 125
N_CHANNELS = 3


# ─────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """Immutable data pipeline configuration."""
    data_dir: str = DATA_DIR
    max_events: Optional[int] = None
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac is implicitly 1 - train_frac - val_frac


@dataclass(frozen=True)
class TrainConfig:
    """Shared training hyper-parameters."""
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    force_cpu: bool = False
    num_workers: int = 0  # Safest default for Colab and local


@dataclass(frozen=True)
class AEConfig:
    """Autoencoder-specific hyper-parameters."""
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    use_skip_connections: bool = True
    lr_min: float = 1e-5


@dataclass(frozen=True)
class GNNConfig:
    """Graph Attention Network hyper-parameters."""
    knn_k: int = 8
    gat_hidden: int = 32
    gat_heads: int = 4
    gat_layers: int = 3
    dropout: float = 0.3
    gat_dropout: float = 0.1


@dataclass(frozen=True)
class DDPMConfig:
    """Denoising Diffusion Probabilistic Model hyper-parameters."""
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    unet_channels: Tuple[int, ...] = (64, 128, 256)
    time_embed_dim: int = 256
    num_res_blocks: int = 2
    use_attention: bool = True
    attention_resolution: int = 15  # Apply attention at 15x15


# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure structured logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def set_seed(seed: int = 42) -> None:
    """
    Fix all sources of stochasticity for deterministic execution.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU + CUDA).
    Also configures CuDNN for deterministic operation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Auto-detect the best available hardware accelerator.

    Priority: CUDA GPU → CPU
    Logs VRAM information when GPU is available for memory planning.
    """
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logging.getLogger(__name__).info(
            "GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb
        )
        return device
    logging.getLogger(__name__).info("Using CPU backend")
    return torch.device("cpu")


def get_auto_batch_size(task_num: int) -> int:
    """
    Dynamically scale batch size based on available GPU VRAM to prevent OOM
    and maximize training efficiency on high-end GPUs like A100.
    """
    if not torch.cuda.is_available():
        return 32  # Safe CPU default

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    if task_num == 3:  # DDPM (Resource heavy)
        if vram_gb < 12:
            return 16   # RTX 3060/4060, RTX 5060 (8GB)
        elif vram_gb < 20:
            return 32   # Colab T4 (15GB), RTX 4080 (16GB)
        elif vram_gb < 30:
            return 64   # RTX 3090/4090 (24GB)
        else:
            return 128  # Colab A100 (80GB)
    else:  # Task 1 Autoencoder & Task 2 GNN (Lighter)
        if vram_gb < 12:
            return 64
        elif vram_gb < 20:
            return 128
        elif vram_gb < 30:
            return 256
        else:
            return 512


def ensure_dirs() -> None:
    """Create output and checkpoint directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def ensure_task_dirs(task: str, exp_name: str = "default") -> str:
    """
    Create and return a structured experiment output directory.

    Returns:
        Path like ``outputs/task1/baseline/``, guaranteed to exist.
    """
    path = os.path.join(OUTPUT_DIR, task, exp_name)
    os.makedirs(path, exist_ok=True)
    return path
