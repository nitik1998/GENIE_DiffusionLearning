# GENIE — Learning Latent Structure with Diffusion Models

> **ML4SCI GSoC 2026 Evaluation** — Quark/Gluon Jet Generation & Classification  
> *Applicant: Nitik Jain · Mentor Organization: [ML4SCI](https://ml4sci.org/)*

This repository implements a three-task evaluation pipeline for the ML4SCI GENIE project, working with 139,306 calorimeter jet images from the [Quark/Gluon Dataset](https://cernbox.cern.ch/index.php/s/b5WrtcHe0xQ26M4). Each task builds on the previous one — from learning compact representations, to graph-based classification, to generative modeling in latent space.

---

## Results at a Glance

| Task | Model | Key Metric | Value |
|------|-------|------------|-------|
| **Task 1** — Reconstruction | DeepFalcon VAE | PSNR / SSIM | **37.93 dB / 0.967** |
| **Task 2** — Classification | GraphSAGE (k-NN) | ROC-AUC / Accuracy | **0.774 / 70.6%** |
| **Task 3** — Generation | Latent Diffusion (DDPM) | PSNR / SSIM | **30.32 dB / 0.931** |

---

## Task 1 — Variational Autoencoder

A convolutional VAE (**DeepFalconVAE**) with transpose-convolution decoder learns a 256-dimensional latent space from three-channel calorimeter images (Tracks, ECAL, HCAL). The model uses a custom `detector_reference` preprocessing pipeline with per-channel normalization tailored to the physics of each sub-detector.

### Architecture
- **Encoder:** Conv2d (stride-2) → BatchNorm → LeakyReLU, 4 downsampling stages `[32, 64, 128, 256]`
- **Latent:** 256-dim with KL-divergence warmup over 100 epochs
- **Decoder:** ConvTranspose2d with BatchNorm, symmetric to encoder
- **Loss:** MSE (α=0.7 mix) + nonzero-weighted reconstruction (4×) + KL

### Training
- **200 epochs**, batch size 512, AdamW (lr=1e-3), cosine annealing
- Trained on NVIDIA H100 (80GB), ~15 minutes

### Results

| Metric | Value |
|--------|-------|
| PSNR | **37.93 dB** |
| SSIM | **0.967** |
| Active IoU | **0.998** |
| Background False Activation | **0.000** |

<p align="center">
  <img src="assets/task1_training_curves.png" width="48%" alt="Task 1 Training Curves"/>
  <img src="assets/task1_best_reconstruction.png" width="48%" alt="Task 1 Best Reconstruction"/>
</p>

<p align="center">
  <img src="assets/task1_original_vs_reconstructed.png" width="80%" alt="Task 1 Original vs Reconstructed"/>
</p>
<p align="center"><i>Original (left 3 columns) vs VAE Reconstruction (right 3 columns) — per-channel comparison across 8 test jets</i></p>

---

## Task 2 — Graph Neural Network Classification

Detector images are converted to **point clouds** (only active pixels) and connected into **k-NN graphs** (k=8) in (η, φ) space. A **GraphSAGE** classifier then distinguishes quark jets from gluon jets using the graph topology and multi-channel energy features.

### Pipeline
```
125×125×3 image → active pixel extraction → (η, φ) point cloud → k-NN graph (k=8) → GraphSAGE → quark/gluon
```

### Node & Edge Features
- **Node:** `[η_norm, φ_norm, E_tracks, E_ECAL, E_HCAL, r_centroid]` (6 features)
- **Edge:** `[Δη, Δφ, distance, ΔE]` (4 features)

### Results

| Metric | Baseline (MLP) | GraphSAGE |
|--------|----------------|-----------|
| ROC-AUC | 0.564 | **0.774** |
| Accuracy | 54.3% | **70.6%** |
| F1 Score | 0.562 | **0.727** |

<p align="center">
  <img src="assets/task2_training_curves.png" width="48%" alt="Task 2 Training Curves"/>
  <img src="assets/task2_roc_curve.png" width="48%" alt="Task 2 ROC Curve"/>
</p>

<p align="center">
  <img src="assets/task2_pipeline.png" width="80%" alt="Task 2 Image-to-Graph Pipeline"/>
</p>
<p align="center"><i>Image → Point Cloud → k-NN Graph conversion pipeline for a single jet event</i></p>

---

## Task 3 — Latent Diffusion

The main generative task: a **DDPM** denoiser operates directly in the **256-dim latent space** learned by the Task 1 VAE. This avoids the computational cost of pixel-space diffusion while preserving the sparse, physics-meaningful structure of jet images.

### Architecture
- **VAE Encoder/Decoder:** Frozen DeepFalconVAE from Task 1
- **Denoiser:** Time-conditioned residual MLP (`LatentDenoiser`)
  - 6 residual blocks, hidden_dim=1024, time_emb_dim=128
  - DiT-style timestep injection at every layer
- **Schedule:** 1000 DDPM timesteps, linear β schedule

### Key Design Decision
The VAE was trained with a custom `detector_reference` preprocessing. To ensure compatibility, Task 3 loads the exact preprocessing parameters from the VAE checkpoint — guaranteeing that the encoder receives correctly normalised data and the decoder produces physically meaningful outputs.

### Results

| Metric | Value |
|--------|-------|
| PSNR | **30.32 dB** |
| SSIM | **0.931** |
| MSE | **0.0009** |
| Generated Active Density | **66.67%** |
| Real Active Density | **66.77%** |

<p align="center">
  <img src="assets/task3_training_curves.png" width="48%" alt="Task 3 Training Curves"/>
  <img src="assets/task3_generated_samples.png" width="48%" alt="Task 3 Generated Samples"/>
</p>
<p align="center"><i>Left: Latent MSE loss convergence over 30 epochs. Right: Original (left) vs Generated (right) jet images — per-channel comparison</i></p>

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset
```bash
mkdir -p data
wget -O data/quark-gluon_data-set_n139306.hdf5 \
  https://cernbox.cern.ch/remote.php/dav/public-files/b5WrtcHe0xQ26M4/quark-gluon_data-set_n139306.hdf5
```

### Local Training

```bash
# Task 1 — VAE
python src/task1_autoencoder.py --epochs 200 --batch-size 512

# Task 2 — GNN
python src/task2_gnn.py --epochs 30 --exp-name task2_graph_classifier

# Task 3 — Latent Diffusion
python src/task3_diffusion.py --mode latent_diffusion --epochs 100 --exp-name latent_diffusion
```

### Colab Notebooks

Each task has a companion notebook in `notebooks/` with a `RUN_MODE` toggle:
- `"sanity"` — quick check run (3 epochs, 1000 events)
- `"full"` — full training run for final results

| Task | Notebook |
|------|----------|
| Task 1 | `notebooks/Task1_Autoencoder.ipynb` |
| Task 2 | `notebooks/Task2_Graph_Classifier.ipynb` |
| Task 3 | `notebooks/Task3_Diffusion_Exploration.ipynb` |

---

## Repository Structure

```
├── src/
│   ├── config.py                 # Shared configuration & utilities
│   ├── data_utils.py             # Data loading, normalization, splitting
│   ├── metrics.py                # PSNR, SSIM, active density metrics
│   ├── task1_autoencoder.py      # VAE training pipeline
│   ├── task2_gnn.py              # GNN classification pipeline
│   ├── task3_diffusion.py        # Latent diffusion pipeline
│   ├── experiment_tracker.py     # Experiment logging
│   └── models/
│       ├── autoencoder.py        # ConvAutoEncoder, DeepFalconVAE
│       ├── diffusion_core.py     # DDPM scheduler
│       ├── diffusion_unet.py     # SimpleUNet (pixel-space baseline)
│       └── latent_denoiser.py    # Residual MLP denoiser
├── notebooks/                    # Colab-ready notebooks
├── results_from_colab/           # Final trained models & metrics
├── docs/                         # Task reports & proposal ideas
├── assets/                       # README images
└── requirements.txt
```

---

## Pre-trained Models

All trained models and final results are saved in `results_from_colab/`:

| File | Contents |
|------|----------|
| `task1_autoencoder.zip` | VAE checkpoint, training curves, reconstructions |
| `task2_graph_classifier.zip` | GraphSAGE checkpoint, ROC curves, confusion matrix |
| `latent_diffusion.zip` | Latent denoiser checkpoint, generated samples |

---

## License

This project is developed as part of the [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) evaluation for [ML4SCI](https://ml4sci.org/).
