# GENIE — Learning Latent Structure with Diffusion Models

> **ML4SCI GSoC 2026 Evaluation** · Quark/Gluon Jet Generation & Classification  
> *Applicant: Nitik Jain · Organization: [ML4SCI](https://ml4sci.org/)*

This repository implements a three-task evaluation pipeline for the [GENIE project](https://ml4sci.org/gsoc/projects/2025/project_GENIE.html), working with 139,306 calorimeter jet images from the quark/gluon dataset [[1]](#references). Each task builds on the previous: learn compact latent representations (Task 1), classify jet type using graph structure (Task 2), and generate new physics-realistic jets via latent diffusion (Task 3).

---

## Results Summary

| Task | Model | Key Metric | Score |
|------|-------|------------|-------|
| **1** — Reconstruction | Convolutional VAE | PSNR / SSIM | **37.93 dB** / **0.967** |
| **2** — Classification | GraphSAGE on k-NN graphs | ROC-AUC / Accuracy | **0.774** / **70.6%** |
| **3** — Generation | Latent DDPM | PSNR / SSIM | **30.32 dB** / **0.931** |

---

## Dataset

The quark/gluon jet dataset contains **139,306** simulated proton–proton collision events, each represented as a `125 × 125 × 3` calorimeter image with three detector channels:

| Channel | Detector | Physics |
|---------|----------|---------|
| 0 | **Tracks** | Charged particle trajectories — extremely sparse (< 0.1% nonzero) |
| 1 | **ECAL** | Electromagnetic calorimeter — captures electron and photon showers |
| 2 | **HCAL** | Hadronic calorimeter — captures hadron energy deposits |

On average only **~0.06%** of pixels are nonzero per image. This extreme sparsity is the defining challenge: models must learn to reconstruct mostly-empty images with sharp, localized energy clusters — motivating graph-based representations [[1]](#references) and latent-space generation [[2]](#references).

---

## Metrics Guide

| Metric | Measures | Higher or Lower? |
|--------|----------|------------------|
| **MSE** | Pixel-wise reconstruction error | Lower ↓ |
| **Nonzero MSE** | Error on active detector cells only (ignores background) | Lower ↓ |
| **PSNR** | Signal-to-noise ratio in dB; ≥ 30 dB is good quality | Higher ↑ |
| **SSIM** | Structural/perceptual similarity [0–1] [[4]](#references) | Higher ↑ |
| **Active IoU** | Overlap of nonzero pixels between original and output | Higher ↑ |
| **ROC-AUC** | Classification power; 0.5 = random, 1.0 = perfect | Higher ↑ |
| **Active Density** | Fraction of nonzero pixels — validates sparsity of generated images | Match ≈ |

---

## Task 1 — Variational Autoencoder

**Objective:** Learn a compact 256-dimensional latent representation of jet images that faithfully reconstructs the sparse energy deposits across all three detector channels.

**Model:** A 5-stage convolutional VAE (`JetVAE`) with transpose-convolution decoder. Uses a physics-informed `detector_reference` preprocessing:
- **Tracks** — log-scaled with percentile boosting (compensates for extreme sparsity)
- **ECAL / HCAL** — standard mean/std normalization mapped to [0, 1]

**Training:** 200 epochs · batch 512 · Adam (lr = 1e-3) · cosine annealing · KL warmup 100 epochs · nonzero pixels weighted 4×.

### Results

| Metric | Score |
|--------|-------|
| PSNR | **37.93 dB** |
| SSIM | **0.967** |
| Active IoU | **0.998** |
| False Activation Rate | **0.000** |

---
<p align="center">
  <img src="assets/task1_training_curves.png" width="70%" alt="Figure 1a"/>
</p>
<p align="center"><em>Figure 1a — Training and validation loss over 200 epochs.</em></p>

<p align="center">
  <img src="assets/task1_best_reconstruction.png" width="70%" alt="Figure 1b"/>
</p>
<p align="center"><em>Figure 1b — Best VAE reconstruction samples at convergence.</em></p>

<p align="center">
  <img src="assets/task1_original_vs_reconstructed.png" width="85%" alt="Figure 1c"/>
</p>
<p align="center"><em>Figure 1c — Original (left) vs. reconstructed (right), per channel: Tracks, ECAL, HCAL.</em></p>

**Observations:** The VAE achieves near-perfect reconstruction with zero false activations on background pixels. The nonzero-weighted loss ensures focus on the rare active cells rather than trivially predicting all-black. The 256-dim latent space provides the foundation for Task 3.

---

## Task 2 — Graph Neural Network Classification

**Objective:** Classify jets as **quark** or **gluon** using a graph representation that respects the sparse, irregular structure of detector data — rather than treating it as a dense image [[1]](#references).

**Pipeline:**
```
125×125×3 image → active pixel extraction → (η, φ) point cloud → k-NN graph (k=8) → GraphSAGE → quark/gluon
```

**Features:**

| Type | Dimension | Components |
|------|-----------|------------|
| Node | 6 | η_norm · φ_norm · E_tracks · E_ECAL · E_HCAL · r_centroid |
| Edge | 4 | Δη · Δφ · distance · ΔE |

This discards the 99.94% background and operates directly on the physics-relevant active cells, following the graph-based approach from [[1]](#references).

### Results

| Metric | MLP Baseline | GraphSAGE [[3]](#references) |
|--------|--------------|------------------------------|
| ROC-AUC | 0.564 | **0.774** |
| Accuracy | 54.3% | **70.6%** |
| F1 | 0.562 | **0.727** |

---

<p align="center">
  <img src="assets/task2_training_curves.png" width="70%" alt="Figure 2a"/>
</p>
<p align="center"><em>Figure 2a — Training and validation loss/accuracy over 30 epochs.</em></p>

<p align="center">
  <img src="assets/task2_roc_curve.png" width="60%" alt="Figure 2b"/>
</p>
<p align="center"><em>Figure 2b — ROC: GraphSAGE (AUC 0.774) vs. MLP baseline (AUC 0.564).</em></p>

<p align="center">
  <img src="assets/task2_pipeline.png" width="85%" alt="Figure 2c"/>
</p>
<p align="center"><em>Figure 2c — Pipeline: detector image → point cloud → k-NN graph.</em></p>

**Observations:** The graph representation provides a **37% relative AUC improvement** over the MLP baseline. Quark jets produce broader, higher-multiplicity particle sprays compared to narrower gluon jets — structure that graph topology captures naturally through node connectivity. The approach is also ~100× more memory-efficient.

---

## Task 3 — Latent Diffusion

**Objective:** Generate new, physically realistic jet images by training a DDPM denoiser [[2]](#references) in the 256-dim latent space from Task 1.

**Method:** Two-stage pipeline:
1. **Encode** — freeze the Task 1 VAE; pre-compute latent vectors for all training data
2. **Denoise** — train a time-conditioned residual MLP on the latent vectors using a standard DDPM noise schedule

| Component | Details |
|-----------|--------|
| VAE (frozen) | Task 1 `JetVAE`, 256-dim latent |
| Denoiser | 6 residual blocks · hidden 1024 · time_emb 128 |
| Schedule | 1000 DDPM timesteps · linear β |
| Training | 30 epochs · AdamW (lr = 1e-4) · cosine annealing |

**Key design decision:** The VAE uses a custom `detector_reference` preprocessing. Task 3 loads the exact preprocessing parameters from the VAE checkpoint — guaranteeing the encoder receives correctly normalized data and the decoder produces physically meaningful outputs.

### Results

| Metric | Score |
|--------|-------|
| PSNR | **30.32 dB** |
| SSIM | **0.931** |
| MSE | **0.0009** |
| Generated Active Density | **66.67%** |
| Real Active Density | **66.77%** |

---

<p align="center">
  <img src="assets/task3_training_curves.png" width="60%" alt="Figure 3a"/>
</p>
<p align="center"><em>Figure 3a — Latent MSE loss convergence over 30 epochs.</em></p>

<p align="center">
  <img src="assets/task3_generated_samples.png" width="85%" alt="Figure 3b"/>
</p>
<p align="center"><em>Figure 3b — Original (left) vs. generated (right) jets, per channel: Tracks, ECAL, HCAL.</em></p>

**Observations:** Generated jets match real sparsity (66.67% vs 66.77% active density) with high structural fidelity (SSIM 0.931). Working in latent space provides two advantages: (1) training takes minutes instead of hours, and (2) the VAE bottleneck enforces physically meaningful structure. The 7.5 dB gap between generation PSNR (30.3) and VAE reconstruction PSNR (37.9) represents the latent-space modelling error — the natural next step is replacing the MLP denoiser with more expressive architectures (e.g., Diffusion Transformers).

---

## Reproducibility

### Setup

```bash
git clone https://github.com/nitik1998/GENIE_DiffusionLearning.git
cd GENIE_DiffusionLearning
pip install -r requirements.txt

# Download dataset
mkdir -p data
wget -O data/quark-gluon_data-set_n139306.hdf5 \
  https://cernbox.cern.ch/remote.php/dav/public-files/b5WrtcHe0xQ26M4/quark-gluon_data-set_n139306.hdf5
```

### Training

```bash
# Task 1 — VAE (~15 min on H100)
python src/task1_autoencoder.py --epochs 200 --batch-size 512

# Task 2 — GNN (~5 min)
python src/task2_gnn.py --epochs 30 --exp-name task2_graph_classifier

# Task 3 — Latent Diffusion (~8 min on H100)
python src/task3_diffusion.py --mode latent_diffusion --epochs 30

# Quick smoke tests (CPU, < 1 min each)
python src/task1_autoencoder.py --max-events 1000 --epochs 3 --force-rerun
python src/task2_gnn.py --max-events 200 --epochs 2 --force-cpu
python src/task3_diffusion.py --max-events 64 --epochs 1 --timesteps 20 --force-cpu
```

### Colab Notebooks

| Task | Notebook | Run Mode |
|:-----|:---------|:---------|
| 1 | [`Task1_Autoencoder.ipynb`](notebooks/Task1_Autoencoder.ipynb) | `"sanity"` / `"full"` |
| 2 | [`Task2_Graph_Classifier.ipynb`](notebooks/Task2_Graph_Classifier.ipynb) | `"sanity"` / `"full"` |
| 3 | [`Task3_Diffusion_Exploration.ipynb`](notebooks/Task3_Diffusion_Exploration.ipynb) | `"sanity"` / `"full"` |

---

## Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, GPU detection, logging
│   ├── data_utils.py             # Data loading, preprocessing, splitting
│   ├── metrics.py                # PSNR, SSIM, active density metrics
│   ├── experiment_tracker.py     # Experiment logging and checkpointing
│   ├── eda.py                    # Exploratory data analysis
│   ├── task1_autoencoder.py      # VAE training pipeline
│   ├── task2_gnn.py              # GNN classification pipeline
│   ├── task3_diffusion.py        # Latent diffusion pipeline
│   ├── visualize_task2_pipeline.py
│   ├── data/
│   │   ├── dataset.py            # Dataset classes
│   │   ├── graph_builder.py      # Point cloud → graph conversion
│   │   └── transforms.py         # Data transforms
│   └── models/
│       ├── __init__.py
│       ├── autoencoder.py        # JetVAE, ConvAutoEncoder
│       ├── gnn.py                # GraphSAGE classifier
│       ├── diffusion_core.py     # DDPM noise scheduler
│       ├── diffusion_unet.py     # SimpleUNet (pixel-space baseline)
│       └── latent_denoiser.py    # Time-conditioned residual MLP
├── notebooks/                    # Colab-ready notebooks
├── results_from_colab/           # Final trained models & metrics
├── assets/                       # README figures
└── requirements.txt
```

### Pre-trained Models

| Archive | Contents |
|:--------|:---------|
| `results_from_colab/task1_autoencoder.zip` | VAE checkpoint · training curves · reconstructions |
| `results_from_colab/task2_graph_classifier.zip` | GraphSAGE checkpoint · ROC curves · confusion matrix |
| `results_from_colab/latent_diffusion.zip` | Latent denoiser checkpoint · generated samples |

---

## References

1. A. Baldini *et al.*, "Graph Generative Models for Fast Detector Simulations in High Energy Physics," *Machine Learning and the Physical Sciences (NeurIPS Workshop)*, 2021. [arXiv:2104.01725](https://arxiv.org/abs/2104.01725)

2. J. Ho, A. Jain, P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

3. W. Hamilton, R. Ying, J. Leskovec, "Inductive Representation Learning on Large Graphs," *NeurIPS*, 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

4. Z. Wang *et al.*, "Image Quality Assessment: From Error Visibility to Structural Similarity," *IEEE TIP*, 2004. [DOI:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

5. D.P. Kingma, M. Welling, "Auto-Encoding Variational Bayes," *ICLR*, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

---

*Developed as part of the [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) evaluation for [ML4SCI](https://ml4sci.org/).*
