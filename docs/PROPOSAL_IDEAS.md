# GSoC 2026 Proposal Ideas: Learning the Latent Structure with Diffusion Models

This document tracks ideas, insights, and evidence from the evaluation tasks to feed into the GSoC proposal.

---

## What We Proved in the Evaluation

### Baseline Results

| Task | Model | Key Metric | Value |
|------|-------|------------|-------|
| Task 1 — Reconstruction | JetVAE (Conv VAE) | PSNR / SSIM | **37.93 dB / 0.967** |
| Task 2 — Classification | GraphSAGE (k-NN) | ROC-AUC | **0.774** |
| Task 3 — Generation | Latent Diffusion (DDPM) | PSNR / SSIM | **30.32 dB / 0.931** |

### Key Findings

1. **Preprocessing alignment is critical.** The VAE was trained with a custom `detector_reference` normalization (per-channel log/boost for Tracks, mean/std for ECAL/HCAL). Using a different normalization for downstream tasks produced saturated orange noise (PSNR 4.66 → 30.32 after fixing). *Proposal angle: propose a unified preprocessing registry that all pipeline stages share.*

2. **VAE is the generation ceiling.** The diffusion model converged by epoch ~10. Training for 30 vs 100 epochs gave identical results (30.32 → 30.37 PSNR). The 7.5 dB gap between generation (30.3) and reconstruction (37.9) is entirely from the latent-space error, not the denoiser. *Proposal angle: improving the VAE or using a hierarchical latent space is the highest-leverage path to better generation.*

3. **Sparsity dominates everything.** 99.94% of pixels are zero. The graph representation (Task 2) exploits this by only processing active cells — yielding a 37% AUC improvement over MLP baselines. *Proposal angle: graph/point-cloud representations are the natural substrate for both classification and generation of sparse detector data.*

4. **MLP denoisers hit a capacity wall.** The residual MLP denoiser (6 layers, 1024 hidden) converges fast but plateaus. It cannot model complex multi-modal latent distributions. *Proposal angle: transformer-based denoisers (DiT) or graph-conditional architectures will break through this ceiling.*

---

## Proposal Idea 1: Advanced Representation Learning

*Current baseline: Convolutional VAE predicting dense 125×125 images.*

### Point-Cloud JEPA (Joint-Embedding Predictive Architecture)
- Represent jets as 3D point clouds `(η, φ, E)` instead of dense images
- Mask random particles and predict their *latent embeddings* (not raw pixels)
- Aligns with I-JEPA / V-JEPA self-supervised paradigm
- **Evidence:** Task 2 showed graph representations outperform image representations for classification. The same should hold for generation.

### Equivariant Masked Autoencoders
- Train a Transformer-based MAE that respects rotational/translational symmetries of collider physics
- Physics-motivated augmentations: φ-rotation invariance, η-reflection symmetry

### Hierarchical VAE
- Current VAE uses a single 256-dim bottleneck
- Multi-scale latent hierarchy (e.g., VQ-VAE-2 style) could capture both global jet shape and fine-grained energy patterns
- Would directly close the 7.5 dB gap between VAE reconstruction and diffusion generation

---

## Proposal Idea 2: Next-Generation Generative Modeling

*Current baseline: Discrete-time DDPM (1000 steps) on latent vectors via residual MLP.*

### Flow Matching / Rectified Flows
- Replace DDPM with ODE-based flow matching (the math behind Stable Diffusion 3)
- Straighter generation trajectories → generate samples in 10–20 solver steps instead of 1000
- **90% inference speedup** with equal or better quality
- Direct drop-in replacement for the current DDPM scheduler

### Graph/Point-Cloud Diffusion
- Skip the 125×125 grid entirely
- Run diffusion directly over continuous particle coordinates and energies
- E(n)-Equivariant Diffusion Models preserve physical symmetries
- Natural for sparse data — no wasted compute on empty pixels
- **Evidence:** Our evaluation shows the dense image representation wastes 99.94% of pixels

### Transformer Denoiser (DiT)
- Replace the MLP denoiser with a Diffusion Transformer
- Multi-head self-attention over latent patches can model complex multi-modal distributions
- **Evidence:** Our MLP denoiser converged to a ceiling at 30.3 dB PSNR; attention mechanisms could break through this wall

---

## Proposal Idea 3: Physics-Aware Conditional Generation

*Current baseline: Unconditional generation.*

### Kinematic Cross-Attention
- Inject physics variables (target pT, jet mass, quark vs gluon label) into the denoiser via cross-attention
- Allows physicists to request specific types of simulated events
- **Evidence:** Task 2 showed quark/gluon jets have structurally different graph topologies (broader vs narrower). Conditional generation should leverage this.

### Energy Conservation Guidance
- Classifier-free guidance or gradient-based conditioning during reverse diffusion
- Enforce that generated components obey conservation laws (energy, momentum)
- Physics constraint as an inductive bias → more realistic samples with fewer training iterations

### Class-Conditional Latent Diffusion
- Simplest extension: train separate denoiser heads (or condition with class embeddings) for quark vs gluon
- Use Task 2's GraphSAGE as a discriminator to evaluate generation quality per class
- **Evidence:** ROC-AUC 0.774 means quark/gluon jets are distinguishable — generated samples should respect this distinction

---

## Proposal Idea 4: Evaluation & Validation Framework

*Current baseline: PSNR, SSIM, active density.*

### Physics-Motivated Metrics
- **Jet mass distribution:** compare histograms of generated vs real jet masses
- **Particle multiplicity:** distribution of active pixel counts per event
- **Energy flow polynomials:** rotationally-invariant jet substructure observables
- **Fréchet Physics Distance:** analogous to FID but computed over physics-relevant features

### Ablation Studies
- Latent dim: 64 vs 128 vs 256 vs 512
- Denoiser architecture: MLP vs Transformer vs GNN
- Noise schedule: linear vs cosine vs learned
- Preprocessing: how much does the choice of normalization affect downstream generation?

---

## Timeline Sketch (12 weeks)

| Week | Milestone |
|------|-----------|
| 1–2 | Set up infrastructure: data pipeline, evaluation suite, experiment tracking |
| 3–4 | Implement flow matching / rectified flows as DDPM replacement |
| 5–6 | Graph/point-cloud representation for generation pipeline |
| 7–8 | Conditional generation (class-conditioned, kinematic cross-attention) |
| 9–10 | Physics evaluation metrics, ablation studies |
| 11–12 | Documentation, paper draft, final benchmarks |

---

*The evaluation code demonstrates end-to-end competence with VAE → GNN → latent diffusion. These proposal ideas show how to elevate the project to research-grade during the GSoC coding period.*
