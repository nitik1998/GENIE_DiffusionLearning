# Task 3 Context (For Claude)

## Objective
Implement TWO experiments for Task 3 efficiently:

1. Image DDPM baseline (expected to underperform)
2. Latent diffusion main model (expected to perform better)

## Key Idea
- Sparse detector data ≠ natural images
- Pixel diffusion struggles
- Latent diffusion is more stable and aligned with “latent structure”

## Experiments

### A. Image DDPM
- Simple U-Net
- Small timesteps
- Goal: show limitation

### B. Latent Diffusion
Pipeline:
x → encoder → z → diffusion → decoder → x_hat

- Use Task 1 encoder/decoder if possible
- Keep model simple (MLP ok)

## Outputs (both experiments)
- original_vs_reconstructed.png
- abs_error.png
- samples_channelwise.png
- training_curves.png
- metrics.json

## Metrics
- nonzero MSE (primary)
- SSIM (optional)

## Visualization
- MUST be channel-wise
- No summed plots as main evidence

## Constraints
- Existing repo only
- No heavy redesign
- Notebook = thin wrapper
- Optimize for H100

## Goal
Produce:
- one weak baseline
- one strong main result
- clear visual comparison
