# GSoC 2024 Proposal Ideas: Learning the Latent Structure with Diffusion Models

This document tracks advanced, cutting-edge ideas to include in the GSoC proposal to demonstrate a deep understanding of modern generative AI and High Energy Physics (HEP) challenges.

---

## 1. Advanced Representation Learning (The "Latent Structure")
*Current Baseline: Variational Autoencoder (VAE) predicting sparse pixels.*

**Proposal Ideas:**
*   **Point-Cloud JEPA (Joint-Embedding Predictive Architecture):** Jet images are highly sparse (99% empty space). Instead of reconstructing absolute zero-energy background pixels, represent the jet as a 3D Point Cloud (X, Y, Energy) or a Graph. Mask out random particles and train the network to predict the *latent representation* (the embedding) of the missing particles. This aligns with state-of-the-art self-supervised learning for physics.
*   **Equivariant Masked Autoencoders (MAE):** Train a Transformer-based MAE that respects rotational and translational symmetries inherent to high-energy collider physics.

## 2. Next-Generation Generative Modeling (The "Diffusion Models")
*Current Baseline: Discrete-Time DDPM (1000 steps) on Latent Vectors via an MLP.*

**Proposal Ideas:**
*   **Flow Matching / Rectified Flows:** Shift from standard DDPM to ODE-based Flow Matching (the mathematics behind Stable Diffusion 3). It provides straighter generation trajectories, meaning perfect physics samples can be generated in 10–20 solver steps instead of 1,000 steps, cutting inference time by 90%.
*   **Graph/Point-Cloud Diffusion Models:** Skip the 125x125 grid entirely and run the diffusion process directly over the continuous coordinates and energies of the constituent particles (e.g., E(n)-Equivariant Diffusion Models).

## 3. Physics-Aware Conditional Generation
*Current Baseline: Unconditional generation of arbitrary jets.*

**Proposal Ideas:**
*   **Kinematic Cross-Attention:** Inject specific physics variables (e.g., Target overall Transverse Momentum $p_T$, Jet Mass, or exact Jet Flavor: Quark vs Gluon) into the diffusion network using Cross-Attention layers. This allows physicists to request highly specific simulated events.
*   **Energy Conservation Guidance:** Implement classifier-free guidance or gradient-based conditioning during the reverse diffusion sampling to strictly enforce that the generated components obey physical laws of energy and momentum conservation.

---
*Note: The current submission code perfectly proves that you know how to build a working, end-to-end Latent Diffusion pipeline. These proposal ideas show the mentors how you plan to elevate the project to a globally competitive research level during the GSoC coding period.*
