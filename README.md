# ML4SCI GENIE GSoC 2026

This repository contains the local development, evaluation, and mentor-facing Colab workflows for the ML4SCI GENIE GSoC 2026 project on latent-structure learning for sparse detector data.

The repo is organized around three tasks:

- Task 1: a stable image-based reconstruction baseline
- Task 2: a graph-first jet-classification pipeline for sparse detector hits
- Task 3: an honest exploratory diffusion baseline, framed by what we learned about sparsity

## Dataset

The quark/gluon jet dataset contains 139,306 detector images with shape `(125, 125, 3)`. In the active repo convention, the detector channels are:

- channel 0: `Tracks`
- channel 1: `ECAL`
- channel 2: `HCAL`

These are not RGB channels. They are correlated detector views of the same physical event. Channel-wise plots are therefore the primary evidence throughout the repo.

```bash
mkdir -p data
wget -O data/quark-gluon_data-set_n139306.hdf5 \
  https://cernbox.cern.ch/remote.php/dav/public-files/b5WrtcHe0xQ26M4/quark-gluon_data-set_n139306.hdf5
```

## Project Positioning

### Task 1: Image-Based Baseline

- Model: convolutional VAE with transpose decoder
- Goal: provide a stable, reproducible image baseline for sparse detector reconstruction
- Positioning: useful baseline, but limited by pixel-space sparsity
- Entry points:
  - script: `src/task1_autoencoder.py`
  - notebook: `notebooks/Task1_Autoencoder.ipynb`
  - outputs: `outputs/ae/task1_image_baseline/`

### Task 2: Graph-First Sparse Detector Modeling

- Model: tuned GraphSAGE on deterministic k-NN graphs in `(eta, phi)` space
- Graph representation:
  - node = active detector pixel
  - node features = `[eta_norm, phi_norm, E_Tracks, E_ECAL, E_HCAL, r_centroid]`
  - edge features = `[delta_eta, delta_phi, distance, delta_intensity]`
- Positioning: strongest technical section of the repo and closest to the mentor-aligned sparse-detector view
- Entry points:
  - script: `src/task2_gnn.py`
  - notebook: `notebooks/Task2_Graph_Classifier.ipynb`
  - outputs: `outputs/task2/<exp_name>/`

### Task 3: Exploratory Diffusion

- Model: pixel-space DDPM with `SimpleUNet`
- Positioning: exploratory baseline, intentionally not oversold
- Main lesson: sparse detector structure is not naturally dense-image-like; latent-aware or graph-aware diffusion is the more principled next step
- Entry points:
  - script: `src/task3_diffusion.py`
  - notebook: `notebooks/Task3_Diffusion_Exploration.ipynb`
  - outputs: `outputs/task3/<exp_name>/`

## Running Locally

```bash
pip install -r requirements.txt
```

### Task 1

```bash
# smoke test
python src/task1_autoencoder.py --exp-name task1_image_baseline --max-events 10000 --epochs 10 --batch-size 16 --force-rerun

# full run
python src/task1_autoencoder.py --exp-name task1_image_baseline --epochs 200 --batch-size 16 --force-rerun
```

### Task 2

```bash
# smoke test
python src/task2_gnn.py --max-events 200 --epochs 2 --force-cpu --exp-name smoke_test_notebook

# full run
python src/task2_gnn.py --epochs 30 --exp-name task2_graph_classifier
```

### Task 3

```bash
# lightweight smoke test
python src/task3_diffusion.py --max-events 64 --epochs 1 --timesteps 20 --n-samples 2 --force-cpu --exp-name smoke_test_notebook_fast

# exploratory full run
python src/task3_diffusion.py --epochs 30 --exp-name task3_final_diffusion
```

## Running On Colab

There are exactly three active mentor-facing notebooks:

- `notebooks/Task1_Autoencoder.ipynb`
- `notebooks/Task2_Graph_Classifier.ipynb`
- `notebooks/Task3_Diffusion_Exploration.ipynb`

Each notebook is a thin wrapper around the corresponding script. The notebooks only handle:

- Colab environment setup
- optional Google Drive mount
- dataset/output path configuration
- running the existing script
- showing saved outputs inline
- zipping artifacts for download

For Task 1 and Task 2, each notebook exposes a simple run-mode configuration near the top:

- `RUN_MODE = "sanity"` for a short check run
- `RUN_MODE = "full"` for the final submission run

The dataset should live at either:

- `/content/drive/MyDrive/quark-gluon_data-set_n139306.hdf5`
- or `/content/quark-gluon_data-set_n139306.hdf5`

## Output Layout

```text
outputs/
├── ae/task1_image_baseline/
├── task2/<exp_name>/
├── task3/<exp_name>/
├── eda/
└── experiments_log.*
```

Important artifact locations:

- Task 1 final baseline: `outputs/ae/task1_image_baseline/`
- Task 2 notebook smoke outputs: `outputs/task2/smoke_test_notebook/`
- Task 3 notebook smoke outputs: `outputs/task3/smoke_test_notebook_fast/`

## Key Docs

- `docs/task1_report.md`
- `docs/task2_report.md`
- `docs/task3_report.md`
- `docs/final_project_summary.md`
- `context.md`

## Repository Structure

```text
src/
├── config.py
├── data_utils.py
├── metrics.py
├── task1_autoencoder.py
├── task2_gnn.py
├── task3_diffusion.py
├── visualize_task2_pipeline.py
└── models/
```

The public repo is kept intentionally clean around the current active workflows and final notebooks.
