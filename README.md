# VLA for Autonomous Driving (VLAD)

A research project exploring **Vision-Language-Action (VLA)** models for autonomous driving. VLAD combines visual perception with language understanding to predict driving actions and trajectories in simulation, using CARLA and the Bench2Drive dataset.

## Overview

VLAD integrates:
- **Vision-Language Models (VLMs)** — Qwen3-VL for multimodal reasoning over ego-view images
- **DriveFusion** — Transformer-based fusion of image embeddings with diffusion for trajectory prediction
- **Diffusion Policy** — Action prediction conditioned on visual and state inputs
- **Bench2Drive** — Large-scale driving dataset from CARLA (HuggingFace: `rethinklab/Bench2Drive`)

The model predicts future waypoints and actions from camera history, ego-state, and navigation commands, suitable for end-to-end autonomous driving in simulation.

## Project Structure

```
├── src/
│   ├── models/           # DriveFusion, diffusion policy, diffusion transformer
│   ├── dataloaders/      # Bench2Drive dataset loaders (single-frame & history)
│   ├── vlm/              # Qwen VLM wrappers, embedding cache
│   ├── driver/           # CARLA driver with VLM backbone
│   └── utils/            # Bench2Drive parsing, visualization
├── scripts/              # Setup, dataset download, testing
├── media/                # Example ego-view images for VLM testing
└── carla.sh              # CARLA server launcher (Docker)
```

Example ego-view image used for VLM queries:

![Ego-view example](media/ego-image-speed-limit.png)

## Quick Start

### 1. Environment Setup

**On Bridges-2:**
```bash
source scripts/setup_env.sh
# Then: conda activate ./conda/vlad
```

**Local:** Create a conda env with Python 3.8 and install `src/requirements.txt` (PyTorch 2.2, CARLA 0.9.15, etc.).

### 2. Run CARLA Simulation (CARLA 0.9.15)

Open two terminals:

**Terminal 1 — Start CARLA server:**
```bash
./carla.sh
```

**Terminal 2 — Run client:**
```bash
python3 src/CarlaClientTest.py
```

This spawns a vehicle in Town02 and runs autopilot for 60 seconds. CARLA listens on port 2000.

### 3. Download Dataset

**Bench2Drive Base** (full set):
```bash
scripts/download_bench2drive_base.sh
```

### 4. Test Dataloader

```bash
python3 scripts/test_dataloader.py
```

This loads the Bench2Drive dataset, prints statistics, and saves sample visualizations (images with overlaid waypoints) to `output/samples/`.

## Training

- **Diffusion Policy:** `python3 src/models/train_diffusion_policy.py`
- **DriveFusion:** `python3 src/models/train_drivefusion.py`

Both use Hydra for config and Weights & Biases for logging.

## Dependencies

Key dependencies: PyTorch 2.2, CARLA 0.9.15, PyTorch Lightning, HuggingFace Hub, Hydra, OpenCV, Pandas. See `src/requirements.txt` for full list.

---
*CMU Intro to Deep Learning 11785 Project — VLA for Autonomous Driving*
