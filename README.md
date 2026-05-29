<p align="center">
  <h1 align="center">
    LeHome
  </h1>
  <h2 align="center">
    A Simulation Environment for Deformable Object Manipulation in Household Scenarios
  </h2>
</p>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.1-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![LeRobot](https://img.shields.io/badge/LeRobot-0.4.3-yellow.svg)](https://github.com/huggingface/lerobot)

</div>

LeHome provides a high-fidelity simulation platform by integrating various household scenarios and various objects within the scenarios, especially deformable objects.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Quick Start](#quick-start)
  - [1. Installation](#1-installation)
    - [Use UV](#use-uv)
  - [2. Assets \& Data Preparation](#2-assets--data-preparation)
    - [Download Simulation Assets](#download-simulation-assets)
    - [Collect Your Own Data](#collect-your-own-data)
  - [3. Object and Scene Configuration](#3-object-and-scene-configuration)
    - [Object and Scene Configuration Guide](#object-and-scene-configuration-guide)
    - [Scene Deactivation Guide](#scene-deactivation-guide)
  - [4. Train](#4-train)
    - [Quick Start](#quick-start-1)
  - [5. Eval](#5-eval)
    - [Common Options](#common-options)
- [Documentation Index](#documentation-index)
- [Acknowledgments](#acknowledgments)

## Quick Start

> **IMPORTANT**:
> For Ubuntu version and GPU-related settings, please refer to the [Isaac Sim 5.1.0 documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

### 1. Installation

#### Use UV

The simulation environment is based on the Isaac Lab and LeRobot repositories; please refer to [UV installation guide](docs/installation.md).

### 2. Assets & Data Preparation

#### Download Simulation Assets

Download the required simulation assets (Material, scenes, objects, robots) from HuggingFace:

```bash
# This creates the Assets/ directory with the required simulation resources
hf download lehome/lehome_release --repo-type dataset --local-dir Assets
```

#### Collect Your Own Data

For detailed instructions on teleoperation data collection and dataset processing, please refer to our [Dataset Collection and Processing Guide](docs/datasets.md) (`SO101 Leader` is strongly recommended).

### 3. Object and Scene Configuration

#### Object and Scene Configuration Guide

For recommended object import patterns, task-local asset configuration, and the distinction between task objects and shared scene assets, see [Object and Scene Configuration Guide](docs/object_scene_configuration.md).

#### Scene Deactivation Guide

If your task depends on assets under `/World/Scene`, review the whitelist-based deactivation mechanism before modifying scene content or keep-path prefixes. For details, see [Scene Deactivation Guide](docs/scene_deactivation.md).

### 4. Train

LeHome provides several training examples; the models and training framework are from LeRobot.

#### Quick Start

Train using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_<policy>.yaml
```

**Available config files:**

- `configs/train_act.yaml` - ACT
- `configs/train_dp.yaml` - Diffusion Policy
- `configs/train_smolvla.yaml` - SmolVLA

**Key configuration options:**

- Update `dataset.root` to point to your dataset.
- Adjust `policy.input_features` and `policy.output_features` to match your dataset schema.
- Modify `batch_size`, `steps`, `save_freq`, and `log_freq` based on your training budget.
- Change `output_dir` if you want checkpoints stored elsewhere.

For detailed training instructions, feature selection guide, and configuration options, see our [Training Guide](docs/training.md).

### 5. Eval

Evaluate your trained LeRobot policy on LeHome tasks.

**Examples:**

```bash
# Note: --policy_path and --dataset_root are required, ready to run once the dataset and model checkpoints are prepared.
python -m scripts.eval \
    --task <task_name> \
    --policy_type lerobot \
    --policy_path outputs/train/<output_name>/checkpoints/last/pretrained_model \
    --dataset_root Datasets/<dataset_name> \
    --task_description "<task_description>" \
    --num_episodes 5 \
    --enable_cameras \
    --device <sim_device>
```

Use `cpu`, `cuda`, or `cuda:N` for `<sim_device>` according to the task and runtime environment.

#### Common Options

| Parameter | Description | Default | Required For |
|-----------|-------------|---------|--------------|
| `--task` | Task ID registered in this repository | - | All |
| `--policy_type` | Policy type, use `lerobot` | `lerobot` | All |
| `--policy_path` | Path to LeRobot model checkpoint | - | All |
| `--dataset_root` | Dataset path (for metadata) | - | LeRobot only |
| `--num_episodes` | Number of evaluation episodes | `5` | All |
| `--max_steps` | Max steps per episode | `600` | All |
| `--save_video` | Save evaluation videos | disabled | Optional |
| `--video_dir` | Directory to save evaluation videos | `outputs/eval_videos` | `--save_video` |
| `--enable_cameras` | Enable camera rendering | disabled | Recommended |
| `--device` | Simulator device: `cpu`, `cuda`, or `cuda:N` | `cuda:0` | Optional |
| `--headless` | Run without GUI | disabled | Optional |

For detailed evaluation usage and troubleshooting, see [Policy Evaluation Guide](docs/policy_eval.md).

## Documentation Index

- [UV installation guide](docs/installation.md)
  - Step-by-step environment setup with `uv`, Isaac Lab, editable package install, and optional server dependencies.
- [Dataset Collection and Processing Guide](docs/datasets.md)
  - SO101 leader setup, teleoperation recording, replay, dataset inspection, augmentation, pointcloud conversion, merge, and dataset schema notes.
- [Object and Scene Configuration Guide](docs/object_scene_configuration.md)
  - Recommended patterns for adding task objects and scene assets, including `DeformableObjectCfg` / `RigidObjectCfg`, `GarmentObject` / `FluidObject`, and whitelist-based activation for fixed scene assets.
- [Scene Deactivation Guide](docs/scene_deactivation.md)
  - Overview of subtree deactivation, configuration fields, task defaults, and runtime logs.
- [Training Guide](docs/training.md)
  - Provided training configs, dataset feature layout, feature selection advice, and repository-verified YAML configuration keys.
- [Policy Evaluation Guide](docs/policy_eval.md)
  - LeRobot policy evaluation, common options, and troubleshooting.

## Acknowledgments

This project stands on the shoulders of giants. We utilize and build upon the following excellent open-source projects:

- **[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)** - For photorealistic physics simulation
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html)** - For modular robot learning environments
- **[LeRobot](https://github.com/huggingface/lerobot)** - For dataset tooling and policy learning baselines

## 📚 Citation

If you use LeHome in your research, please consider citing:

```bibtex
@misc{li2026lehomesimulationenvironmentdeformable,
      title={LeHome: A Simulation Environment for Deformable Object Manipulation in Household Scenarios}, 
      author={Zeyi Li and Yushi Yang and Shawn Xie and Kyle Xu and Tianxing Chen and Yuran Wang and Zhenhao Shen and Yan Shen and Yue Chen and Wenjun Li and Yukun Zheng and Chaorui Zhang and Siyi Lin and Fei Teng and Hongjun Yang and Ming Chen and Steve Xie and Ruihai Wu},
      year={2026},
      eprint={2604.22363},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.22363}, 
}
