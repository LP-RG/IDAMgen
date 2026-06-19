# CNN_At — Approximate Computing CNN Framework

A research framework for training, quantizing, and evaluating CNN models on
approximate-hardware multipliers, with built-in t-SNE visualization of
misclassification patterns across exact, quantized, and approximate stages,
and an orchestration pipeline that closes the loop between SubXPAT circuit
generation and CNN accuracy evaluation.

## Setup

```bash
make setup     # creates .venv, installs requirements.txt, torch (cu121 wheel),
               # and builds the custom mat_mul CUDA extension
```

```bash
. .venv/bin/activate   # activate the venv
deactivate              # exit the venv
make clean              # remove __pycache__, *.tmp files, and the .venv
```

Prerequisites not handled by `make setup`:

| Requirement | Needed for |
|---|---|
| CUDA toolkit matching your driver | building `src/extension_cpp` (`mat_mul`) |
| Yosys + OpenSTA on `PATH` (or set the path in `tools/synthesis_npy_generation/circuits_analizer.py`) | circuit synthesis via `vpadanalyzer` |
| [SubXPAT](https://github.com/) checked out at `subxpat/` (repo root) | `tools/orchestrator/orchestrator.py` only |

## Quick Start

All commands are run from the repo root.

```bash
# 1. Train exact (float) model
python3 src/cnn_training.py --model_name lenet5 --conv_type 1

# 2. Quantize
python3 src/cnn_training.py --model_name lenet5 --conv_type 2 --bit_width 8

# 3. Approximate retrain
python3 src/cnn_training.py --model_name lenet5 --conv_type 3 \
    --input_path multipliers/my_table.npy --bit_width 8

# 4. Visualise with t-SNE
python3 tools/tsne_visualization/tsne_runner.py --model_name lenet5
```

## Supported Models

| Key | Architecture | Dataset |
|---|---|---|
| `lenet5` | LeNet-5 | MNIST |
| `resnet` | ResNet-20 | CIFAR-10 |
| `resnet8` | ResNet-8 | CIFAR-10 |
| `resnet56` | ResNet-56 | CIFAR-100 |
| `vgg16` | VGG-16 | CIFAR-10 |
| `alexnet_cifar10` | AlexNet (adapted) | CIFAR-10 |

## Documentation

| Document | Description |
|---|---|
| [Training](docs/training.md) | Conv-type pipeline, CLI flags, checkpoint naming |
| [t-SNE Visualization](docs/tsne.md) | Feature embedding, misclassification overlay, CLI & API |
| [Dash App](docs/dash_app.md) | Interactive t-SNE explorer |
| [Hardware Evaluation Pipeline](docs/orchestrator.md) | SubXPAT → synthesis → CNN accuracy orchestration |

## Project Layout

```
CNN_At/
├── Makefile                        ← venv + dependency setup (make setup / make clean)
├── requirements.txt
├── src/
│   ├── cnn_training.py             ← main CLI: training & evaluation (conv_type 1–5)
│   ├── extension_cpp/              ← custom CUDA mat_mul extension (error-aware backprop)
│   ├── models/                     ← lenet5, resnet8/20/56, vgg16, alexnet_cifar10
│   └── modules/
│       ├── convolution.py          ← Conv2d_custom (exact / quantized / approximate)
│       ├── quantization.py, observers.py, functions.py
│       ├── data_loaders.py         ← MNIST / CIFAR-10 / CIFAR-100
│       └── common.py               ← model registry, paths, seeding, GPU cleanup
├── tools/
│   ├── tsne_visualization/
│   │   ├── tsne_runner.py          ← CLI entry point for t-SNE analysis
│   │   ├── tsne_visualization.py, tsne_utils.py
│   │   └── web_visualization/
│   │       └── tsne_dash_app.py    ← interactive Dash viewer
│   ├── orchestrator/
│   │   ├── orchestrator.py         ← SubXPAT → synthesis → training pipeline
│   │   └── npy_generator.py        ← per-circuit synth + simulation + CSV logging
│   └── synthesis_npy_generation/   ← circuit synthesis & error-metric building blocks
│       ├── circuits_analizer.py, sub_xpat_circuits_generator.py
│       └── sub_x_pat_simulator.py, multiplier_outputs_plotting.py
├── heat_maps/
│   ├── heat_map_plotting.py        ← plots per-layer activation co-occurrence maps
│   └── npy_matrix/                 ← raw activation histograms (conv_type 5 output)
├── syn_lib/
│   └── nangate_45nm_typ.lib        ← standard-cell lib used by the OpenSTA synthesis backend
├── subxpat/                        ← external SubXPAT checkout (not included, see Setup)
├── trained_models/                 ← saved checkpoints (generated)
└── docs/
    ├── training.md
    ├── tsne.md
    ├── dash_app.md
    └── orchestrator.md
```
