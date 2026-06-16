# CNN_At — Approximate Computing CNN Framework

A research framework for training, quantizing, and evaluating CNN models on
approximate-hardware multipliers, with built-in t-SNE visualization of
misclassification patterns across exact, quantized, and approximate stages.

## Quick Start

```bash
# 1. Train exact (float) model
python3 res_net_training.py --model_name lenet5 --conv_type 1

# 2. Quantize
python3 res_net_training.py --model_name lenet5 --conv_type 2 --bit_width 8

# 3. Approximate retrain
python3 res_net_training.py --model_name lenet5 --conv_type 3 \
    --input_path multipliers/my_table.npy --bit_width 8

# 4. Visualise with t-SNE
python3 res_net_training.py --model_name lenet5 --tsne
```

## Supported Models

| Key | Architecture | Dataset |
|---|---|---|
| `lenet5` | LeNet-5 | MNIST / CIFAR-10 |
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


## Project Layout

```
CNN_At/
├── res_net_training.py     ← main CLI: training + t-SNE entry point
├── orchestrator.py         ← batch pipeline (SubXPAT → analyzer → training)
├── apps/
│   └── tsne_dash_app.py    ← interactive Dash browser app
├── models/                 ← model definitions
├── modules/
│   ├── convolution.py      ← Conv2d_custom (exact / quantized / approximate)
│   ├── tsne_visualization.py
│   └── tsne_utils.py
├── trained_models/         ← saved checkpoints
└── plots/                  ← t-SNE PNGs and Dash .npz artifacts
```
