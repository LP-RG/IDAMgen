# Training Workflow

All training is driven by `res_net_training.py`.
Stages are selected via `--conv_type` and must be run **in order** — each
stage depends on the checkpoint produced by the previous one.

## Stage Overview

```
conv_type 1 ──► exact (float) checkpoint
     │
     ▼
conv_type 2 ──► quantized checkpoint
     │
     ▼
conv_type 3 ──► approximate checkpoint  (requires a .npy multiplier table)
```

---

## Convolution Types

| `--conv_type` | Mode | Description |
|---|---|---|
| `1` | Exact | Standard float32 convolution. |
| `2` | Quantized | Quantized convolution, no approximation error. Fine-tunes for 5 epochs from the exact checkpoint. |
| `3` | Approximate (STE) | Quantized + approximate multiplier, straight-through estimator gradient. Retrains for 3 epochs. |
| `4` | Approximate (error-aware) | Like type 3 but with error-aware gradient. |
| `5` | Stats collection | Collects per-layer activation statistics (used for heat maps); no accuracy output. |

---

## Checkpoint Naming

All checkpoints are saved under `trained_models/`.

| Stage | Filename |
|---|---|
| Exact | `<model>.pth` |
| Quantized | `<model>_q<bits>.pth` |
| Approximate — no retrain | `<model>_a<bits>_<table>_noretrain.pth` |
| Approximate — retrained best | `<model>_a<bits>_<table>_retrained_best.pth` |

`<table>` is the stem of the `.npy` multiplier file (e.g. `my_table` for `my_table.npy`).

---

## CLI Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `resnet` | Model key (see README for full list). |
| `--conv_type` | int | `1` | Convolution / training stage (1–5). |
| `--bit_width` | int | `8` | Quantization bit width. |
| `--signed` | flag | `False` | Use signed quantization. |
| `--zone` | flag | `False` | Enable zone-based quantization. |
| `--input_path` | str | `None` | Path to a `.npy` multiplier table, or a directory of `.npy` files (batch mode). |
| `--exact_accuracy` | float | `0` | Exact baseline accuracy; retrain aborts if drop exceeds 3 pp. |
| `--no_retraining` | flag | `False` | Skip retrain loop for conv_type 3; saves `_noretrain.pth` and exits. |

---

## Examples

### Train and test exact model

```bash
python res_net_training.py --model_name lenet5 --conv_type 1
```

### Quantize

```bash
python res_net_training.py --model_name lenet5 --conv_type 2 --bit_width 8
```

### Approximate retrain with a single multiplier table

```bash
python res_net_training.py \
    --model_name lenet5 \
    --conv_type 3 \
    --input_path multipliers/my_table.npy \
    --bit_width 8 \
    --exact_accuracy 98.5
```

### Batch mode — evaluate a whole folder of tables

```bash
python res_net_training.py \
    --model_name resnet \
    --conv_type 3 \
    --input_path multipliers/ \
    --bit_width 8
```

> **Note:** Batch mode iterates over every `.npy` file in the folder and prints
> `FINAL_ACCURACY:<value>` for each. Results are also appended to `results.csv`.
