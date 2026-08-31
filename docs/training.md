# Training Workflow

All training is driven by `src/cnn_training.py`.
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

## Checkpoint Naming

All checkpoints are saved under `trained_models/`.

| Stage | Filename |
|---|---|
| Exact | `<model>.pth` |
| Quantized | `<model>_q<bits>.pth` |
| Approximate — no retrain | `<model>_a<bits>_<table>_noretrain.pth` |
| Approximate — retrained best | `<model>_a<bits>_<table>_retrained_best.pth` |

`<table>` is the stem of the `.npy` multiplier file (e.g. `my_table` for `my_table.npy`).

Per-epoch checkpoints follow `<model>[_q<bits>|_a<bits>_<table>]_epoch<N>.pth`
and are what the training-history pipeline consumes (see §6).

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
| `--seed` | int | `42` | RNG seed for training. |
| `--log_file` | str | `None` | Tee `stdout` to this file as well as the terminal. Required if you intend to run the log parser (§6). Opened with `"w"`, so each run overwrites its log. |

---

## Examples

### Train and test exact model

```bash
python3 src/cnn_training.py --model_name lenet5 --conv_type 1
```

### Quantize

```bash
python3 src/cnn_training.py --model_name lenet5 --conv_type 2 --bit_width 8
```

### Approximate retrain with a single multiplier table

```bash
python3 src/cnn_training.py \
    --model_name lenet5 \
    --conv_type 3 \
    --input_path multipliers/my_table.npy \
    --bit_width 8 \
    --exact_accuracy 98.5
```

### Batch mode — evaluate a whole folder of tables

```bash
python3 src/cnn_training.py \
    --model_name resnet \
    --conv_type 3 \
    --input_path multipliers/ \
    --bit_width 8
```

> **Note:** Batch mode iterates over every `.npy` file in the folder and prints
> `FINAL_ACCURACY:<value>` for each. Results are also appended to `results.csv`.

---

## 6 — Training History Pipeline (`tools/train_visualization/`)

Turns a stage's per-epoch checkpoints into per-epoch 3D embeddings, and its raw
training log into structured JSON, for the `/train-history` page of the
[Dash App](dash_app.md). The two halves are independent — neither calls the
other; they simply write into sibling stores that the dashboard reads.

> **Before the first run:** create the log directory, as `--log_file` does not
> create it for you.
> ```bash
> mkdir -p tools/train_visualization/raw_logs
> ```
> `epoch_features/`, `epoch_artifacts/` and `dashboard_data/` are created
> automatically.

### Branch 1 — Feature extraction (`train_epoch_extractor.py`)

Loads each epoch's checkpoint in order, extracts `penultimate` activations via
the existing forward hook, and computes a **3D** t-SNE embedding per epoch.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `lenet5` | Model key. |
| `--conv_type` | int | `1` | Stage to extract: 1 exact, 2 quantized, 3 approximate. |
| `--bit_width` | int | `8` | Bit width, used to resolve checkpoint filenames. |
| `--signed` | flag | `False` | Signed quantization (must match training). |
| `--zone` | flag | `False` | Zone-based quantization (must match training). |
| `--multiplier_matrix` | str | `None` | `.npy` lookup table; **required** for `--conv_type 3`. Note this is the same file `cnn_training.py` takes as `--input_path`. |

Embeddings are **warm started**: every epoch after the first is initialised
with the previous epoch's coordinates (rescaled) instead of a fresh PCA. t-SNE's
cost depends only on pairwise distances and so has no canonical orientation —
without warm starting, consecutive epochs land in arbitrarily different
rotations and the slider animation shows the embedding reorienting rather than
the model learning. The cost is that epochs are no longer independent and must
be computed strictly in order.

Outputs:

| Path | Contents |
|---|---|
| `epoch_features/<stage>_epoch<N>_features.npz` | Raw activations |
| `epoch_artifacts/<stage>_epoch<N>_artifact.npz` | Embedded 3D coordinates |
| `dashboard_data/train_manifest.json` | Combined stage/epoch → path lookup |

> **The manifest is rebuilt by rescan, not by merge.** `save_train_manifest`
> runs at the end of *every* extraction and re-scans the whole artifact folder.
> This is safe only because each filename bakes in its stage tag —
> `exact_epoch1_artifact.npz`, `quant_q8_epoch1_artifact.npz` and
> `approx_a8_my_table_epoch1_artifact.npz` cannot collide — so re-running the
> approximate stage picks up its new files while leaving the others intact,
> provided their `.npz` files are still on disk.

### Branch 2 — Log parsing (`train_log_parser.py`)

Reads a raw training log and emits structured JSON: a metadata header, a
per-epoch record (`epoch`, `train_loss`, `train_acc`, `test_acc`) and the final
accuracy. No checkpoints involved.

```bash
python3 tools/train_visualization/train_log_parser.py <log_path> [--output <path.json>]
```

| Argument | Description |
|---|---|
| `log_path` | Positional — the `.log` file written by `--log_file`. |
| `--output` | Destination JSON. Defaults to `parsed_logs/<log stem>.json`. **The parent directory must already exist when `--output` is given.** |

The dashboard expects one JSON per stage in `dashboard_data/`, named after the
stage tag: `exact.json`, `quant_q8.json`, `approx_a8_my_table.json`.

### Full command sequence

Run once per stage: train → extract → parse.

```bash
# 1. Training (conv_type 1 = exact, 2 = quantized, 3 = approximate)
python3 src/cnn_training.py --model_name lenet5 --conv_type 1 \
    --log_file tools/train_visualization/raw_logs/run_exact.log
python3 src/cnn_training.py --model_name lenet5 --conv_type 2 --bit_width 8 \
    --log_file tools/train_visualization/raw_logs/run_quant.log
python3 src/cnn_training.py --model_name lenet5 --conv_type 3 --bit_width 8 \
    --input_path multipliers/my_table.npy --exact_accuracy 99.13 \
    --log_file tools/train_visualization/raw_logs/run_approx.log

# 2. Extraction (only the approximate run needs the multiplier table)
python3 tools/train_visualization/train_epoch_extractor.py \
    --model_name lenet5 --conv_type 1 --bit_width 8
python3 tools/train_visualization/train_epoch_extractor.py \
    --model_name lenet5 --conv_type 2 --bit_width 8
python3 tools/train_visualization/train_epoch_extractor.py \
    --model_name lenet5 --conv_type 3 --bit_width 8 \
    --multiplier_matrix multipliers/my_table.npy

# 3. Parsing (identical shape for all three; only the filenames change)
python3 tools/train_visualization/train_log_parser.py \
    tools/train_visualization/raw_logs/run_exact.log \
    --output tools/train_visualization/dashboard_data/exact.json
```

Then launch the Dash app and open **Train History**, pasting
`tools/train_visualization/dashboard_data` as the run directory.
