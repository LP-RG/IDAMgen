# t-SNE Visualization

The t-SNE tool runs [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
on CNN feature activations (or raw pixels), overlays classification errors, and
produces both static PNG outputs and artifacts for an interactive Dash web
application. Embeddings can be computed in 2D, 3D, or both.

---

## 1 — CLI Usage (`tools/tsne_visualization/tsne_runner.py`)

The t-SNE tool is a standalone entry point (it is **not** a flag on
`src/cnn_training.py`).

### Minimal example

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name lenet5
```

This runs t-SNE on the **exact** (float) checkpoint with all defaults (see table below).

### Full argument reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model_name` | str | `resnet` | Model key: `lenet5`, `resnet` (=resnet20), `resnet8`, `vgg16`, `alexnet_cifar10`, `resnet56`. |
| `--tsne_perplexity` | int | `30` | t-SNE perplexity. Lower values focus on local structure. |
| `--tsne_max_iter` | int | `1000` | Maximum t-SNE iterations. |
| `--tsne_max_train` | int | `2000` | Max training samples fed into t-SNE. |
| `--tsne_max_test` | int | `1000` | Max test samples fed into t-SNE. |
| `--tsne_seed` | int | `42` | RNG seed for subsampling and t-SNE init. |
| `--tsne_classes` | int… | `None` (all) | Restrict to a subset of classes, e.g. `--tsne_classes 0 1 2`. |
| `--tsne_feature_space` | `layer`/`pixels` | `layer` | Feature type to embed (see §2). |
| `--tsne_feature_layer` | str… | `penultimate` | Layer alias(es) or explicit module path(s) (see §2). You can specify multiple layers to process them all simultaneously. |
| `--tsne_stages` | str… | `exact` | Which checkpoints to visualise: `exact`, `quantized`, `approximate`. |
| `--tsne_multiplier_path` | str… | `None` | Path(s) to `.npy` lookup table(s); **required** for `approximate` stage. Accepts multiple paths to run several approximate models simultaneously. |
| `--tsne_components` | `2D`/`3D`/`2D+3D` | `2D` | Which embedding dimensionalities to compute (see §3). |
| `--bit_width` | int | `8` | Bit width for quantized/approximate checkpoint filenames. |
| `--show_misclassifications` | flag | `False` | Also save a grid of misclassified test images as a PNG. |
| `--tsne-no-save-static` | flag | — | Suppress static PNG output. |
| `--tsne-no-save-dash-artifact` | flag | — | Suppress `.npz` Dash artifact output. |
| `--train-if-missing` | flag | `False` | Automatically train missing checkpoints before running t-SNE. |

Sweep-only flags are documented separately in §5.

### Multi-stage comparison example

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name resnet \
    --tsne_stages exact quantized approximate \
    --tsne_multiplier_path multipliers/my_table.npy multipliers/my_other_table.npy \
    --tsne_feature_layer penultimate \
    --tsne_max_train 3000 --tsne_max_test 1500 \
    --tsne_seed 7
```

When multiple stages are specified, the **same** random train/test subsets are
reused across all stages, making the embeddings directly comparable.

---

## 2 — Feature Spaces and Layer Aliases

The `--tsne_feature_space` parameter controls what gets embedded:

| `feature_space` | What is embedded |
|---|---|
| `"layer"` | Activations extracted via a **forward hook** on the named module specified by `--tsne_feature_layer`. |
| `"pixels"` | Flattened raw input tensors (no model inference for training data). |

### Layer aliases (for `--tsne_feature_layer`)

You can pass one or more of these aliases separated by spaces. All requested layers will be processed and saved into the same run directory.

| Alias | Resolves to |
|---|---|
| `penultimate` | Second-to-last `nn.Linear`; or last `AdaptiveAvgPool2d` for ResNet-style models. |
| `logits` | Last `nn.Linear` (raw class scores before softmax). |
| `conv1` | First `Conv2d_custom` in the model. |
| `conv2` | Second `Conv2d_custom` (or first if only one exists). |
| `block1` | First of `layer1` / `block1` / `pool1` found in `named_modules()`. |
| `block2` | First of `layer2` / `block2` / `pool2` found in `named_modules()`. |
| *any other string* | Used **verbatim** as a `named_modules()` path, e.g. `layer2.0` or `features.3`. |

> If the literal string is already a valid `named_modules()` key, it is used
> directly without alias lookup.

---

## 3 — Dimensionality (`--tsne_components`)

| Value | Effect |
|---|---|
| `2D` (default) | One `TSNE.fit_transform` call; writes `X_2d`. |
| `3D` | One call; writes `X_3d`. |
| `2D+3D` | **Two** calls per layer/stage; writes both. |

The 3D fit reuses the 2D fit's `init`, `random_state` and `max_iter`, so the
two embeddings stay directly comparable, and stage alignment carries over
unchanged from the 2D pipeline.

**Cost.** Because `--tsne_feature_layer` and `--tsne_stages` both accept
multiple values, the runner performs one fit per combination:

```
N_fits = N_LAYERS × N_STAGES × N_COMPONENTS
```

Selecting `2D+3D` therefore *doubles* an already multiplicative total — five
layers across three stages goes from 15 fits to 30. Keep the `2D` default for
layer/stage sweeps and reserve `3D` / `2D+3D` for one-off manual comparisons
where the rotatable view is worth the extra compute.

**Backward compatibility.** Artifacts written before this feature contain 2D
coordinates only. `has_2d()` / `has_3d()` guard every read, so old run
directories still load correctly in the Dash app.

---

## 4 — Output Files

All outputs are written under `tools/tsne_visualization/plots/` — i.e. relative
to the `tsne_runner.py` script's own directory, **not** the repo root
(configurable via `save_dir` in the Python API). Each run is logically grouped
into a timestamped directory containing a `metadata.json` file.

```
tools/tsne_visualization/plots/
└── <feature_space>/          # "layer" or "pixels"
    └── <model_name>/
        ├── run_YYYYMMDD_HHMMSS/
        │   ├── metadata.json
        │   ├── tsne/
        │   │   └── tsne_<model>[_layer-<layer>][_<tag>][_classes<ids>].png
        │   ├── misclassified/
        │   │   └── misclassified_<model>[…].png
        │   └── dash_data/
        │       └── tsne_<model>[_layer-<layer>][_<tag>][_classes<ids>].npz
        └── sweep/                       # only when --tsne_sweep is used (§5)
            └── run_YYYYMMDD_HHMMSS/
                ├── sweep_manifest.json
                └── <per-step>.npz       # only with --tsne_sweep_save_dash_artifact
```

### Static PNG (`tsne/`)
A matplotlib scatter plot with:
- **Grey dots** — training samples (position only, not evaluated).
- **Coloured dots** — test samples, coloured by true class label.
- **Red ×** — misclassified test samples.
- **Class label text** — centred on each cluster.
- Accuracy annotation in the lower-left corner.

Produced only for the 2D embedding, i.e. with `--tsne_components 2D` or `2D+3D`.
3D results are viewed in the Dash app.

### Misclassification grid (`misclassified/`)
Saved only when `--show_misclassifications` is set. Each cell shows the raw
image with `true=` / `pred=` labels in red.

### Dash artifact (`.npz`)
A compressed NumPy archive consumed by the interactive app (see [Dash App](dash_app.md)).
Contains `X_2d` and/or `X_3d` plus everything needed to reconstruct the plot and
preview test images without re-running t-SNE.

### Run Metadata (`metadata.json`)
Automatically generated on every run. Logs hyper-parameters (seed, max_train, etc.) ensuring that artifacts in the same `run_` directory are completely comparable.

---

## 5 — Point-Count Sweep (KL Divergence)

An exploratory mode that measures how t-SNE embedding quality degrades as more
points are rendered, by tracking t-SNE's own KL-divergence loss across a
schedule of training-set sizes. Running it in both dimensionalities quantifies
how much the extra dimension actually buys at each scale.

`--tsne_sweep` replaces the normal fit; the flags below apply only in this mode.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--tsne_sweep` | flag | `False` | Master switch — run the sweep instead of a normal t-SNE fit. |
| `--tsne_sweep_train_sizes` | int… | — | The `N` values to test. Their count is `n_steps` in the cost formula below. |
| `--tsne_sweep_test_size` | int | `1000` | Test-set size held **constant** across every step, so KL values stay comparable. |
| `--tsne_sweep_feature_layer` | str | `penultimate` | Layer whose features to sweep. Accepts the §2 aliases or an explicit module path. |
| `--tsne_sweep_n_repeats` | int | `5` | Seed repeats per step, feeding the multi-seed KL median. |
| `--tsne_sweep_save_dash_artifact` | flag | `False` | Save each step's `.npz` coordinates so per-step scatters are inspectable in the dashboard. Off by default — it persists a full coordinate set per step, where the curve itself needs only a scalar. |

**Why repeats.** Even with `init="pca"`, scikit-learn's PCA solver has a random
component depending on data shape, governed by `random_state`. Since t-SNE's
optimization landscape is non-convex, a nudged starting point can settle into a
different minimum — different final coordinates and a different KL divergence
for identical input. `--tsne_sweep_n_repeats` fits several seeds and keeps the
repeat closest to the group median, giving a representative run rather than a
lucky or unlucky outlier. The default of 5 removes ~55 % of seed noise; beyond
that each additional repeat costs the same fixed increment for progressively
less benefit.

**Cost.** These factors multiply:

```
N_fits = n_steps × n_repeats × n_components
```

Eight training sizes at the default five repeats in both dimensionalities is
already 80 fits.

### Example

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name lenet5 \
    --tsne_sweep \
    --tsne_sweep_train_sizes 1000 2000 3000 5000 7500 10000 \
    --tsne_sweep_test_size 500 \
    --tsne_sweep_feature_layer fc1 \
    --tsne_sweep_n_repeats 5 \
    --tsne_sweep_save_dash_artifact \
    --tsne_components 2D+3D
```

Results are written to `.../<model>/sweep/run_YYYYMMDD_HHMMSS/` and viewed on
the `/kld-sweep` page of the Dash app.

---

## 6 — Common Workflows

### A — Quick sanity check (raw pixels, all classes)

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name lenet5 \
    --tsne_feature_space pixels \
    --tsne_max_train 1000 --tsne_max_test 500
```

### B — Layer embedding, compare exact vs. quantized

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name resnet \
    --tsne_stages exact quantized \
    --tsne_feature_layer penultimate
```

### C — Multi-layer comparison

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name lenet5 \
    --tsne_stages exact \
    --tsne_feature_layer conv1 penultimate
```

### D — Approximate hardware, specific classes, with image grid

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name resnet \
    --tsne_stages approximate \
    --tsne_multiplier_path multipliers/my_table.npy \
    --tsne_classes 3 5 8 \
    --show_misclassifications \
    --tsne_feature_layer conv2
```

### E — Rotatable 3D comparison of all three stages

```bash
python3 tools/tsne_visualization/tsne_runner.py \
    --model_name lenet5 \
    --tsne_stages exact quantized approximate \
    --tsne_multiplier_path multipliers/my_table.npy \
    --tsne_components 2D+3D
```

### F — View an existing run in the Dash app

```bash
python3 tools/web_visualization/app.py
# In the browser (http://localhost:8051), open the "t-SNE Compare" page and paste a
# run directory path. Relative paths are resolved against the repo root, so use e.g.:
# tools/tsne_visualization/plots/layer/resnet/run_20260515_181919
```
