# Interactive Dash App (`tools/tsne_visualization/web_visualization/tsne_dash_app.py`)

The Dash app is a single **multi-page** application hosting every interactive
viewer in the project. It replaces the former standalone
`tools/tsne_visualization/web_visualization/tsne_dash_app.py`.

---

## Launching

Run from the repo root:

```bash
python3 tools/tsne_visualization/web_visualization/tsne_dash_app.py [--host 0.0.0.0] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host address to bind to. |
| `--port` | `8050` | Port to listen on. |
| `--debug` | `False` | Enable Dash debug/hot-reload mode. |

A shared navigation bar links the three pages. Its links are generated from
`dash.page_registry` and ordered by URL path, so adding a page needs only a new
module in `pages/` containing a `register_page()` call — no manual nav edits.

| Page | Route | Purpose |
|---|---|---|
| **t-SNE Compare** | `/` | Compare embeddings across stages and layers |
| **N_tot vs KL_div** | `/kld-sweep` | Point-count sweep results |
| **Train History** | `/train-history` | Per-epoch embeddings and training curves |


### Structure

```
tools/web_visualization/
├── app.py              ← the only Dash(...) instance; entry point
├── navbar.py           ← shared navbar, built from dash.page_registry
├── shared_figures.py   ← helpers (never calls register_page)
├── assets/             ← static files (css, icons)
└── pages/
    ├── tsne_dash_app.py
    ├── kld_dash_app.py
    └── train_dash_app.py
```

> **Adding a page.** Expose a module-level `layout` and register callbacks with
> the bare `@callback` decorator (**not** `@app.callback`) — this is what keeps
> pages import-safe and free of circular imports. Put shared helpers in
> `shared_figures.py` rather than importing one page from another, since the
> Dash loader would register that page, and its callbacks, a second time.
> Every `(component_id, property)` pair must be owned by exactly one callback
> across the entire app.

---

## 1 — t-SNE Compare (`/`)

Interactive viewer for the `.npz` artifacts produced by the t-SNE tool
(see [t-SNE Visualization](tsne.md)).

1. **Loading a run**
   Paste the path to a run directory (e.g.
   `tools/tsne_visualization/plots/layer/resnet/run_2026...`) into the input
   field. Relative paths are resolved against the repo root; a single `.npz`
   file also works for backward compatibility.
   Click **"Load Run"** — the app scans the directory recursively for
   `dash_data/*.npz`, populates both dropdowns, shows the run's
   `metadata.json` in a panel above the graphs, and reports the number of
   artifacts found on the status line.

2. **Comparing artifacts**
   The **"Left View"** and **"Right View"** dropdowns each select one artifact
   (i.e. one layer/stage combination), so you can pin `exact` on the left and
   step the right dropdown through `quantized` and `approximate`. The left
   dropdown defaults to the first artifact, the right to the second, so a
   freshly loaded run already shows a comparison.

3. **2D / 3D panels**
   The layout adapts to what the run actually contains. A `2D` or `3D` run
   shows two side-by-side panels; a `2D+3D` run lays them out as a 2×2 grid
   with 2D on the top row and 3D on the bottom. A panel whose dimensionality
   was not computed is hidden rather than left blank.

4. **Interacting with a plot**
   - **Hide cluster** — click its legend entry; click again to restore.
   - **Isolate cluster** — double-click a legend entry; double-click to restore.
   - **Toolbar** — hover the plot's top-right corner for Plotly's zoom, pan,
     rotate and reset controls. 3D plots support rotation by dragging.

5. **Inspecting points**
   Click **any** test point — correct or misclassified — to see its raw image
   and metadata in the **"Preview"** panel. Misclassified points are marked
   with a red ×, and their preview also shows predicted-vs-true labels. Train
   points (grey) are not previewable and show a message only.

6. **Clearing**
   **"Clear"** resets dropdowns, graphs, metadata panel and preview at once.

---

## 2 — KL-Divergence Sweep (`/kld-sweep`)

Viewer for the point-count sweep (see [t-SNE Visualization §5](tsne.md)).

1. **Loading a sweep**
   Paste either a sweep run directory or a direct path to its
   `sweep_manifest.json`; a directory has the filename appended automatically.
   - **"Load Run"** — fresh parse; resets the step slider to the step nearest
     the production point count.
   - **"Update"** — re-reads the same manifest while leaving the slider where
     it is. Useful for picking up new steps from a sweep that is still running.

2. **Main curve plot**
   KL divergence against `N_total` on a log-scaled x-axis, with one median
   trend-line per dimensionality so 2D and 3D read against each other. Each
   step is drawn as a **boxplot over its seed repeats**, which shows at a
   glance whether two curves are genuinely separated or merely within seed
   noise. A dotted vertical line marks the production point count.

3. **Step slider and per-step scatters**
   The slider steps through the sweep and renders that step's 2D and 3D
   embeddings side by side. Marks are labelled by `N_total`; **steps shown in
   red have no saved coordinates**, which happens when the sweep was run
   without `--tsne_sweep_save_dash_artifact`. A panel whose dimensionality was
   not computed is hidden.

---

## 3 — Train History (`/train-history`)

Viewer for the per-epoch embeddings produced by the training-history pipeline
(see [Training §6](training.md)).

1. **Loading a run**
   Paste the path to the pipeline's `dashboard_data` directory (e.g.
   `tools/train_visualization/dashboard_data`) and click **"Load"**. The page
   reads `train_manifest.json` plus the selected stage's parsed-log JSON.

2. **Stage selector**
   Radio buttons switch between `exact`, `quant_q8` and `approx_a8_my_table`.
   Changing stage reloads that stage's log **and** rescales the epoch slider to
   its epoch count in one action — necessary because the three stages train for
   different numbers of epochs.

3. **Curve plot**
   Two vertically stacked subplots on a shared epoch axis: train loss on top
   (log-scaled y, so the early collapse and later fine movement are both
   readable) and train/test accuracy below. A vertical line spanning both
   subplots marks the epoch selected on the slider.

4. **Epoch slider and 3D scatter**
   The slider selects an epoch and renders its warm-started 3D embedding. The
   camera angle is held fixed across epoch changes, so scrubbing through
   training shows the clusters moving rather than the viewpoint.

---
