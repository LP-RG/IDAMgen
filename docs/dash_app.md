# Interactive Dash App (`apps/tsne_dash_app.py`)

> **Status:** Work in progress — this section will be expanded as the app develops.

The Dash app provides an interactive browser-based viewer for t-SNE artifacts
produced by the t-SNE tool (see [t-SNE Visualization](tsne.md)).

---

## Launching

```bash
python3 apps/tsne_dash_app.py [--host 0.0.0.0] [--port 8050] [--debug]
```

Then open `http://localhost:8050` in your browser.

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host address to bind to. |
| `--port` | `8050` | Port to listen on. |
| `--debug` | `False` | Enable Dash debug/hot-reload mode. |

---

## Usage

1. Paste the path to a run directory (e.g., `plots/layer/resnet/run_2026...`)
   into the input field at the top. (You may also paste a single `.npz` file for backward compatibility).
2. Click **"Load Run"**. The app will automatically discover all artifacts in that run and populate the dropdown menus.
3. **Compare Artifacts:** Use the dropdowns above the **Left View** and **Right View** to instantly display and compare different stages (e.g. `exact` vs `approximate`).
4. **Inspect Errors:** Click a red × (misclassified point) on either plot to see the raw image and metadata in the **Misclassified Preview** panel at the bottom.
5. Click **"Clear"** to unload the current artifacts.

> **Note:** Only red × misclassified crosses are interactive.
> Clicking grey or coloured dots shows a message but no image preview.

---