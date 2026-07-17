# Interactive Dash App (`tools/tsne_visualization/web_visualization/tsne_dash_app.py`)

> **Status:** Work in progress — this section will be expanded as the app develops.

The Dash app provides an interactive browser-based viewer for t-SNE artifacts
produced by the t-SNE tool (see [t-SNE Visualization](tsne.md)).

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

---

## Usage

1. **Loading a run**
Paste the path to a run directory (e.g., `tools/tsne_visualization/plots/layer/resnet/run_2026...`) into the input field at the top. 
Relative paths are resolved against the repo root. (You may also paste a single `.npz` file for backward compatibility).
Click __"Load Run"__. The app will automatically discover all artifacts in that run and populate the dropdown menus.

2. **Comparing Artifacts**
Use the dropdowns above the __"Left View"__ and __"Right View"__ to instantly display and compare different stages (e.g. `exact` vs `approximate`).

3. **Interacting with a Plot**
   - __Hide Cluster__ - click its legend entry to hide those points; click again to restore.
   - __Isolate Cluster__ - double-click a legend entry to display only that cluster; double-click again to restore all.
   - __Toolbar__ - hover over the top-right corner of a plot to reveal Plotly's toolbar (zoom, pan, rotate, reset).

4. **Inspecting Points**
Click any colored test point to see its raw image and metadata in the __"Preview"__ panel at the bottom. 
Misclassified points are marked with a red x, clicking one shows the same preview plus its predicted-vs-true labels. 
Train points (grey) aren't previewable — clicking one shows a message only

5. **Clearing**
Click __"Clear"__ to unload the current artifacts.

---