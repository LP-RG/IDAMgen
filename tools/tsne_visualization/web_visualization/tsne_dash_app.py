import argparse
import base64
import os
import sys
from io import BytesIO
from pathlib import Path

TSNE_VIS_DIR = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parents[3]
# Ensure project root is importable when running apps/tsne_dash_app.py directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from PIL import Image
from tools.tsne_visualization.tsne_utils import (
    get_misclassified_indices,
    load_dash_artifact,
)

def _image_to_data_url(flat_img, image_shape):
    """Convert a flattened sample into a base64 PNG data URL for html.Img."""
    c, h, w = image_shape
    img = flat_img.reshape(c, h, w).astype(np.float32, copy=False)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # Match previous matplotlib gray_r visual behavior.
    if c == 1:
        pil_img = Image.fromarray(255 - img_u8[0], mode="L")
    else:
        pil_img = Image.fromarray(np.transpose(img_u8, (1, 2, 0)), mode="RGB")

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_plot_figure(data):
    """Build the interactive Plotly t-SNE figure and misclassification mapping."""
    import plotly.graph_objects as go
    import seaborn as sns

    X_2d = data["X_2d"]
    y_all = data["y_all"]
    test_mask = data["test_mask"].astype(bool)
    y_test_sub = data["y_test_sub"]
    y_pred_sub = data["y_pred_sub"]

    train_mask = ~test_mask
    wrong_mask = np.zeros(len(y_all), dtype=bool)
    wrong_mask[test_mask] = (y_test_sub != y_pred_sub)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=X_2d[train_mask, 0],
        y=X_2d[train_mask, 1],
        mode="markers",
        marker=dict(size=5, color="#cccccc", opacity=0.4),
        name="Train (not evaluated)",
        hoverinfo="skip",
        showlegend=True,
    ))

    unique_labels = np.unique(y_all)
    palette = np.array(sns.color_palette("hls", len(unique_labels)))
    label_to_color = {
        int(lab): f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        for lab, rgb in zip(unique_labels, palette)
    }

    for lab in unique_labels:
        mask = test_mask & (y_all == lab)
        if not np.any(mask):
            continue
        fig.add_trace(go.Scattergl(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode="markers",
            marker=dict(size=8, color=label_to_color[int(lab)], opacity=0.85),
            name=f"Test label {int(lab)}",
            customdata=np.full(mask.sum(), -1, dtype=np.int64),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>true=%{text}<extra></extra>",
            text=np.full(mask.sum(), int(lab), dtype=np.int64),
            showlegend=True,
        ))

    wrong_indices = np.where(wrong_mask)[0]
    wrong_test_indices = get_misclassified_indices(y_test_sub, y_pred_sub)
    mis_trace_idx = None
    if len(wrong_indices) > 0:
        mis_trace_idx = len(fig.data)
        fig.add_trace(go.Scattergl(
            x=X_2d[wrong_indices, 0],
            y=X_2d[wrong_indices, 1],
            mode="markers",
            marker=dict(symbol="x", size=14, color="red", line=dict(width=2, color="red")),
            name=f"Misclassified ({len(wrong_indices)})",
            customdata=wrong_test_indices,
            hovertemplate=("x=%{x:.2f}<br>y=%{y:.2f}<br>"
                           "true=%{text}<br>pred=%{meta}<extra></extra>"),
            text=y_test_sub[wrong_test_indices],
            meta=y_pred_sub[wrong_test_indices],
            showlegend=True,
        ))

    ref_mask = test_mask if np.any(test_mask) else np.ones(len(y_all), dtype=bool)
    for lab in unique_labels:
        pts = X_2d[ref_mask & (y_all == lab)]
        if len(pts) == 0:
            continue
        xtext, ytext = np.median(pts, axis=0)
        fig.add_annotation(
            x=float(xtext),
            y=float(ytext),
            text=str(int(lab)),
            showarrow=False,
            font=dict(size=20, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
        )

    title_value = data.get("title", "t-SNE")
    # Backward/forward compatible: artifacts may expose title as numpy scalar
    # (with .item()) or already as a plain Python string.
    if hasattr(title_value, "item"):
        title_value = title_value.item()
    title = str(title_value)
    fig.update_layout(
        title=title,
        template="simple_white",
        width=1000,
        height=800,
        clickmode="event+select",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig, mis_trace_idx, wrong_test_indices


def _blank_figure(message="Load an artifact to visualize t-SNE."):
    """Return a placeholder figure shown before any artifact is loaded."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        template="simple_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=18, color="#666"),
            )
        ],
    )
    return fig


def _load_artifact(artifact_path):
    """Load an artifact and build both figure and lightweight callback state."""
    data = load_dash_artifact(artifact_path)
    fig, mis_trace_idx, _ = _build_plot_figure(data)
    payload = {
        "path": artifact_path,
        "mis_trace_idx": mis_trace_idx,
    }
    return fig, payload


def _load_run_dir(run_dir):
    """Scan a run directory for metadata and artifacts."""
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise ValueError(f"Path does not exist: {run_path}")
        
    if run_path.is_file() and run_path.suffix == ".npz":
        return {run_path.name: str(run_path)}, {}
        
    metadata = {}
    meta_path = run_path / "metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    artifacts = {}
    for npz_path in run_path.rglob("dash_data/*.npz"):
        artifacts[npz_path.stem] = str(npz_path)
        
    if not artifacts:
        raise ValueError("No .npz artifacts found in the specified directory.")
        
    return artifacts, metadata

def create_app():
    app = Dash(__name__)
    initial_msg = "Load a run, then click a red cross."
    app.layout = html.Div(
        style={
            "display": "flex", "flexDirection": "column", "gap": "16px",
            "padding": "16px", "minHeight": "100vh", "boxSizing": "border-box",
            "backgroundColor": "#f4f4f9", "fontFamily": "sans-serif"
        },
        children=[
            # TOP ROW
            html.Div(
                style={
                    "display": "flex", "flexDirection": "row", "gap": "12px",
                    "alignItems": "center", "backgroundColor": "white",
                    "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                },
                children=[
                    html.H3("t-SNE Compare", style={"margin": "0", "minWidth": "160px"}),
                    dcc.Input(id="run-path", type="text", style={"flex": 1, "padding": "8px"},
                              placeholder="Path to run directory (e.g. plots/pixels/resnet/run_2026...)"),
                    html.Button("Load Run", id="load-run", n_clicks=0, style={"padding": "8px 16px"}),
                    html.Button("Clear", id="clear-reset", n_clicks=0, style={"padding": "8px 16px"}),
                    html.Div(id="status-log", style={"marginLeft": "16px", "fontSize": "12px", "color": "#666", "flex": 1, "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"})
                ]
            ),
            # METADATA ROW
            html.Div(
                id="metadata-container",
                style={"display": "none"},
                children=[
                    html.H4("Run Metadata", style={"margin": "0 0 8px 0"}),
                    html.Pre(id="run-metadata-display", style={"margin": "0", "fontSize": "13px", "whiteSpace": "pre-wrap", "color": "#333"})
                ]
            ),
            # MIDDLE ROW
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "16px", "flex": 1, "minHeight": "600px"},
                children=[
                    html.Div(
                        style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
                        children=[
                            html.Div(
                                style={"display": "flex", "flexDirection": "row", "alignItems": "center", "gap": "8px"},
                                children=[
                                    html.Label("Left View:", style={"fontWeight": "bold"}),
                                    dcc.Dropdown(id="left-dropdown", style={"flex": 1}, clearable=False),
                                ]
                            ),
                            dcc.Loading(dcc.Graph(id="left-graph", figure=_blank_figure(), style={"flex": 1}, config={"displaylogo": False}))
                        ]
                    ),
                    html.Div(
                        style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
                        children=[
                            html.Div(
                                style={"display": "flex", "flexDirection": "row", "alignItems": "center", "gap": "8px"},
                                children=[
                                    html.Label("Right View:", style={"fontWeight": "bold"}),
                                    dcc.Dropdown(id="right-dropdown", style={"flex": 1}, clearable=True, placeholder="Select an artifact to compare..."),
                                ]
                            ),
                            dcc.Loading(dcc.Graph(id="right-graph", figure=_blank_figure("Select a second artifact to compare."), style={"flex": 1}, config={"displaylogo": False}))
                        ]
                    ),
                ]
            ),
            # BOTTOM ROW
            html.Div(
                style={
                    "display": "flex", "flexDirection": "row", "gap": "16px", "backgroundColor": "white",
                    "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "alignItems": "center", "minHeight": "140px"
                },
                children=[
                    html.Div(style={"width": "200px"}, children=[
                        html.H4("Misclassified Preview", style={"margin": "0 0 8px 0"}),
                        html.Div(id="preview-message", children=initial_msg, style={"fontSize": "13px"}),
                    ]),
                    html.Img(id="preview-image", style={"height": "120px", "objectFit": "contain", "border": "1px solid #ddd", "backgroundColor": "#fafafa", "display": "none"}),
                    html.Pre(id="preview-meta", style={"whiteSpace": "pre-wrap", "margin": "0", "fontSize": "13px", "flex": 1}),
                ]
            ),
            dcc.Store(id="run-artifacts"),
            dcc.Store(id="left-state"),
            dcc.Store(id="right-state"),
        ]
    )

    @app.callback(
        Output("run-artifacts", "data"),
        Output("left-dropdown", "options"),
        Output("right-dropdown", "options"),
        Output("left-dropdown", "value"),
        Output("right-dropdown", "value"),
        Output("status-log", "children"),
        Output("metadata-container", "style"),
        Output("run-metadata-display", "children"),
        Input("load-run", "n_clicks"),
        Input("clear-reset", "n_clicks"),
        State("run-path", "value"),
        prevent_initial_call=True,
    )
    def _load_run_directory(load_clicks, clear_clicks, run_path):
        trigger = callback_context.triggered_id
        meta_style_hidden = {"display": "none"}
        meta_style_visible = {"display": "block", "backgroundColor": "white", "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}
        
        if trigger == "clear-reset":
            return None, [], [], None, None, "Cleared.", meta_style_hidden, ""
            
        if not run_path:
            return None, [], [], None, None, "Please provide a run directory path.", meta_style_hidden, ""
            
        try:
            if not os.path.isabs(run_path):
                full_path = str((ROOT / run_path).resolve())
            else:
                full_path = run_path
            artifacts, metadata = _load_run_dir(full_path)
            
            options = [{"label": k, "value": v} for k, v in artifacts.items()]
            keys = sorted(artifacts.keys())
            left_val = artifacts[keys[0]]
            right_val = artifacts[keys[1]] if len(keys) > 1 else None
            
            status = f"Loaded {len(artifacts)} artifacts from {Path(full_path).name}"
            
            import json
            meta_str = json.dumps(metadata, indent=2) if metadata else "No metadata found."
            return artifacts, options, options, left_val, right_val, status, meta_style_visible, meta_str
        except Exception as e:
            return None, [], [], None, None, f"Error: {e}", meta_style_hidden, ""

    @app.callback(
        Output("left-graph", "figure"),
        Output("left-state", "data"),
        Input("left-dropdown", "value"),
        Input("clear-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_left_graph(artifact_path, clear_clicks):
        if callback_context.triggered_id == "clear-reset" or not artifact_path:
            return _blank_figure(), None
        try:
            return _load_artifact(artifact_path)
        except Exception as e:
            return _blank_figure(f"Error loading artifact: {e}"), None

    @app.callback(
        Output("right-graph", "figure"),
        Output("right-state", "data"),
        Input("right-dropdown", "value"),
        Input("clear-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_right_graph(artifact_path, clear_clicks):
        if callback_context.triggered_id == "clear-reset" or not artifact_path:
            return _blank_figure("Select a second artifact to compare."), None
        try:
            return _load_artifact(artifact_path)
        except Exception as e:
            return _blank_figure(f"Error loading artifact: {e}"), None

    @app.callback(
        Output("preview-message", "children"),
        Output("preview-image", "src"),
        Output("preview-image", "style"),
        Output("preview-meta", "children"),
        Input("left-graph", "clickData"),
        Input("right-graph", "clickData"),
        Input("clear-reset", "n_clicks"),
        State("left-state", "data"),
        State("right-state", "data"),
        prevent_initial_call=True,
    )
    def _update_preview(left_click, right_click, clear_clicks, left_state, right_state):
        trigger = callback_context.triggered_id
        img_style_hidden = {"display": "none"}
        img_style_visible = {"height": "120px", "objectFit": "contain", "border": "1px solid #ddd", "backgroundColor": "white", "display": "block"}
        
        if trigger == "clear-reset":
            return initial_msg, "", img_style_hidden, ""
            
        click_data = left_click if trigger == "left-graph" else right_click
        state = left_state if trigger == "left-graph" else right_state
        
        if not state or not click_data:
            raise PreventUpdate
            
        mis_trace_idx = state.get("mis_trace_idx")
        if mis_trace_idx is None:
            return "No misclassified points in this artifact.", "", img_style_hidden, ""
            
        point = click_data["points"][0]
        if int(point.get("curveNumber", -1)) != mis_trace_idx:
            return "Only red misclassified crosses are interactive.", "", img_style_hidden, ""
            
        pi = int(point.get("pointIndex", -1))
        artifact = load_dash_artifact(state["path"])
        wrong_test_indices = get_misclassified_indices(artifact["y_test_sub"], artifact["y_pred_sub"])
        
        if pi < 0 or pi >= len(wrong_test_indices):
            return "Could not resolve clicked point.", "", img_style_hidden, ""
            
        test_idx = int(wrong_test_indices[pi])
        image_shape = artifact["image_shape"]
        X_test_pixels = artifact["X_test_pixels"]
        img_src = _image_to_data_url(X_test_pixels[test_idx], image_shape)
        
        meta = (
            f"Artifact: {Path(state['path']).name}\n"
            f"Test Index: {test_idx}\n"
            f"True Label: {int(artifact['y_test_sub'][test_idx])}\n"
            f"Predicted Label: {int(artifact['y_pred_sub'][test_idx])}"
        )
        return "Selected misclassified sample:", img_src, img_style_visible, meta

    return app


def main():
    parser = argparse.ArgumentParser(description="Dash app for t-SNE artifacts and on-demand runs.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    print(ROOT)
    main()
