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
import dash
from dash import Dash, Input, Output, State, callback, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from PIL import Image

from tools.web_visualization.shared_figures import _blank_figure
from tools.tsne_visualization.tsne_utils import (
    get_misclassified_indices,
    load_dash_artifact,
    has_2d,
    has_3d
)

dash.register_page(__name__, path="/", name="t-SNE Compare")

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
    
    if has_2d(data):
        X_2d = data["X_2d"]    
    else:
        X_2d = []
    if has_3d(data):
        X_3d = data["X_3d"]    
    else:
        X_3d = []

    y_all = data["y_all"]
    test_mask = data["test_mask"].astype(bool)
    y_test_sub = data["y_test_sub"]
    y_pred_sub = data["y_pred_sub"]

    train_mask = ~test_mask
    wrong_mask = np.zeros(len(y_all), dtype=bool)
    wrong_mask[test_mask] = (y_test_sub != y_pred_sub)

    fig_2d = go.Figure()
    fig_3d = go.Figure()

    # layout[i] describes curveNumber i for fig_2d.data and fig_3d.data
    layout = []

    # train trace
    layout.append(None) # train points not resolvable
    if has_2d(data):
        fig_2d.add_trace(go.Scattergl(
            x=X_2d[train_mask, 0],
            y=X_2d[train_mask, 1],
            mode="markers",
            marker=dict(size=4, color="#cccccc", opacity=0.6),
            name="Train (NE)",
            hoverinfo="skip",
            showlegend=True,
        ))
    if has_3d(data):
        fig_3d.add_trace(go.Scatter3d(
            x=X_3d[train_mask, 0],
            y=X_3d[train_mask, 1],
            z=X_3d[train_mask, 2],
            mode="markers",
            marker=dict(size=4, color="#cccccc", opacity=0.6),
            name="Train (NE)",
            hoverinfo="skip",
            showlegend=True,        
        ))

    unique_labels = np.unique(y_all)
    palette = np.array(sns.color_palette("hls", len(unique_labels)))
    label_to_color = {
        int(lab): f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        for lab, rgb in zip(unique_labels, palette)
    }

    # test traces
    n_train = int(train_mask.sum())
    for lab in unique_labels:
        mask = test_mask & (y_all == lab)
        if not np.any(mask):
            continue
        test_sub_indices = np.where(mask)[0] - n_train
        layout.append(test_sub_indices)
        if has_2d(data):
            fig_2d.add_trace(go.Scattergl(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1],
                mode="markers",
                marker=dict(size=7, color=label_to_color[int(lab)], opacity=0.85),
                name=f"Test label {int(lab)}",
                customdata=np.full(mask.sum(), -1, dtype=np.int64),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>true=%{text}<extra></extra>",
                text=np.full(mask.sum(), int(lab), dtype=np.int64),
                showlegend=True
            ))
        if has_3d(data):
            fig_3d.add_trace(go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode="markers",
                marker=dict(size=7, color=label_to_color[int(lab)], opacity=1),
                name=f"Test label {int(lab)}",
                customdata=np.full(mask.sum(), -1, dtype=np.int64),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>true=%{text}<extra></extra>",
                text=np.full(mask.sum(), int(lab), dtype=np.int64),
                showlegend=True
            ))

    # misclassified trace
    wrong_indices = np.where(wrong_mask)[0]
    wrong_test_indices = get_misclassified_indices(y_test_sub, y_pred_sub)
    if len(wrong_indices) > 0:
        layout.append(wrong_test_indices)
        if has_2d(data):
            fig_2d.add_trace(go.Scattergl(
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
                showlegend=True
            ))
        if has_3d(data):
            fig_3d.add_trace(go.Scatter3d(
                x=X_3d[wrong_indices, 0],
                y=X_3d[wrong_indices, 1],
                z=X_3d[wrong_indices, 2],
                mode="markers",
                marker=dict(symbol="x", size=10, color="red"),
                name=f"Misclassified ({len(wrong_indices)})",
                customdata=wrong_test_indices,
                hovertemplate=("x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>"
                            "true=%{text}<br>pred=%{meta}<extra></extra>"),
                text=y_test_sub[wrong_test_indices],
                meta=y_pred_sub[wrong_test_indices],
                showlegend=True
            ))

    # cluster label annotations
    if np.any(test_mask):   # if test_mask has any points use those
        ref_mask = test_mask
    else:   # otherwise use entire dataset
        ref_mask = np.ones(len(y_all), dtype=bool)

    scene_annotations = []    
    for lab in unique_labels:
        d = 2  # halo thickness in pixels
        offsets = [(-d, -d), (-d, 0), (-d, d), ( 0, -d), ( 0, d), ( d, -d), ( d, 0), ( d, d)]
        if has_2d(data):
            pts_2d = X_2d[ref_mask & (y_all == lab)]
            if len(pts_2d) == 0:
                continue
            xtext, ytext = np.median(pts_2d, axis=0)
            
            
            for i in range(len(offsets)):
                fig_2d.add_annotation(
                    x=float(xtext), y=float(ytext),
                    xshift = offsets[i][0], yshift = offsets[i][1],
                    text=str(int(lab)),
                    showarrow=False,
                    font=dict(size=20, color="white"),
                    # bgcolor="rgba(255,255,255,0.8)",
                )
            fig_2d.add_annotation(
                x=float(xtext),
                y=float(ytext),
                text=str(int(lab)),
                showarrow=False,
                font=dict(size=20, color="black")
            )

        if has_3d(data):
            pts_3d = X_3d[ref_mask & (y_all == lab)]
            if len(pts_3d) == 0:
                continue            
            xtext, ytext, ztext = np.median(pts_3d, axis=0)
            for i in range(len(offsets)):
                scene_annotations.append(dict(
                    x=float(xtext), y=float(ytext), z=float(ztext),
                    xshift = offsets[i][0], yshift = offsets[i][1],
                    text=str(int(lab)),
                    showarrow=False,
                    font=dict(size=20, color="white")
                ))
            scene_annotations.append(dict(
                x=float(xtext),
                y=float(ytext),
                z=float(ztext),
                text=str(int(lab)),
                showarrow=False,
                font=dict(size=20, color="black"),
            ))
            fig_3d.update_layout(scene = dict(annotations = scene_annotations))

    # Backward/forward compatible: artifacts may expose title as numpy scalar
    # (with .item()) or already as a plain Python string.
    title_value = data.get("title", "t-SNE")
    if hasattr(title_value, "item"):
        title_value = title_value.item()
    title = str(title_value)

    if has_2d(data):
        fig_2d.update_layout(
            template="plotly_white",
            autosize = True,
            clickmode="event+select",
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top", entrywidth=0, entrywidthmode='pixels'),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(visible=True),
            yaxis=dict(visible=True, scaleanchor="x", scaleratio=1)
        )
    if has_3d(data):
        fig_3d.update_layout(
            template="plotly_white",
            clickmode="event+select",
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                aspectmode="data",
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
    return fig_2d, fig_3d, layout, title


def _load_artifact(artifact_path):
    """Load an artifact and build both figure and lightweight callback state."""
    data = load_dash_artifact(artifact_path)
    fig_2d, fig_3d, layout, title = _build_plot_figure(data)
    payload = {
        "path": artifact_path,
        "layout": [None if i is None else i.tolist() for i in layout]
    }
    return fig_2d, fig_3d, payload, title


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


# --- Page layout -------------------------------------------------------
# this used to be built inside a `create_app()`. 
# Under Dash Pages, the shared app instance lives once in web_visualization/app.py, 
# and this module just needs to expose a module-level `layout` plus `@callback`-registered functions.
# Both get picked up automatically the moment app.py imports this page.


initial_msg = "Load a run, then click a red cross."
layout = html.Div(
    style={
        "display": "flex", "flexDirection": "column", "gap": "16px",
        "padding": "16px", "minHeight": "100vh", "boxSizing": "border-box",
        "backgroundColor": "#f4f4f9", "fontFamily": "sans-serif"
    },
    children=[
        # TOP ROW
        html.Div(
            style={
                "display": "flex", "flexDirection": "column", "gap": "12px",
                "alignItems": "center", "backgroundColor": "white",
                "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=[
                html.H2("t-SNE Visualization — Compare Embeddings Across Runs", style={"margin": "0", "minWidth": "160px"}),
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "min-width": "600px"},
                    children=[
                        dcc.Input(id="run-path", type="text", style={"flex": 1, "padding": "8px"},
                                    placeholder="Path to run directory (e.g. tsne_visualization/plots/pixels/resnet/run_2026...)"),
                        html.Button("Load Run", id="load-run", n_clicks=0, style={"padding": "8px 16px"}),
                        html.Button("Clear", id="clear-reset", n_clicks=0, style={"padding": "8px 16px"})
                    ]
                ),
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
            style={"display": "flex", "flexDirection": "row", "gap": "16px", "flex": 1, "minHeight": "1100px"},
            children=[
                html.Div(
                    style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
                    children=[
                        html.Div(
                            style={"display": "flex", "flexDirection": "row", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Label("Left View:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(id="left-dropdown", style={"flex": 1}, clearable=False)
                            ]
                        ),
                        html.Div(
                            id = "left-graph-2d-container",
                            style = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0},
                            children = [
                                html.Div(id = "left-graph-2d-title", style = {"fontWeight": "bold", "textAlign": "center"}),
                                dcc.Loading(
                                    dcc.Graph(id="left-graph-2d", figure=_blank_figure(), style={"flex": 1, "minWidth": 0}, config={"displaylogo": False, "responsive": True}),
                                    style={"flex": 1, "display": "flex", "width": "100%", "minWidth": 0},
                                    parent_style={"flex": 1, "display": "flex", "width": "100%"},
                                )

                            ]
                        ),
                        html.Div(
                            id = "left-graph-3d-container",
                            style = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0},
                            children = [
                                html.Div(id = "left-graph-3d-title", style = {"fontWeight": "bold", "textAlign": "center"}),
                                dcc.Loading(
                                    dcc.Graph(id="left-graph-3d", figure=_blank_figure(), style={"flex": 1, "minWidth": 0}, config={"displaylogo": False, "responsive": True}),
                                    style={"flex": 1, "display": "flex", "width": "100%", "minWidth": 0},
                                    parent_style={"flex": 1, "display": "flex", "width": "100%"}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
                    children=[
                        html.Div(
                            style={"display": "flex", "flexDirection": "row", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Label("Right View:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(id="right-dropdown", style={"flex": 1}, clearable=True, placeholder="Select an artifact to compare...")
                            ]
                        ),
                        html.Div(
                            id = "right-graph-2d-container",
                            style = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0},
                            children = [
                                html.Div(id = "right-graph-2d-title", style = {"fontWeight": "bold", "textAlign": "center"}),
                                dcc.Loading(
                                    dcc.Graph(id="right-graph-2d", figure=_blank_figure(), style={"flex": 1, "minWidth": 0}, config={"displaylogo": False, "responsive": True}),
                                    style={"flex": 1, "display": "flex", "width": "100%", "minWidth": 0},
                                    parent_style={"flex": 1, "display": "flex", "width": "100%"}
                                )
                            ]
                        ),
                        html.Div(
                            id = "right-graph-3d-container",
                            style = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0},
                            children = [
                                html.Div(id = "right-graph-3d-title", style = {"fontWeight": "bold", "textAlign": "center"}), 
                                # "padding": "4px 0"                                   
                                dcc.Loading(
                                    dcc.Graph(id="right-graph-3d", figure=_blank_figure(), style={"flex": 1, "minWidth": 0}, config={"displaylogo": False, "responsive": True}),
                                    style={"flex": 1, "display": "flex", "width": "100%", "minWidth": 0},
                                    parent_style={"flex": 1, "display": "flex", "width": "100%"}
                                )
                            ]
                        )
                    ]
                )
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
                    html.H4("Preview", style={"margin": "0 0 8px 0"}),
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


@callback(
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


@callback(
    Output("left-graph-2d", "figure"),
    Output("left-graph-3d", "figure"),
    Output("left-graph-2d-container", "style"),
    Output("left-graph-3d-container", "style"),
    Output("left-graph-2d-title", "children"),
    Output("left-graph-3d-title", "children"),
    Output("left-state", "data"),
    Input("left-dropdown", "value"),
    Input("clear-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _update_left_graph(artifact_path, clear_clicks):
    visible = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0}
    hidden = {"display": "none"}
    if callback_context.triggered_id == "clear-reset" or not artifact_path:
        return _blank_figure(), _blank_figure(), visible, visible, "", "", None
    try:
        fig_2d, fig_3d, payload, title = _load_artifact(artifact_path)
        # determine each container's visibility based on if that specific figure has any traces
        style_2d = visible if len(fig_2d.data) > 0 else hidden
        style_3d = visible if len(fig_3d.data) > 0 else hidden
        title_2d = "2D " + title
        title_3d = "3D " + title
        return fig_2d, fig_3d, style_2d, style_3d, title_2d, title_3d, payload
    except Exception as e:
        er_fig = _blank_figure(f"Error loading artifact: {e}")
        return er_fig, er_fig, visible, visible, "", "", None


@callback(
    Output("right-graph-2d", "figure"),
    Output("right-graph-3d", "figure"),
    Output("right-graph-2d-container", "style"),
    Output("right-graph-3d-container", "style"),
    Output("right-graph-2d-title", "children"),
    Output("right-graph-3d-title", "children"),        
    Output("right-state", "data"),
    Input("right-dropdown", "value"),
    Input("clear-reset", "n_clicks"),
    prevent_initial_call=True   # wont run funciton when page first loads
)
def _update_right_graph(artifact_path, clear_clicks):
    visible = {"display": "flex", "flexDirection": "column", "flex": 1, "minHeight": "500px", "minWidth": 0}
    hidden = {"display": "none"}
    if callback_context.triggered_id == "clear-reset" or not artifact_path:
        mes = _blank_figure("Select a second artifact to compare.")
        return mes, mes, visible, visible, "", "", None
    try:
        fig_2d, fig_3d, payload, title = _load_artifact(artifact_path)
        style_2d = visible if len(fig_2d.data) > 0 else hidden
        style_3d = visible if len(fig_3d.data) > 0 else hidden
        title_2d = "2D " + title
        title_3d = "3D " + title
        return fig_2d, fig_3d, style_2d, style_3d, title_2d, title_3d, payload
    except Exception as e:
        er_fig = _blank_figure(f"Error loading artifact: {e}")
        return er_fig, er_fig, visible, visible, "", "",  None


@callback(
    Output("preview-message", "children"),
    Output("preview-image", "src"),
    Output("preview-image", "style"),
    Output("preview-meta", "children"),
    Input("left-graph-2d", "clickData"),
    Input("left-graph-3d", "clickData"),
    Input("right-graph-2d", "clickData"),
    Input("right-graph-3d", "clickData"),
    Input("clear-reset", "n_clicks"),
    State("left-state", "data"),
    State("right-state", "data"),
    prevent_initial_call=True
)
def _update_preview(left_click_2d, left_click_3d, right_click_2d, right_click_3d, 
                    clear_clicks, left_state, right_state):
    trigger = callback_context.triggered_id
    img_style_hidden = {"display": "none"}
    img_style_visible = {"height": "120px", "objectFit": "contain", "border": "1px solid #ddd", "backgroundColor": "white", "display": "block"}
    
    if trigger == "clear-reset":
        return initial_msg, "", img_style_hidden, ""
        
    if trigger == "left-graph-2d":
        click_data = left_click_2d
        state = left_state
    elif trigger == "left-graph-3d":
        click_data = left_click_3d
        state = left_state
    elif trigger == "right-graph-2d":
        click_data = right_click_2d
        state = right_state
    else:
        click_data = right_click_3d
        state = right_state
    
    if not state or not click_data:
        raise PreventUpdate

    point = click_data["points"][0]
    curve_num = int(point.get("curveNumber", -1))
    point_num = int(point.get("pointNumber", -1))

    layout = state.get("layout") or []
    if curve_num < 0 or curve_num >= len(layout):
        return "Could not resolve clicked point.", "", img_style_hidden, ""

    id_array = layout[curve_num]
    if id_array is None:
        return "No preview available for train points (test points only).", "", img_style_hidden, ""
    if point_num < 0 or point_num >= len(id_array):
        return "Could not resolve clicked point.", "", img_style_hidden, ""

    idx = int(id_array[point_num])
    artifact = load_dash_artifact(state["path"])
    if idx < 0 or idx >= len(artifact["y_test_sub"]):
        return "Could not resolve clicked point.", "", img_style_hidden, ""

    image_shape = artifact["image_shape"]
    X_test_pixels = artifact["X_test_pixels"]
    img_src = _image_to_data_url(X_test_pixels[idx], image_shape)

    meta = (
        f"Artifact: {Path(state['path']).name}\n"
        f"Test Index: {idx}\n"
        f"True Label: {int(artifact['y_test_sub'][idx])}\n"
        f"Predicted Label: {int(artifact['y_pred_sub'][idx])}"
    )

    return "Selected sample:", img_src, img_style_visible, meta