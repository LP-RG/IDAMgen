import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import plotly.graph_objects as go
import dash
from dash import Dash, Input, Output, State, callback, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from tools.tsne_visualization.tsne_sweep_figures import build_step_scatter_figure
from tools.train_visualization.train_utils import (
    load_epoch_artifact, load_train_manifest, get_artifact_path, get_max_epoch)
from tools.train_visualization.train_figures import build_loss_curve_figure
from tools.web_visualization.shared_figures import _blank_figure


dash.register_page(__name__, path="/train-history", name="Train History")


layout = html.Div(
    style={
        "display": "flex", "flexDirection": "column", "gap": "16px",
        "padding": "16px", "minHeight": "100vh", "boxSizing": "border-box",
        "backgroundColor": "#f4f4f9", "fontFamily": "sans-serif"
    },
    children=[
        # HEADER SECTION
        html.Div(
            style={
                "display": "flex", "flexDirection": "column", "gap": "12px",
                "alignItems": "center", "backgroundColor": "white", 
                "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=[
                html.H2("Training History Visualization", style={"margin": "0", "minWidth": "160px"}),
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "min-width": "600px"},
                    children=[
                        dcc.Input(id="run-path", type="text", style={"flex": 1, "padding": "8px"}, placeholder="Path to run directory (e.g. train_visualization/dashboard_data)"),
                        html.Button("Load", id="load-button", n_clicks=0, style={"padding": "8px 16px"})
                    ]
                )
            ]
        ),
        # LOSS/ACCURACY PLOT
        html.Div(
            style={
                "display": "flex", "flexDirection": "column", "gap": "8px",
                "backgroundColor": "white", "padding": "12px", "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=[
                dcc.Loading(
                    dcc.Graph(id="main-plot", figure=_blank_figure(), style={"minHeight": "500px", "minWidth": "700px"}, config={"displaylogo": False, "responsive": True})
                )
            ]
        ),
        # MENU & SCATTER SECTION
        html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
            children=[
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "gap": "8px", "width":"65%"},
                    children=[
                        html.Label("Stage:", style={"fontWeight": "bold", "minWidth": "50px"}),
                        dcc.RadioItems(
                            options=[
                                {"label": "Exact", "value": "exact"},
                                {"label": "Quant", "value": "quant_q8"},
                                {"label": "Approx", "value": "approx_a8_my_table"}
                            ],
                            value="exact",
                            id="stage-selector",
                            labelStyle={"display": "inline-block", "marginRight": "16px"}
                        ),
                        html.Label("Epoch:", style={"fontWeight": "bold", "minWidth": "50px"}),
                        html.Div(
                            style={"flex": 1, "minWidth": 0},
                            children=[dcc.Slider(id="epoch-slider", min=0, max=0, step=1, value=0, marks={})]
                        )
                    ]
                ),
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "gap": "16px", "flex": 1},
                    children=[
                        # 3D GRAPH
                        html.Div(
                            style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px"},
                            children=[
                                dcc.Loading(
                                    dcc.Graph(id="train-scatter-figure", figure=_blank_figure(), style={"minHeight": "1000px", "minWidth": "900px", "width": "100%"}, config={"displaylogo": False, "responsive": True})
                                )
                            ]
                        )
                    ]
                )
            ]   
        ),
        dcc.Store(id="manifest-store"),
        dcc.Store(id="epochs-store")
    ]
)


@callback(
    Output("manifest-store", "data"),
    Output("epochs-store", "data"),
    Output("epoch-slider", "min"),
    Output("epoch-slider", "max"),
    Output("epoch-slider", "value"),
    Input("load-button", "n_clicks"),
    Input("stage-selector", "value"),
    State("run-path", "value"),
    prevent_initial_call=True
)
def _load_run(load_clicks, stage_value, run_path):
    """Load a stage's manifest and parsed-log JSON and reset the epoch slider to its range."""
    no_change = dash.no_update

    if not run_path:
        return no_change, no_change, no_change, no_change, no_change

    manifest_path = os.path.join(run_path, "train_manifest.json")
    manifest = load_train_manifest(manifest_path)
    
    log_path = os.path.join(run_path, f"{stage_value}.json")
    with open(log_path) as f:
        data = json.load(f)
    epochs = data["epochs"]
    
    max_epoch = get_max_epoch(manifest, stage_value)
    slider_value = 1

    return manifest, epochs, 1, max_epoch, slider_value


@callback(
    Output("train-scatter-figure", "figure"),
    Output("main-plot", "figure"),
    Input("stage-selector", "value"),
    Input("epoch-slider", "value"),
    State("manifest-store", "data"),
    State("epochs-store", "data"),
    prevent_initial_call=True,
)
def _render_train_scatter(stage_value, slider_value, manifest, epochs):
    """Render the loss/accuracy curve and the 3D embedding scatter for the selected stage and epoch."""

    curve_fig = build_loss_curve_figure(epochs, slider_value, stage_value)
    artifact_path = get_artifact_path(manifest, stage_value, slider_value)
    art = load_epoch_artifact(artifact_path)

    scatter_fig = build_step_scatter_figure(
        coords=art["X_3d"],
        y=art["y"],
        test_mask=None,
        title=f"{stage_value} stage, epoch {slider_value} visualization",
        y_pred=art["y_pred"])
    
    return scatter_fig, curve_fig