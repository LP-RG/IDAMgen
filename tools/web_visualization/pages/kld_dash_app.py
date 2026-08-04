import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
from dash import Dash, Input, Output, State, callback, callback_context, dcc, html

from tools.web_visualization.shared_figures import _blank_figure
from tools.tsne_visualization.tsne_sweep_figures import build_kld_sweep_figure
from tools.tsne_visualization.tsne_sweep_figures import build_step_scatter_figure
from tools.tsne_visualization.tsne_utils import load_sweep_manifest, load_sweep_step_artifact

dash.register_page(__name__, path="/kld-sweep", name="N_tot vs KL_div")

def _resolve_manifest_path(run_path_str):
    """Resolve user input (run directory or direct manifest path) to sweep_manifest.json"""
    p = Path(run_path_str)
    if not p.is_absolute(): # make path absolute if it isn't already
        p = (ROOT / p).resolve()
    if p.is_dir():  # appends json file name if user specified directory
        p = p / "sweep_manifest.json"
    if not p.exists():
        raise ValueError(f"No sweep_manifest.json found at: {p}")
    return p


def _build_slider_marks(rows):
    """Enable one mark per step, having red x over the steps without an associated saved artifact path"""
    marks = {}
    for i, row in enumerate(rows):
        if row.get("artifact_path") is None:
            marks[i] = {"label": row["n_total"],
                        "style": {"color": "#d62728", "fontWeight": "bold"}}
        else:
            marks[i] = str(row["n_total"])
    return marks


def _default_step_index(rows, reference_n=3000):
    """Compute index of step whose n_total is closest to reference_n"""
    if not rows:
        return 0
    return min(range(len(rows)), key=lambda i: abs(rows[i]["n_total"] - reference_n))


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
                html.H2("N_total vs KL Divergence — Point-Count Sweep", style={"margin": "0", "minWidth": "160px"}),
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "min-width": "600px"},
                    children=[
                        dcc.Input(id="run-path", type="text", style={"flex": 1, "padding": "8px"}, placeholder="Path to run directory (e.g. plots/pixels/resnet/sweep/run_2026...)"),
                        html.Button("Load Run", id="load-run", n_clicks=0, style={"padding": "8px 16px"}),
                        html.Button("Update", id="update-step", n_clicks=0, style={"padding": "8px 16px"})
                    ]
                )
            ]
        ),
        html.Div(id="kld-status-log"),
        # MAIN CURVE
        html.Div(
            style={
                "display": "flex", "flexDirection": "column", "gap": "8px",
                "backgroundColor": "white", "padding": "12px", "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=[
                dcc.Loading(
                    dcc.Graph(id="main-curve", figure=_blank_figure(), style={"minHeight": "800px", "minWidth": "1000px"}, config={"displaylogo": False, "responsive": True})
                )
            ]
        ),
        # SCATTERS SECTION
        html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "backgroundColor": "white", "padding": "12px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
            children=[
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "gap": "8px", "width":"65%"},
                    children=[
                        html.Label("Step:", style={"fontWeight": "bold", "minWidth": "50px"}),
                        html.Div(
                            style={"flex": 1, "minWidth": 0},
                            children=[dcc.Slider(id="step-slider", min=0, max=0, step=1, value=0, marks={})]
                        )
                    ]
                ),
                html.Div(
                    style={"display": "flex", "flexDirection": "row", "gap": "16px", "flex": 1},
                    children=[
                        # 2D GRAPH
                        html.Div(
                            id="2d-scatter-container",
                            style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px"},
                            children=[
                                dcc.Loading(
                                    dcc.Graph(id="2d-scatter-figure", figure=_blank_figure(), style={"minHeight": "800px", "minWidth": "750px"}, config={"displaylogo": False, "responsive": True})
                                )
                            ]
                        ),
                        # 3D GRAPH
                        html.Div(
                            id="3d-scatter-container",
                            style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px"},
                            children=[
                                dcc.Loading(
                                    dcc.Graph(id="3d-scatter-figure", figure=_blank_figure(), style={"minHeight": "800px", "minWidth": "750px"}, config={"displaylogo": False, "responsive": True})
                                )
                            ]
                        )
                    ]
                )
            ]   
        ),
        dcc.Store(id="sweep-rows")
    ]
)


@callback(
    Output("sweep-rows", "data"),
    Output("main-curve", "figure"),
    Output("step-slider", "min"),
    Output("step-slider", "max"),
    Output("step-slider", "marks"),
    Output("step-slider", "value"),    
    Output("kld-status-log", "children"),
    Input("load-run", "n_clicks"),
    Input("update-step", "n_clicks"),
    State("run-path", "value"),
    prevent_initial_call=True
)
def _load_or_update_run(load_clicks, update_clicks, run_path):
    trigger = callback_context.triggered_id
    no_change = dash.no_update

    if not run_path:
        return no_change, no_change, no_change, no_change, no_change, no_change, "Please provide a run directory path."

    try:
        manifest_path = _resolve_manifest_path(run_path)
        rows = sorted(load_sweep_manifest(manifest_path), key=lambda r: r["n_total"])
    except Exception as e:
        return no_change, no_change, no_change, no_change, no_change, no_change, f"Error: {e}"

    fig = build_kld_sweep_figure(rows)
    marks = _build_slider_marks(rows)
    # update refreshes data but leaves slider untouched
    slider_value = _default_step_index(rows) if trigger == "load-run" else no_change
    status = f"Loaded {len(rows)} steps from {manifest_path.name}"

    return rows, fig, 0, len(rows) - 1, marks, slider_value, status


@callback(
    Output("2d-scatter-figure", "figure"),
    Output("3d-scatter-figure", "figure"),
    Output("2d-scatter-container", "style"),
    Output("3d-scatter-container", "style"),
    Input("step-slider", "value"),
    State("sweep-rows", "data"),
    prevent_initial_call=True
)
def render_step_scatter(step_idx, rows):
    visible = {"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px"}
    hidden = {"display": "none"}

    if not rows or step_idx is None or step_idx >= len(rows):
        blank = _blank_figure("Load a run to see step scatters.")
        return blank, blank, visible, visible
    
    row = rows[step_idx]
    art_path = row.get("artifact_path")
    if art_path is None:
        blank = _blank_figure(f"No data was saved for this step at N_total={row['n_total']}.")
        return blank, blank, visible, visible
    
    data = load_sweep_step_artifact(art_path)
    y_all = data["y_all"]
    test_mask = data["test_mask"]
    title_base = f"N_train={row['n_train']}, N_total={row['n_total']}"

    if data["X_2d"] is not None and len(data["X_2d"]) > 0:
        fig_2d = build_step_scatter_figure(data["X_2d"], y_all, test_mask, f"{title_base} (2D)")
        style_2d = visible
    else:
        fig_2d = _blank_figure("No 2D coordinates for this step.")
        style_2d = hidden

    if data["X_3d"] is not None and len(data["X_3d"]) > 0:
        fig_3d = build_step_scatter_figure(data["X_3d"], y_all, test_mask, f"{title_base} (3D)")
        style_3d = visible
    else:
        fig_3d = _blank_figure("No 3D coordinates for this step.")
        style_3d = hidden
    
    return fig_2d, fig_3d, style_2d, style_3d