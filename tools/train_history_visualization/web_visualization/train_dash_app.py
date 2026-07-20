import os
import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from pathlib import Path

parsed_logs_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parsed_logs"
)

runs = {}
# .glob("*.json") searches within that directory and gives Path objects for each match
for path in Path(parsed_logs_dir).glob("*.json"):
    stage_name = path.stem  # directly gives filename without extension
    with open(path) as f:
        runs[stage_name] = json.load(f)

palette = {
    "1": "#03a1fc",    # exact
    "2": "#fca903",    # quant
    "3": "#f50707"     # approx
}

fig = go.Figure()

inset_epochs = []
inset_losses = []

for stage_name, parsed in runs.items():
    epochs = [e["epoch"] for e in parsed["epochs"]]
    losses = [e["train_loss"] for e in parsed["epochs"]]
    conv_type = parsed["metadata"].get("conv_type")
    color = palette.get(conv_type, "#919191")

    fig.add_trace(go.Scatter(
        x=epochs, y=losses,
        mode="lines+markers",
        marker=dict(color=color),
        line=dict(color=color),
        name=stage_name,
        legendgroup=stage_name
    ))

    # insert graph axes addition by defining a secondary set of domain-based coordinates
    if conv_type in ("2", "3"):
        fig.add_trace(go.Scatter(
            x=epochs, y=losses,
            mode="lines+markers",
            marker=dict(color=color),
            line=dict(color=color),
            xaxis="x2", yaxis="y2",
            legendgroup=stage_name,
            showlegend=False
        ))
        inset_epochs.extend(epochs)
        inset_losses.extend(losses)

zoom_x0, zoom_x1 = min(inset_epochs), max(inset_epochs)
zoom_y0, zoom_y1 = min(inset_losses), max(inset_losses)        

fig.update_layout(
    title="Training loss across stages"
    height=700,
    xaxis=dict(title="Epoch", color="#000000", anchor="x", gridcolor= "white", gridwidth=1),
    yaxis=dict(title="Train Loss (log scale)", type="log", dtick=1, color="#000000", anchor="y", gridcolor= "white", gridwidth=1, minor=dict(showgrid=True, ticks="outside")),
    # add inset graph independent axes with respective border rectangle
    xaxis2=dict(domain=[0.55, 0.95], anchor="y2"),
    yaxis2=dict(domain=[0.55, 0.95], anchor="x2", type="log"),
    shapes=[dict(type="rect", xref="paper", yref="paper", x0=0.55, x1=0.95, y0=0.55, y1=0.95, line=dict(color="black", width=1))],
    showlegend=True
)
fig.update_xaxes(showline=True, linewidth=1, linecolor='black', ticks="inside", nticks=10)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', ticks="inside", nticks=10)

app = Dash(__name__)
app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "gap": "16px", "padding": "16px", "minHeight": "100vh", "boxSizing": "border-box", "backgroundColor": "#f4f4f9", "fontFamily": "sans-serif"},
    children=[
        # TOP ROW
        html.Div(
            style={
                "display": "flex", "flexDirection": "row", "gap": "12px",
                "alignItems": "center", "backgroundColor": "white",
                "padding": "16px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            children=[
                html.H3("Training History", style={"margin": "0", "minWidth": "160px"}),
            ]
        ),
        # GRAPH ROW
        html.Div(
            style={"display": "flex", "flexDirection": "row", "gap": "16px", "flex": 1, "minHeight": "600px"},
            children=[
                html.Div(
                    style={"flex": 1, "display": "flex", "flexDirection": "column", "gap": "8px",
                           "backgroundColor": "white", "padding": "12px", "borderRadius": "8px",
                           "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "minWidth": 0},
                    children=[
                        html.Div(
                            id="loss-graph-container",
                            style={"display": "flex", "flex": 1},
                            children=[dcc.Loading(dcc.Graph(
                                id="loss-graph", figure=fig,
                                style={"flex": 1}, config={"displaylogo": False}
                            ))]
                        )
                    ]
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)



"""
The graph still looks really bad.
//- get rid of inset axes labels
//- make main graph axes visible
- make graph bg lines more uniform
- keep logarithmic ticks on Yaxis but also add in more regular ones
- slightly change inset graph bg color
- inset graph Yaxis ticks
- dashed rect around original section that is "enlarged"
"""    