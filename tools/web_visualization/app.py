import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# parents[2] as app.py is two directories below IDAAMgen
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dash
from dash import Dash, html, Input, Output, callback
from tools.web_visualization.navbar import build_navbar

app = Dash(__name__, use_pages=True, pages_folder="pages")
app.title = "IDAAMgen Visualizations"

app.layout = html.Div(
    style = {},
    children = [
        html.Div(id="navbar-container", children=[build_navbar()]),
        html.Div(dash.page_container, style={"minHeight": "calc(100vh - 56px)"})
    ]
)

@callback(
    Output("navbar-container", "children"),
    Input("_pages_location", "pathname"),
)
def _update_navbar(pathname):
    return build_navbar(pathname)

def main():
    parser = argparse.ArgumentParser(description="Dash app for t-SNE artifacts and on-demand runs.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()