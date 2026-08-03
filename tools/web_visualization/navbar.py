from dash import dcc, html, page_registry

def build_navbar(pathname="/"):
    links = [
        dcc.Link(
            page["name"],
            href=page["relative_path"],
            style={
                "padding": "8px 16px",
                "color": "#E7E8EC" if pathname == page["relative_path"] else "#8A8D93",
                "textDecoration": "none",
                "fontSize": "14px",
                "fontWeight": "bold" if pathname == page["relative_path"] else "normal",
                "borderRadius": "6px"
            },
            className="nav-link"
        )
        # order by URL path so link order stays fixed
        for page in sorted(page_registry.values(), key=lambda p: p["relative_path"])
    ]

    github_icon = html.A(
        html.Img(src="/assets/github-mark.svg", style={"height": "20px", "width": "2    0px"}),
        href="https://github.com/LP-RG/IDAAMgen",
        target="_blank",
        rel="noopener noreferrer",
        style={"marginLeft": "auto", "display": "flex", "alignItems": "center"}
    )

    return html.Nav(
        style={
            "display": "flex", "flexDirection": "row", "alignItems": "center",
            "gap": "8px", "padding": "12px 16px", "backgroundColor": "#161616",
            "boxShadow": "0 6px 12px -2px rgba(0,0,0,0.4)",
            "font-family": "sans-serif"
        },
        children=[
            html.Span(
                "IDAAMgen Visualizations",
                style={"color": "#FFFFFF", "fontWeight": "bold", "marginRight": "24px", "fontSize": "22px"},
            ),
            *links,
            github_icon
        ],
    )