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