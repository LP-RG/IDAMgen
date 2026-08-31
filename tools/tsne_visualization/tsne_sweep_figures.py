import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import math
import statistics

# configuration table - handing 2D and 3D in a single block of code
_DIM_STYLE = {
    "2d": {"label": "2D", "kl_field": "kl_div_2d", "median_field": "median_kl_2d", ""
           "color": "#7fb8e0", "line_color": "#1f5fa0", "offset_sign": -1},
    "3d": {"label": "3D", "kl_field": "kl_div_3d", "median_field": "median_kl_3d",
           "color": "#f2c572", "line_color": "#c9740a", "offset_sign": 1},
}


def build_kld_sweep_figure(rows, reference_n=3000, width_frac=0.15, offset_frac=0.03):
    fig = go.Figure()
    all_x = []
    for dim_key, style in _DIM_STYLE.items():   # runs twice - once per dimentionality
        dim_rows = [r for r in rows if r.get(style["kl_field"])]    # filters and drops any step for which dimensionality wasn't computed
        if not dim_rows:
            continue

        legendgroup = style["label"]
        line_x, line_y = [], []

        for row in sorted(dim_rows, key=lambda r: r["n_total"]):
            # sort matters - line_x/line_y are built by appending in loop order. If steps arrived out of sequence, 
            # the eventual median line would zig-zag left-and-right instead of tracing a clean trend.
            n_total = row["n_total"]
            values = row[style["kl_field"]]
            median = row.get(style["median_field"])
            if median is None:
                median = statistics.median(values)

            x_pos = n_total * (1 + style["offset_sign"] * offset_frac)
            all_x.append(x_pos)
            width = x_pos * width_frac

            fig.add_trace(go.Box(
                # plotly box trace expects x,y arrays of matching length
                x=[x_pos] * len(values),
                y=values,
                legendgroup=legendgroup,
                name=style["label"],
                marker_color=style["color"],
                boxpoints="all",    # draws all repeats as dot instead of default outliers only
                jitter=0.4,     # horizontal scatter so points don't eclipse each other
                showlegend=False    # would otherwise result in a legend "2D" entry per step
            ))

            line_x.append(x_pos)
            line_y.append(median)

        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines+markers",
            name=f"{style['label']} median",
            legendgroup=legendgroup,
            showlegend=True,
            line=dict(color=style["line_color"], width=3),
            marker=dict(size=8, color=style["line_color"])
        ))

    fig.add_vline(
        x=reference_n,
        line_dash="dot",
        line_color="#666666",
        annotation_text=f"current production N ({reference_n})",
        annotation_position="top right",
    )

    if all_x:
        log_min, log_max = math.log10(min(all_x)), math.log10(max(all_x))
        pad = max((log_max - log_min) * 0.2, 0.3)   # floor of 0.3 decades so a single-step sweep still gets breathing room
        x_range = [log_min - pad, log_max + pad]
    else:
        x_range = None

    fig.update_xaxes(type="log", title="N_total (points rendered)", range=x_range)
    fig.update_yaxes(title="KL divergence")

    fig.update_layout(
        title="t-SNE sweep: KL divergence vs. N_total",
        template="plotly_white",
        boxmode="overlay",
        legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=750,
        autosize=True
    )
    return fig


def build_step_scatter_figure(coords, y, test_mask=None, title=None, y_pred=None, height=700):
    """Draw one labeled 2D/3D scatter for a single sweep step or training epoch"""
    
    if test_mask is not None and y_pred is not None:
        raise ValueError(
            "build_step_scatter_figure: test_mask and y_pred cannot both be provided "
            "(the two aren't currently reconciled with each other in this function)."
        )
    
    coords = np.asarray(coords)
    y_all = np.asarray(y)
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
    is_2d = coords.shape[1] == 2

    unique_labels = np.unique(y_all)
    palette = np.array(sns.color_palette("hls", len(unique_labels)))
    label_to_color = {
        int(lab): f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        for lab, rgb in zip(unique_labels, palette)
    }

    fig = go.Figure()

    if y_pred is not None:
        correct_mask = (y_all == y_pred)
        wrong_mask = ~correct_mask
        misclass = coords[wrong_mask]
        wrong_indices = np.where(wrong_mask)[0]

    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        train_coords = coords[~test_mask]
        coords = coords[test_mask]
        y_all = y_all[test_mask]

    # train trace
        if len(train_coords) > 0:
            marker = dict(color="#cccccc", size=4, opacity=0.6)
            if is_2d:
                fig.add_trace(go.Scatter(
                    x=train_coords[:, 0], y=train_coords[:, 1],
                    mode="markers", name="Train (NE)", marker=marker
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=train_coords[:, 0], y=train_coords[:, 1], z=train_coords[:, 2],
                    mode="markers", name="Train (NE)", marker=marker
                ))

    # test trace
    for lab in unique_labels:
        pts = coords[y_all == lab]
        if len(pts) == 0:
            continue
        marker = dict(color=label_to_color[int(lab)], size=5)
        if is_2d:
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="markers", name="Test label" + str(lab), marker=marker))
        else:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", name="Test label" + str(lab), marker=marker))

    # misclassified trace
    if y_pred is not None:
        if is_2d:
            marker=dict(symbol="x", size=14, color="red", line=dict(width=2, color="red"))
            fig.add_trace(go.Scatter(
                x=misclass[:, 0], y=misclass[:, 1],
                mode="markers",
                name=f"Misclassified ({len(wrong_indices)})",
                marker=marker,
                customdata=wrong_indices,
                hovertemplate=("x=%{x:.2f}<br>y=%{y:.2f}<br>"
                    "true=%{text}<br>pred=%{meta}<extra></extra>"),
                text=y_all[wrong_mask],
                meta=y_pred[wrong_mask]
            ))
        else:
            marker=dict(symbol="x", size=10, color="red")
            fig.add_trace(go.Scatter3d(
                x=misclass[:, 0], y=misclass[:, 1], z=misclass[:, 2],
                mode="markers", 
                name=f"Misclassified ({len(wrong_indices)})", 
                marker=marker,
                customdata=wrong_indices,
                hovertemplate=("x=%{x:.2f}<br>y=%{y:.2f}<br>"
                    "true=%{text}<br>pred=%{meta}<extra></extra>"),
                text=y_all[wrong_mask],
                meta=y_pred[wrong_mask]
            ))

    camera = dict(eye=dict(x=1.5, y=1.5, z=1.2), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))

    fig.update_layout(title=title, 
        legend_title_text="class",
        margin=dict(l=20, r=20, t=40, b=20), 
        height=height,
        # if figure gets re-rendered keep whatever camera state the user last set by hand
        uirevision="scatter-view",
    )

    if not is_2d:
        fig.update_layout(scene=dict(camera=camera, aspectmode="data"))
    
    # cluster labels
    d = 1  # halo thickness in pixels
    offsets = [(-d, -d), (-d, 0), (-d, d), ( 0, -d), ( 0, d), ( d, -d), ( d, 0), ( d, d)]

    scene_annotations = []
    for lab in unique_labels:
        if is_2d:
            pts = coords[y_all == lab]
            if len(pts) == 0:
                continue
            xt, yt = np.median(pts, axis=0)
            for ox, oy in offsets:
                fig.add_annotation(
                    x=float(xt), y=float(yt), xshift=ox, yshift=oy,
                    text=str(int(lab)), showarrow=False,
                    font=dict(size=20, color="white"),
                )
            fig.add_annotation(
                x=float(xt), y=float(yt),
                text=str(int(lab)), showarrow=False,
                font=dict(size=16, color="black"),
            )
        else:        
            pts = coords[y_all == lab]
            if len(pts) == 0:
                continue
            xt, yt, zt = np.median(pts, axis=0)
            for i in range(len(offsets)):
                scene_annotations.append(dict(
                    x=float(xt), y=float(yt), z=float(zt),
                    xshift = offsets[i][0], yshift = offsets[i][1],
                    text=str(int(lab)),
                    showarrow=False,
                    font=dict(size=20, color="white")
                ))
            scene_annotations.append(dict(
                x=float(xt), y=float(yt), z=float(zt),
                text=str(int(lab)), showarrow=False,
                font=dict(size=16, color="black")
            ))
            fig.update_layout(scene=dict(annotations=scene_annotations))

    return fig