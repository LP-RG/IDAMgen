import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_loss_trace(fig, epochs, row=1, col=1):
    """Add training loss line to subpot figure."""
    x, y = [], []
    for e in epochs:
        x.append(e["epoch"])
        y.append(e["train_loss"])
    fig.add_trace(
        go.Scatter(x=x, y=y,
            mode="lines+markers",
            name="Train loss",
            legendgroup="loss"),
        row=row, col=col)
    fig.update_yaxes(title_text="Loss", row=row, col=col)
    return fig


def add_accuracy_traces(fig, epochs, row=2, col=1):
    """Add train and test (if present) accuracy lines to subpot figure."""
    x, y1, y2 = [], [], []
    for e in epochs:
        x.append(e["epoch"])
        y1.append(e["train_acc"])
        y2.append(e["test_acc"])
    fig.add_trace(
        go.Scatter(x=x, y=y1,
            mode="lines+markers",
            name="Train accuracy",
            legendgroup="train_acc"),
        row=row, col=col
    )
    trace = False
    for i in range(len(y1)):
        if y2[i] is not None:
            trace = True
    if trace:
        fig.add_trace(
            go.Scatter(x=x, y=y2,
                mode="lines+markers",
                name="Test accuracy",
                legendgroup="test_acc"),
            row=row, col=col
        )
    fig.update_yaxes(title_text="Accuracy", row=row, col=col)
    return fig


def build_loss_curve_figure(epochs, selected_epoch, stage_tag=None):
    """Build the dual Y-axis (loss, accuracy) figure."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    add_loss_trace(fig, epochs, row=1, col=1)
    add_accuracy_traces(fig, epochs, row=2, col=1)
    fig.add_vline(x=selected_epoch, row="all", col=1)
    
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(showline=True, linecolor="#888", row="all", col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(showline=True, linecolor="#888", row="all", col=1)
    
    fig.update_layout(title=stage_tag)
    return fig