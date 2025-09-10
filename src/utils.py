"""Utility funcitons module."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_history(history):
    """Funtion to visualize the training results of a model.

    Parameters
    ----------
    history : history object from model.fit()
        The training history.
    """
    hist = history.history

    # Create figure
    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Loss", "Accuracy"))

    # Add loss
    if "loss" in hist:
        fig.add_trace(go.Scatter(y=hist["loss"],
                                 mode="lines",
                                 name="Training - Loss"),
                      row=1,
                      col=1)
    if "val_loss" in hist:
        fig.add_trace(go.Scatter(y=hist["val_loss"],
                                 mode="lines",
                                 name="Validation - Loss"),
                      row=1,
                      col=1)

    # Add accuracy
    if "accuracy" in hist:
        fig.add_trace(go.Scatter(y=hist["accuracy"],
                                 mode="lines",
                                 name="Training - Accuracy"),
                      row=2,
                      col=1)
    if "val_accuracy" in hist:
        fig.add_trace(go.Scatter(y=hist["val_accuracy"],
                                 mode="lines",
                                 name="Validation - Accuracy"),
                      row=2,
                      col=1)

    # Layout configuration
    fig.update_layout(title="Training history",
                      xaxis_title="Epoch",
                      yaxis_title="Value",
                      hovermode="x unified",
                      height=1200)

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)

    fig.show()
