"""
FraudLens AI - Visualization Module
=====================================
Plotly-based interactive charts for the analytics dashboard.
All figures use a consistent professional colour palette and
transparent backgrounds for seamless Streamlit embedding.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#48bb78",
    "danger": "#f56565",
    "warning": "#ed8936",
    "info": "#4299e1",
    "dark": "#1a202c",
    "light": "#f7fafc",
    "fraud": "#e53e3e",
    "legitimate": "#38a169",
    "gradient_start": "#667eea",
    "gradient_end": "#764ba2",
}

# Shared layout defaults applied to every figure
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color=COLORS["light"]),
    template="plotly_dark",
    margin=dict(l=40, r=40, t=60, b=40),
    hoverlabel=dict(
        bgcolor=COLORS["dark"],
        font_size=13,
        font_family="Inter, Segoe UI, sans-serif",
    ),
)


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared layout defaults and an optional title."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font_size=16))
    return fig


# ---------------------------------------------------------------------------
# 1.  Prediction distribution – donut chart
# ---------------------------------------------------------------------------
def plot_prediction_distribution(predictions: pd.Series) -> go.Figure:
    """Create a donut chart showing fraud vs legitimate distribution.

    Parameters
    ----------
    predictions : pd.Series
        Series of prediction labels (``0`` / ``1`` or ``"Fraud"`` / ``"Legitimate"``).

    Returns
    -------
    go.Figure
    """
    try:
        counts = predictions.value_counts()

        # Normalise labels
        label_map = {0: "Legitimate", 1: "Fraud", "Legitimate": "Legitimate", "Fraud": "Fraud"}
        labels = [label_map.get(idx, str(idx)) for idx in counts.index]
        values = counts.values.tolist()

        colours = [
            COLORS["fraud"] if lab == "Fraud" else COLORS["legitimate"]
            for lab in labels
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=colours, line=dict(color=COLORS["dark"], width=2)),
                    textinfo="label+percent",
                    textfont_size=13,
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                )
            ]
        )

        _apply_defaults(fig, "Prediction Distribution")
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        return fig

    except Exception as exc:
        logger.error("Prediction distribution plot failed: %s", exc)
        raise RuntimeError(f"Failed to create prediction distribution chart: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.  Amount distribution histogram
# ---------------------------------------------------------------------------
def plot_amount_distribution(
    df: pd.DataFrame,
    amount_col: str = "Amount",
) -> go.Figure:
    """Create a histogram of transaction amounts.

    If a ``Class`` column exists the histogram is split by fraud /
    legitimate with distinct colours.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data.
    amount_col : str
        Name of the amount column.

    Returns
    -------
    go.Figure
    """
    try:
        if amount_col not in df.columns:
            raise ValueError(f"Column '{amount_col}' not found in DataFrame.")

        has_class = "Class" in df.columns

        if has_class:
            plot_df = df[[amount_col, "Class"]].copy()
            plot_df["Label"] = plot_df["Class"].map({0: "Legitimate", 1: "Fraud"})
            colour_map = {"Legitimate": COLORS["legitimate"], "Fraud": COLORS["fraud"]}

            fig = px.histogram(
                plot_df,
                x=amount_col,
                color="Label",
                color_discrete_map=colour_map,
                nbins=60,
                barmode="overlay",
                opacity=0.75,
            )
        else:
            fig = px.histogram(
                df,
                x=amount_col,
                nbins=60,
                color_discrete_sequence=[COLORS["primary"]],
                opacity=0.80,
            )

        _apply_defaults(fig, "Transaction Amount Distribution")
        fig.update_xaxes(title_text="Amount ($)", gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title_text="Count", gridcolor="rgba(255,255,255,0.08)")
        return fig

    except Exception as exc:
        logger.error("Amount distribution plot failed: %s", exc)
        raise RuntimeError(f"Failed to create amount distribution chart: {exc}") from exc


# ---------------------------------------------------------------------------
# 3.  Fraud probability histogram
# ---------------------------------------------------------------------------
def plot_probability_distribution(probabilities: pd.Series) -> go.Figure:
    """Create a histogram of fraud probabilities.

    Parameters
    ----------
    probabilities : pd.Series
        Fraud probabilities in [0, 1].

    Returns
    -------
    go.Figure
    """
    try:
        fig = go.Figure(
            data=[
                go.Histogram(
                    x=probabilities,
                    nbinsx=50,
                    marker=dict(
                        color=COLORS["primary"],
                        line=dict(color=COLORS["secondary"], width=1),
                    ),
                    opacity=0.85,
                    hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>",
                )
            ]
        )

        # Add vertical threshold lines
        for threshold, colour, label in [
            (0.3, COLORS["success"], "Low / Medium"),
            (0.7, COLORS["danger"], "Medium / High"),
        ]:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=colour,
                annotation_text=label,
                annotation_position="top",
                annotation_font_color=colour,
            )

        _apply_defaults(fig, "Fraud Probability Distribution")
        fig.update_xaxes(title_text="Fraud Probability", gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title_text="Count", gridcolor="rgba(255,255,255,0.08)")
        return fig

    except Exception as exc:
        logger.error("Probability distribution plot failed: %s", exc)
        raise RuntimeError(f"Failed to create probability distribution chart: {exc}") from exc


# ---------------------------------------------------------------------------
# 4.  Model comparison – grouped bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(metrics: dict) -> go.Figure:
    """Create a grouped bar chart comparing model metrics.

    Parameters
    ----------
    metrics : dict
        ``{model_name: {metric_name: value, …}, …}``

    Returns
    -------
    go.Figure
    """
    try:
        model_names = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())

        bar_colours = [COLORS["primary"], COLORS["warning"], COLORS["success"]]

        fig = go.Figure()
        for i, metric in enumerate(metric_names):
            values = [metrics[m][metric] for m in model_names]
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=model_names,
                    y=values,
                    marker_color=bar_colours[i % len(bar_colours)],
                    text=[f"{v:.2f}" for v in values],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.2f}<extra></extra>",
                )
            )

        _apply_defaults(fig, "Model Performance Comparison")
        fig.update_layout(
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            yaxis=dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.08)"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        return fig

    except Exception as exc:
        logger.error("Model comparison plot failed: %s", exc)
        raise RuntimeError(f"Failed to create model comparison chart: {exc}") from exc


# ---------------------------------------------------------------------------
# 5.  Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix_heatmap(
    cm_data: Optional[list] = None,
) -> go.Figure:
    """Create an interactive confusion matrix heatmap.

    Parameters
    ----------
    cm_data : list[list[int]], optional
        2×2 confusion matrix. Defaults to pre-computed Random Forest results.

    Returns
    -------
    go.Figure
    """
    try:
        if cm_data is None:
            # Pre-computed Random Forest confusion matrix from training notebook
            cm_data = [[56840, 24], [24, 74]]

        cm = np.array(cm_data)
        labels = ["Legitimate", "Fraud"]

        # Build annotation text with counts and percentages
        total = cm.sum()
        text_matrix = [
            [f"{cm[i][j]}<br>({cm[i][j] / total * 100:.2f}%)" for j in range(2)]
            for i in range(2)
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred: {l}" for l in labels],
                y=[f"Actual: {l}" for l in labels],
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=14),
                colorscale=[
                    [0, COLORS["dark"]],
                    [0.5, COLORS["primary"]],
                    [1, COLORS["danger"]],
                ],
                hovertemplate=(
                    "<b>%{y} → %{x}</b><br>"
                    "Count: %{z}<extra></extra>"
                ),
                showscale=True,
                colorbar=dict(title="Count"),
            )
        )

        _apply_defaults(fig, "Confusion Matrix – Random Forest")
        fig.update_layout(
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="Actual Label", autorange="reversed"),
        )
        return fig

    except Exception as exc:
        logger.error("Confusion matrix plot failed: %s", exc)
        raise RuntimeError(f"Failed to create confusion matrix heatmap: {exc}") from exc


# ---------------------------------------------------------------------------
# 6.  Fraud rate by amount range
# ---------------------------------------------------------------------------
def plot_fraud_by_amount_range(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing fraud rate by transaction-amount ranges.

    Requires ``Amount`` and ``Class`` columns (or ``Prediction``).

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with ``Amount`` and a class/prediction column.

    Returns
    -------
    go.Figure
    """
    try:
        if "Amount" not in df.columns:
            raise ValueError("DataFrame must contain an 'Amount' column.")

        # Determine the target column
        target_col = "Class" if "Class" in df.columns else "Prediction"
        if target_col not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'Class' or 'Prediction' column."
            )

        bins = [0, 10, 50, 100, 250, 500, 1000, 5000, float("inf")]
        labels = ["$0-10", "$10-50", "$50-100", "$100-250", "$250-500", "$500-1K", "$1K-5K", "$5K+"]

        plot_df = df[["Amount", target_col]].copy()
        plot_df["Amount_Range"] = pd.cut(
            plot_df["Amount"], bins=bins, labels=labels, right=False
        )

        grouped = (
            plot_df.groupby("Amount_Range", observed=False)
            .agg(
                Total=(target_col, "count"),
                Fraud=(target_col, "sum"),
            )
            .reset_index()
        )
        grouped["Fraud_Rate"] = (grouped["Fraud"] / grouped["Total"].replace(0, np.nan) * 100).fillna(0)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=grouped["Amount_Range"].astype(str),
                    y=grouped["Fraud_Rate"],
                    marker=dict(
                        color=grouped["Fraud_Rate"],
                        colorscale=[[0, COLORS["success"]], [1, COLORS["danger"]]],
                        line=dict(color=COLORS["dark"], width=1),
                    ),
                    text=[f"{r:.1f}%" for r in grouped["Fraud_Rate"]],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Fraud Rate: %{y:.2f}%<br>"
                        "Total Txns: %{customdata[0]}<br>"
                        "Fraud Txns: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=grouped[["Total", "Fraud"]].values,
                )
            ]
        )

        _apply_defaults(fig, "Fraud Rate by Transaction Amount")
        fig.update_xaxes(title_text="Amount Range", gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title_text="Fraud Rate (%)", gridcolor="rgba(255,255,255,0.08)")
        return fig

    except Exception as exc:
        logger.error("Fraud-by-amount plot failed: %s", exc)
        raise RuntimeError(f"Failed to create fraud-by-amount chart: {exc}") from exc


# ---------------------------------------------------------------------------
# 7.  KPI metric helper
# ---------------------------------------------------------------------------
def create_kpi_metric(
    value,
    label: str,
    delta: Optional[str] = None,
    color: str = "primary",
) -> dict:
    """Build a KPI data dictionary for rendering in Streamlit.

    Parameters
    ----------
    value : int | float | str
        The main metric value.
    label : str
        Short description shown above the value.
    delta : str, optional
        Change indicator (e.g. "+12%").
    color : str
        Key into ``COLORS`` palette (default ``"primary"``).

    Returns
    -------
    dict
        Keys: ``value``, ``label``, ``delta``, ``color``, ``color_hex``.
    """
    resolved_color = COLORS.get(color, COLORS["primary"])

    return {
        "value": value,
        "label": label,
        "delta": delta,
        "color": color,
        "color_hex": resolved_color,
    }