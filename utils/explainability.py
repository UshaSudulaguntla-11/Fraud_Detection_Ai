"""
FraudLens AI - Explainability Module
======================================
Provides SHAP-based model explanations including summary plots,
bar plots, waterfall plots, feature-importance ranking, and
single-transaction "why was this predicted" breakdowns.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – must precede pyplot import
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explainer creation
# ---------------------------------------------------------------------------
def get_shap_explainer(model) -> shap.TreeExplainer:
    """Create a SHAP ``TreeExplainer`` for a tree-based model.

    Parameters
    ----------
    model : tree-based sklearn / XGBoost / LightGBM estimator
        The trained model to explain.

    Returns
    -------
    shap.TreeExplainer
    """
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer created successfully.")
        return explainer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create SHAP TreeExplainer: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------
def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for the given input data.

    For binary classifiers that return a list of arrays (one per class),
    the values for the **fraud class (index 1)** are returned.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Pre-built explainer.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        SHAP values with shape ``(n_samples, n_features)``.
    """
    try:
        shap_values = explainer.shap_values(X)

        # Multi-class / binary list output → pick fraud class (index 1)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                # Fallback: use the last class
                shap_values = shap_values[-1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            if shap_values.shape[2] == 2:
                # Binary classification in 3D array format (n_samples, n_features, 2)
                shap_values = shap_values[:, :, 1]
            else:
                # Fallback: use the last class slice
                shap_values = shap_values[:, :, -1]

        shap_values = np.asarray(shap_values)
        logger.info("SHAP values computed – shape %s", shap_values.shape)
        return shap_values

    except Exception as exc:
        logger.error("SHAP value computation failed: %s", exc)
        raise RuntimeError(f"Failed to compute SHAP values: {exc}") from exc


# ---------------------------------------------------------------------------
# Summary dot plot
# ---------------------------------------------------------------------------
def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
) -> plt.Figure:
    """Generate a SHAP summary (dot / beeswarm) plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array of shape ``(n_samples, n_features)``.
    X : pd.DataFrame
        The feature matrix used for colouring.
    max_display : int, optional
        Maximum number of features to display (default 15).

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title("SHAP Feature Impact Summary", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        return fig
    except Exception as exc:
        logger.error("SHAP summary plot failed: %s", exc)
        plt.close("all")
        raise RuntimeError(f"Failed to create SHAP summary plot: {exc}") from exc


# ---------------------------------------------------------------------------
# Bar plot (mean |SHAP|)
# ---------------------------------------------------------------------------
def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
) -> plt.Figure:
    """Generate a SHAP mean-absolute-value bar plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array.
    X : pd.DataFrame
        Feature matrix (used for feature names).
    max_display : int, optional
        Maximum features to show (default 15).

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title(
            "SHAP Feature Importance (Mean |SHAP|)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        return fig
    except Exception as exc:
        logger.error("SHAP bar plot failed: %s", exc)
        plt.close("all")
        raise RuntimeError(f"Failed to create SHAP bar plot: {exc}") from exc


# ---------------------------------------------------------------------------
# Waterfall plot (single prediction)
# ---------------------------------------------------------------------------
def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_single: pd.DataFrame,
    index: int = 0,
) -> plt.Figure:
    """Generate a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        The explainer (needed for expected / base value).
    shap_values : np.ndarray
        SHAP values – may be for one sample or many; *index* selects the row.
    X_single : pd.DataFrame
        Feature data used for display (at least ``index + 1`` rows).
    index : int, optional
        Row index in *shap_values* / *X_single* to explain (default 0).

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        # Determine the base (expected) value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            # Binary classification → fraud class
            base_value = base_value[1] if len(base_value) == 2 else base_value[-1]
        base_value = float(base_value)

        # Select the correct row of SHAP values
        if shap_values.ndim == 2:
            sv_row = shap_values[index]
        else:
            sv_row = shap_values

        feature_names = list(X_single.columns)
        data_row = X_single.iloc[index].values if len(X_single) > index else X_single.iloc[0].values

        explanation = shap.Explanation(
            values=sv_row,
            base_values=base_value,
            data=data_row,
            feature_names=feature_names,
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title(
            "SHAP Waterfall – Individual Prediction Breakdown",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        return fig

    except Exception as exc:
        logger.error("SHAP waterfall plot failed: %s", exc)
        plt.close("all")
        raise RuntimeError(f"Failed to create SHAP waterfall plot: {exc}") from exc


# ---------------------------------------------------------------------------
# Top-N important features
# ---------------------------------------------------------------------------
def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape ``(n_samples, n_features)``.
    feature_names : list[str]
        Corresponding feature names.
    top_n : int, optional
        Number of top features to return (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: ``Feature``, ``Importance`` – sorted descending.
    """
    try:
        mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": mean_abs}
        )
        importance_df = (
            importance_df.sort_values("Importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        logger.info("Top %d features computed.", top_n)
        return importance_df

    except Exception as exc:
        logger.error("Feature importance ranking failed: %s", exc)
        raise RuntimeError(
            f"Failed to compute top features: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Single-transaction explanation ("Why was this prediction made?")
# ---------------------------------------------------------------------------
def explain_single_transaction(
    model,
    input_df: pd.DataFrame,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, "shap.Explanation"]:
    """Compute a fresh SHAP explanation for one transaction.

    This builds a brand-new ``TreeExplainer`` and computes SHAP values for
    exactly the single row supplied (no cached/sampled background data),
    so the explanation always reflects the transaction the user just
    entered. Intended for the "flagship feature" shown directly under the
    prediction result on the Single Transaction page.

    Parameters
    ----------
    model : tree-based estimator
        The trained fraud-detection model.
    input_df : pd.DataFrame
        A single-row DataFrame (already in the model's expected column
        order) representing the transaction to explain.
    top_n : int, optional
        Number of top contributing features to return (default 5).

    Returns
    -------
    contributions_df : pd.DataFrame
        Columns: ``Feature``, ``SHAP_Value``, ``Contribution_Pct``,
        ``Direction`` (``"Toward Fraud"`` / ``"Toward Legitimate"``) –
        sorted by absolute impact, descending, limited to *top_n* rows.
    explanation : shap.Explanation
        The full SHAP explanation object for the single row, suitable
        for passing straight into ``shap.plots.waterfall``.
    """
    try:
        explainer = get_shap_explainer(model)
        shap_values = compute_shap_values(explainer, input_df)

        # compute_shap_values always returns a 2D array; take the one row.
        sv_row = shap_values[0] if shap_values.ndim == 2 else shap_values

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) == 2 else base_value[-1]
        base_value = float(base_value)

        feature_names = list(input_df.columns)
        data_row = input_df.iloc[0].values

        explanation = shap.Explanation(
            values=sv_row,
            base_values=base_value,
            data=data_row,
            feature_names=feature_names,
        )

        # Rank by absolute SHAP magnitude, keep signed value for direction.
        total_abs = np.abs(sv_row).sum()
        contributions_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP_Value": sv_row,
            }
        )
        contributions_df["Contribution_Pct"] = (
            (contributions_df["SHAP_Value"].abs() / total_abs * 100)
            if total_abs > 0
            else 0.0
        )
        contributions_df["Direction"] = np.where(
            contributions_df["SHAP_Value"] >= 0, "Toward Fraud", "Toward Legitimate"
        )
        contributions_df = (
            contributions_df.reindex(
                contributions_df["SHAP_Value"].abs().sort_values(ascending=False).index
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        logger.info(
            "Single-transaction SHAP explanation computed (top %d features).", top_n
        )
        return contributions_df, explanation

    except Exception as exc:
        logger.error("Single-transaction SHAP explanation failed: %s", exc)
        raise RuntimeError(
            f"Failed to explain single transaction: {exc}"
        ) from exc


def plot_shap_waterfall_from_explanation(
    explanation: "shap.Explanation",
) -> plt.Figure:
    """Render a SHAP waterfall plot directly from a single-row Explanation.

    Companion to :func:`explain_single_transaction` – avoids recomputing
    SHAP values just to draw the waterfall for the same transaction.

    Parameters
    ----------
    explanation : shap.Explanation
        A single-row explanation as returned by
        :func:`explain_single_transaction`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        plt.title(
            "Why This Prediction? – SHAP Waterfall",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        return fig
    except Exception as exc:
        logger.error("Single-transaction waterfall plot failed: %s", exc)
        plt.close("all")
        raise RuntimeError(
            f"Failed to create single-transaction waterfall plot: {exc}"
        ) from exc