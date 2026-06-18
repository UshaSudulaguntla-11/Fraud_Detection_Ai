"""
FraudLens AI - Prediction Module
=================================
Handles model loading, single/batch predictions, input validation,
and risk classification for the fraud detection pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths – resolved relative to the project root (one level above utils/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "fraud_detection_model.pkl")
FEATURE_NAMES_PATH = os.path.join(_PROJECT_ROOT, "feature_names.pkl")


# ---------------------------------------------------------------------------
# Model & feature loading
# ---------------------------------------------------------------------------
def load_model():
    """Load the trained fraud-detection model from disk.

    Returns
    -------
    model : sklearn-compatible estimator
        The de-serialised Random Forest (or compatible) classifier.

    Raises
    ------
    FileNotFoundError
        If the model pickle is missing.
    RuntimeError
        If the pickle cannot be loaded.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Ensure 'fraud_detection_model.pkl' is in the project root."
        )
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


def load_feature_names() -> List[str]:
    """Load the list of feature names used during training.

    Returns
    -------
    list[str]
        Feature names (e.g. ``['Time', 'V1', …, 'V28', 'Amount']``).

    Raises
    ------
    FileNotFoundError
        If the feature-names pickle is missing.
    """
    if not os.path.exists(FEATURE_NAMES_PATH):
        raise FileNotFoundError(
            f"Feature-names file not found at {FEATURE_NAMES_PATH}. "
            "Ensure 'feature_names.pkl' is in the project root."
        )
    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Loaded %d feature names.", len(feature_names))
        return list(feature_names)
    except Exception as exc:
        raise RuntimeError(f"Failed to load feature names: {exc}") from exc


# ---------------------------------------------------------------------------
# Single-transaction prediction
# ---------------------------------------------------------------------------
def predict_single(
    model,
    input_data: Dict[str, float],
) -> Tuple[int, float, float]:
    """Predict whether a single transaction is fraudulent.

    Parameters
    ----------
    model : sklearn-compatible estimator
        A trained classifier exposing ``predict`` and ``predict_proba``.
    input_data : dict[str, float]
        Feature-name → value mapping for **one** transaction.

    Returns
    -------
    prediction : int
        ``0`` (legitimate) or ``1`` (fraud).
    fraud_probability : float
        Probability of the fraud class (class 1).
    confidence : float
        Maximum of the two class probabilities (i.e. how certain the model is).
    """
    try:
        df = pd.DataFrame([input_data])

        # Ensure column order matches what the model was trained on.
        # Dict insertion order in the caller (e.g. the Streamlit form) is not
        # guaranteed to match the training feature order, and recent
        # scikit-learn versions raise if column order doesn't match
        # `feature_names_in_` exactly.
        expected_order = getattr(model, "feature_names_in_", None)
        if expected_order is not None:
            df = df[list(expected_order)]
        else:
            try:
                expected_order = load_feature_names()
                df = df[expected_order]
            except Exception:
                pass  # fall back to whatever order input_data provided

        prediction = int(model.predict(df)[0])

        probabilities = model.predict_proba(df)[0]
        fraud_probability = float(probabilities[1])
        confidence = float(np.max(probabilities))

        return prediction, fraud_probability, confidence

    except Exception as exc:
        logger.error("Single prediction failed: %s", exc)
        raise ValueError(f"Prediction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(
    model,
    df: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """Predict fraud for every row in a DataFrame.

    The function validates the input columns, runs inference, and appends
    three new columns to a **copy** of the input DataFrame.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Trained classifier.
    df : pd.DataFrame
        Input data whose columns must be a superset of *feature_names*.
    feature_names : list[str]
        The exact feature names used during training.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with added columns:

        * ``Prediction`` – ``0`` or ``1``
        * ``Fraud_Probability`` – probability of the fraud class
        * ``Label`` – human-readable label (``"Fraud"`` / ``"Legitimate"``)
    """
    is_valid, error_msg = validate_upload(df, feature_names)
    if not is_valid:
        raise ValueError(f"Batch prediction failed – {error_msg}")

    try:
        result_df = df.copy()
        X = result_df[feature_names]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        result_df["Prediction"] = predictions.astype(int)
        result_df["Fraud_Probability"] = np.round(probabilities, 6)
        result_df["Label"] = result_df["Prediction"].map(
            {0: "Legitimate", 1: "Fraud"}
        )

        logger.info(
            "Batch prediction complete – %d rows, %d flagged as fraud.",
            len(result_df),
            int(result_df["Prediction"].sum()),
        )
        return result_df

    except ValueError:
        raise  # re-raise validation errors as-is
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise RuntimeError(f"Batch prediction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_upload(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[bool, str]:
    """Validate that an uploaded CSV contains the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded data.
    feature_names : list[str]
        Required column names.

    Returns
    -------
    is_valid : bool
        ``True`` when all required columns are present and data looks sane.
    error_message : str
        Empty string on success; descriptive message on failure.
    """
    if df is None or df.empty:
        return False, "The uploaded file is empty or could not be read."

    missing = set(feature_names) - set(df.columns)
    if missing:
        return (
            False,
            f"Missing {len(missing)} required column(s): {sorted(missing)}",
        )

    # Check for non-numeric values in feature columns
    non_numeric = []
    for col in feature_names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        return (
            False,
            f"Non-numeric data found in column(s): {sorted(non_numeric)}",
        )

    # Check for NaN / Inf in feature columns
    feature_data = df[feature_names]
    nan_count = int(feature_data.isna().sum().sum())
    inf_count = int(np.isinf(feature_data.select_dtypes(include=[np.number])).sum().sum())
    if nan_count > 0:
        return False, f"Data contains {nan_count} missing value(s) in feature columns."
    if inf_count > 0:
        return False, f"Data contains {inf_count} infinite value(s) in feature columns."

    return True, ""


# ---------------------------------------------------------------------------
# Model metrics (pre-computed from the training notebook)
# ---------------------------------------------------------------------------
def get_model_metrics() -> Dict[str, Dict[str, float]]:
    """Return pre-computed performance metrics for the evaluated models.

    These are hard-coded from the training notebook evaluation and are
    displayed in the analytics dashboard for model comparison.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of model name → {Precision, Recall, F1 Score}.
    """
    return {
        "Logistic Regression": {
            "Precision": 0.07,
            "Recall": 0.91,
            "F1 Score": 0.12,
        },
        "Random Forest": {
            "Precision": 0.83,
            "Recall": 0.83,
            "F1 Score": 0.83,
        },
        "XGBoost": {
            "Precision": 0.53,
            "Recall": 0.89,
            "F1 Score": 0.66,
        },
    }


# ---------------------------------------------------------------------------
# Risk-level classification
# ---------------------------------------------------------------------------
def get_risk_level(probability: float) -> Tuple[str, str]:
    """Classify a fraud probability into a human-readable risk level.

    Parameters
    ----------
    probability : float
        Fraud probability in [0, 1].

    Returns
    -------
    risk_level : str
        One of ``"Low Risk"``, ``"Medium Risk"``, or ``"High Risk"``.
    color : str
        CSS-compatible colour string for UI rendering.
    """
    if probability < 0.3:
        return "Low Risk", "green"
    elif probability < 0.7:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"