"""
Evaluation utilities for the PHM Bearing RUL project.

Responsibilities
----------------
- Evaluate actual and predicted RUL values in seconds.
- Provide standard regression metrics.
- Provide the IEEE PHM 2012 Challenge score.

This module intentionally does not perform:
- RUL target generation
- RUL normalization or reconstruction
- bearing-specific failure-time calculation
- model training or prediction
- LOBO fold handling
- prefix selection
- visualization
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)


# =============================================================================
# 1. Evaluation Metrics
# =============================================================================

# ----- PHM Score -----
def calculate_phm_score(y_true, y_pred) -> float:
    """
    Calculate the IEEE PHM 2012 Challenge score.

    The metric is an accuracy score, not a loss. Higher values are better,
    and a perfect prediction returns 1.0.

    For each prediction, percent error is defined as:

        error = (predicted - actual) / actual

    The Challenge applies asymmetric penalties to over-prediction and
    under-prediction. The per-sample accuracy is then averaged.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values in seconds.

    y_pred : array-like
        Predicted RUL values in seconds.

    Returns
    -------
    float
        Mean IEEE PHM score. Higher is better.

    Raises
    ------
    ValueError
        If inputs are invalid or actual RUL contains zero values.
        Percent-error-based PHM scoring is undefined when actual RUL is zero.
    """
    true_array, pred_array = _validate_prediction_arrays(y_true, y_pred)

    if np.any(true_array == 0):
        raise ValueError(
            "IEEE PHM score is undefined when actual RUL contains zero "
            "because the official percent-error formula divides by actual RUL."
        )

    percent_error = 100.0 * (true_array - pred_array) / true_array
    
    accuracy = np.where(
        percent_error > 0,
        np.exp(np.log(0.5) * percent_error / 20.0),
        np.exp(-np.log(0.5) * percent_error / 5.0),
    )

    return float(np.mean(accuracy))

# ----- Regression Metrics -----
def calculate_smape(y_true, y_pred) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    The definition used is:

        SMAPE = mean(
            2 * abs(y_true - y_pred)
            / (abs(y_true) + abs(y_pred))
        ) * 100

    Zero policy
    -----------
    When both actual and predicted values are zero, the contribution of
    that sample is defined as zero because the prediction is exact.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values in seconds.

    y_pred : array-like
        Predicted RUL values in seconds.

    Returns
    -------
    float
        SMAPE expressed as a percentage in the range [0, 200].
    """
    true_array, pred_array = _validate_prediction_arrays(y_true, y_pred)

    denominator = np.abs(true_array) + np.abs(pred_array)
    numerator = 2.0 * np.abs(true_array - pred_array)

    contributions = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )

    return float(np.mean(contributions) * 100.0)

def calculate_r2(y_true, y_pred) -> float:
    """
    Calculate R-squared.

    Returns NaN when R-squared is undefined, including the case where
    fewer than two samples are provided or the actual values are constant.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values in seconds.

    y_pred : array-like
        Predicted RUL values in seconds.

    Returns
    -------
    float
        R-squared value, or NaN when undefined.
    """
    true_array, pred_array = _validate_prediction_arrays(y_true, y_pred)

    if len(true_array) < 2:
        return float("nan")

    if np.all(true_array == true_array[0]):
        return float("nan")

    return float(r2_score(true_array, pred_array, force_finite=False))


# =============================================================================
# 2. Unified Evaluation
# =============================================================================

def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """
    Evaluate predicted RUL values in seconds.

    This function is intentionally domain-agnostic. It receives only actual
    and predicted RUL_sec values and does not depend on bearing, prefix,
    normalized RUL, or failure-time information.

    Metrics
    -------
    RMSE
        Root Mean Squared Error. Lower is better.

    MAE
        Mean Absolute Error. Lower is better.

    SMAPE
        Symmetric Mean Absolute Percentage Error (%). Lower is better.

    R2
        Coefficient of determination. Higher is better.
        NaN when undefined.

    PHM_Score
        IEEE PHM 2012 Challenge accuracy score. Higher is better.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values in seconds.

    y_pred : array-like
        Predicted RUL values in seconds.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    true_array, pred_array = _validate_prediction_arrays(y_true, y_pred)

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(true_array, pred_array))),
        "MAE": float(mean_absolute_error(true_array, pred_array)),
        "SMAPE": calculate_smape(true_array, pred_array),
        "R2": calculate_r2(true_array, pred_array),
        "PHM_Score": calculate_phm_score(true_array, pred_array),
    }

    return metrics


# =============================================================================
# 3. DataFrame Convenience Interface
# =============================================================================

def evaluate_prediction_table(
    prediction_df: pd.DataFrame,
    actual_col: str = "actual_RUL_sec",
    predicted_col: str = "predicted_RUL_sec",
) -> dict[str, float]:
    """
    Evaluate a prediction table containing RUL values in seconds.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Prediction table.

    actual_col : str, default="actual_RUL_sec"
        Column containing actual RUL values in seconds.

    predicted_col : str, default="predicted_RUL_sec"
        Column containing predicted RUL values in seconds.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.

    Raises
    ------
    TypeError
        If prediction_df is not a pandas DataFrame.

    KeyError
        If required evaluation columns are missing.
    """
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError("prediction_df must be a pandas DataFrame.")

    required_columns = [actual_col, predicted_col]
    missing_columns = [
        column for column in required_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required evaluation column(s): {missing_columns}"
        )

    return evaluate_predictions(
        prediction_df[actual_col].to_numpy(),
        prediction_df[predicted_col].to_numpy(),
    )

def build_3tier_evaluation_report(
    prediction_df: pd.DataFrame,
    bearing_col: str = "bearing",
    actual_col: str = "actual_RUL_sec",
    predicted_col: str = "predicted_RUL_sec",
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Build bearing-level and global evaluation reports.

    Tier structure
    --------------
    Tier 1
        Sample-level prediction table supplied by ``prediction_df``.

    Tier 2
        Bearing-level metrics calculated independently for each bearing.

    Tier 3
        Global metrics calculated over all prediction rows.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Sample-level prediction table containing bearing identifiers and
        actual/predicted RUL values in seconds.

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    actual_col : str, default="actual_RUL_sec"
        Column containing actual RUL values in seconds.

    predicted_col : str, default="predicted_RUL_sec"
        Column containing predicted RUL values in seconds.

    Returns
    -------
    tuple[dict[str, float], pd.DataFrame]
        A tuple containing:

        global_metrics
            Metrics evaluated over all prediction rows.

        bearing_metrics
            One metrics row per bearing, sorted by bearing name.
            Includes ``n_samples``.

    Raises
    ------
    TypeError
        If prediction_df is not a pandas DataFrame.

    KeyError
        If required columns are missing.

    ValueError
        If prediction_df is empty.
    """
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError("prediction_df must be a pandas DataFrame.")

    required_columns = [bearing_col, actual_col, predicted_col]
    missing_columns = [
        column for column in required_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing required evaluation column(s): "
            f"{missing_columns}"
        )

    if prediction_df.empty:
        raise ValueError("prediction_df must not be empty.")

    bearing_rows = []
    grouped = prediction_df.groupby(bearing_col, sort=True)
    for bearing, bearing_df in grouped:
        metrics = evaluate_prediction_table(
            bearing_df,
            actual_col=actual_col,
            predicted_col=predicted_col,
        )

        bearing_rows.append({
            bearing_col: bearing,
            "n_samples": int(len(bearing_df)),
            **metrics
        })

    bearing_metrics = pd.DataFrame(bearing_rows)
    global_metrics = evaluate_prediction_table(
        prediction_df,
        actual_col=actual_col,
        predicted_col=predicted_col,
    )

    return global_metrics, bearing_metrics


# =============================================================================
# 4. Internal Helpers
# =============================================================================

def _validate_prediction_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize prediction inputs.

    Parameters
    ----------
    y_true : array-like
        Actual RUL values in seconds.

    y_pred : array-like
        Predicted RUL values in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated one-dimensional float arrays.

    Raises
    ------
    ValueError
        If the inputs are empty, have different lengths, are not
        one-dimensional, or contain non-finite values.
    """
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)

    if true_array.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if pred_array.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")

    if len(true_array) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(true_array) != len(pred_array):
        raise ValueError("y_true and y_pred must have the same length.")

    if not np.isfinite(true_array).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(pred_array).all():
        raise ValueError("y_pred contains non-finite values.")

    return true_array, pred_array