"""
Bearing / PHM domain utilities.

Responsibilities
----------------
- Define bearing-domain metadata columns and RUL target candidates.
- Provide primitives for failure time and RUL calculations.
- Generate RUL target candidates for learning dataset.
- Reconstruct normalized RUL predictions back to RUL_sec (Validation & Test).
"""
import numpy as np
import pandas as pd


# =============================================================================
# 1. Domain Constants & Schema
# =============================================================================
META_COLUMNS = [
    "bearing",
    "condition",
    "rpm",
    "load",
    "file_idx",
    "hour",
    "minute",
    "second",
    "elapsed_sec",
]

TARGET_CANDIDATES = ["life_ratio", "RUL_sec", "RUL_norm"]


# =============================================================================
# 2. RUL & Life Calculation Primitives
# =============================================================================
def calculate_failure_time(
    df: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Calculate the failure time for each bearing.
    Failure time is defined as the maximum elapsed time observed
    for each bearing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    bearing_col : str, default="bearing"
        Column identifying each bearing.
    time_col : str, default="elapsed_sec"
        Elapsed-time column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with ``failure_time`` added.
    """
    result = df.copy()

    failure_times = result.groupby(bearing_col)[time_col].transform("max")
    result["failure_time"] = failure_times
    return result

def calculate_rul_sec(
    df: pd.DataFrame,
    time_col: str = "elapsed_sec",
    failure_col: str = "failure_time",
) -> pd.DataFrame:
    """
    Calculate remaining useful life in seconds.

    RUL_sec = failure_time - elapsed_sec

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing failure time.
    time_col : str, default="elapsed_sec"
        Elapsed-time column.
    failure_col : str, default="failure_time"
        Failure-time column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with ``RUL_sec`` added.
    """
    result = df.copy()
    result["RUL_sec"] = result[failure_col] - result[time_col]
    return result

def calculate_normalized_targets(
    df: pd.DataFrame,
    time_col: str = "elapsed_sec",
    rul_col: str = "RUL_sec",
    failure_col: str = "failure_time",
) -> pd.DataFrame:
    result = df.copy()
    safe_failure_time = np.maximum(result[failure_col], np.finfo(float).eps)

    result["life_ratio"] = result[time_col] / safe_failure_time
    result["RUL_norm"] = result[rul_col] / safe_failure_time
    return result


# =============================================================================
# 3. Learning Target Generation
# =============================================================================
def generate_rul_targets(
    df: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Generate RUL and life-related target candidates.
    The calculation is performed independently for each bearing.

    Generated columns
    -----------------
    failure_time
        Maximum elapsed_sec observed for each bearing.
        This is an auxiliary calculation column, not a target candidate.

    life_ratio
        elapsed_sec / failure_time

    RUL_sec
        failure_time - elapsed_sec

    RUL_norm
        RUL_sec / failure_time

    Parameters
    ----------
    df : pd.DataFrame
        Learning dataset containing bearing identifiers and elapsed time.
    bearing_col : str, default="bearing"
        Column identifying each bearing.
    time_col : str, default="elapsed_sec"
        Elapsed-time column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with failure time and target
        candidates added.

    Notes
    -----
    ``remaining_sec`` and ``age_sec`` from the previous Dataset Builder
    design are intentionally not generated. Their roles are replaced by
    ``RUL_sec`` and the existing ``elapsed_sec`` column respectively.
    """
    result = calculate_failure_time(
        df, bearing_col=bearing_col, time_col=time_col
    )

    result = calculate_rul_sec(
        result,
        time_col=time_col,
        failure_col="failure_time",
    )

    result = calculate_normalized_targets(
        result,
        time_col=time_col,
        rul_col="RUL_sec",
        failure_col="failure_time",
    )

    return result


# =============================================================================
# 4. RUL Prediction Reconstruction
# =============================================================================
def reconstruct_rul_sec_from_norm(
    df_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
    pred_norm_col: str = "predicted_RUL_norm",
    actual_norm_col: str = "actual_RUL_norm",
) -> pd.DataFrame:
    """
    Reconstruct normalized RUL values to RUL_sec.

    Failure time is obtained from the learning dataset using the
    established bearing-domain definition:

        failure_time(B) = max(elapsed_sec of bearing B)

    Both actual and predicted normalized RUL values are reconstructed
    using the same bearing-specific failure time:

        actual_RUL_sec
            = actual_RUL_norm * failure_time(B)

        predicted_RUL_sec
            = predicted_RUL_norm * failure_time(B)

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Sample-level prediction table containing bearing, elapsed time,
        actual normalized RUL, and predicted normalized RUL.

    dataset : pd.DataFrame
        Model-ready learning dataset used to obtain bearing-specific
        failure times.

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    time_col : str, default="elapsed_sec"
        Elapsed-time column.

    pred_norm_col : str, default="predicted_RUL_norm"
        Column containing predicted normalized RUL values.

    actual_norm_col : str, default="actual_RUL_norm"
        Column containing actual normalized RUL values.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_predictions`` with the following columns added:

        - failure_time
        - actual_RUL_sec
        - predicted_RUL_sec
        - absolute_error_sec

    Raises
    ------
    TypeError
        If either input is not a pandas DataFrame.

    KeyError
        If required columns are missing.

    ValueError
        If a prediction bearing is missing from the failure-time mapping,
        or if invalid failure times are found.
    """
    if not isinstance(df_predictions, pd.DataFrame):
        raise TypeError("df_predictions must be a pandas DataFrame.")

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame.")

    prediction_required = [bearing_col, time_col, actual_norm_col, pred_norm_col]
    dataset_required = [bearing_col, time_col]

    missing_prediction_columns = [
        column for column in prediction_required
        if column not in df_predictions.columns
    ]

    if missing_prediction_columns:
        raise KeyError(
            "Missing required prediction column(s): "
            f"{missing_prediction_columns}"
        )

    missing_dataset_columns = [
        column for column in dataset_required
        if column not in dataset.columns
    ]

    if missing_dataset_columns:
        raise KeyError(
            "Missing required dataset column(s): "
            f"{missing_dataset_columns}"
        )

    result = df_predictions.copy()

    failure_time_map = (
        dataset.groupby(bearing_col)[time_col]
        .max().rename("failure_time")
    )

    result["failure_time"] = result[bearing_col].map(failure_time_map)

    missing_bearings = sorted(
        result.loc[result["failure_time"].isna(), bearing_col]
        .dropna().unique().tolist()
    )

    if missing_bearings:
        raise ValueError(
            "Prediction bearing(s) missing from dataset failure-time "
            f"mapping: {missing_bearings}"
        )

    if result["failure_time"].isna().any():
        raise ValueError(
            "Unable to resolve failure_time for one or more prediction rows."
        )

    invalid_failure_mask = (
        ~np.isfinite(result["failure_time"])
        | (result["failure_time"] <= 0)
    )

    if invalid_failure_mask.any():
        invalid_bearings = sorted(
            result.loc[invalid_failure_mask, bearing_col]
            .dropna().unique().tolist()
        )

        raise ValueError(
            "Invalid failure_time found for bearing(s): "
            f"{invalid_bearings}"
        )

    result["actual_RUL_sec"] = result[actual_norm_col] * result["failure_time"]
    result["predicted_RUL_sec"] = result[pred_norm_col] * result["failure_time"]

    result["absolute_error_sec"] = np.abs(
        result["actual_RUL_sec"] - result["predicted_RUL_sec"]
    )

    return result


def reconstruct_test_rul_sec_from_norm(
    df_predictions: pd.DataFrame,
    time_col: str = "elapsed_sec",
    pred_norm_col: str = "predicted_RUL_norm",
) -> pd.DataFrame:
    """
    Reconstruct Test RUL_norm predictions to RUL_sec.

    Test failure time is unknown. It is therefore reconstructed from
    the endpoint elapsed time and predicted RUL_norm:

        failure_time
            = elapsed_sec / (1 - predicted_RUL_norm)

        predicted_RUL_sec
            = predicted_RUL_norm * failure_time

            = elapsed_sec * predicted_RUL_norm
              / (1 - predicted_RUL_norm)

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Test endpoint prediction table containing elapsed time and
        ensembled predicted RUL_norm.

    time_col : str, default="elapsed_sec"
        Elapsed-time column.

    pred_norm_col : str, default="predicted_RUL_norm"
        Column containing ensembled predicted RUL_norm.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with:

        - predicted_RUL_sec

        added.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame or required columns are
        non-numeric.

    KeyError
        If required columns are missing.

    ValueError
        If elapsed time is negative or non-finite, or if predicted
        RUL_norm is outside the valid range [0, 1).
    """
    if not isinstance(df_predictions, pd.DataFrame):
        raise TypeError("df_predictions must be a pandas DataFrame.")

    required_columns = [time_col, pred_norm_col]
    missing_columns = [
        column for column in required_columns
        if column not in df_predictions.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required prediction column(s): {missing_columns}")

    if not pd.api.types.is_numeric_dtype(df_predictions[time_col]):
        raise TypeError(f"'{time_col}' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df_predictions[pred_norm_col]):
        raise TypeError(f"'{pred_norm_col}' must be numeric.")

    elapsed_sec = df_predictions[time_col].to_numpy(dtype=float)
    predicted_rul_norm = (
        df_predictions[pred_norm_col].to_numpy(dtype=float)
    )

    if not np.isfinite(elapsed_sec).all():
        raise ValueError(f"'{time_col}' contains NaN or Inf values.")

    if not np.isfinite(predicted_rul_norm).all():
        raise ValueError(f"'{pred_norm_col}' contains NaN or Inf values.")

    if (elapsed_sec < 0).any():
        raise ValueError(f"'{time_col}' must contain non-negative values.")

    if ((predicted_rul_norm < 0) | (predicted_rul_norm >= 1)).any():
        raise ValueError(f"'{pred_norm_col}' must satisfy 0 <= RUL_norm < 1.")

    denominator = 1.0 - predicted_rul_norm
    predicted_rul_sec = elapsed_sec * predicted_rul_norm / denominator

    if not np.isfinite(predicted_rul_sec).all():
        raise ValueError(
            "Reconstructed 'predicted_RUL_sec' contains "
            "NaN or Inf values."
        )

    result = df_predictions.copy()
    result["predicted_RUL_sec"] = predicted_rul_sec
    return result