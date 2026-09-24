"""
dataset_utils.py
Utilities for the 04 Dataset Builder stage.

Responsibilities
----------------
1. Model-ready Dataset Construction
   - Apply the Feature / Target Selection decided in 03.
   - Keep only project metadata and selected model columns.

2. Model-ready Dataset Validation
   - Validate the required schema.
   - Validate temporal ordering.
   - Validate NaN / Inf / numeric integrity of selected model columns.

3. Prefix Scenario Definition
   - Define Learning-only evaluation cut-off points.
   - No RUL / Life Metadata generation is performed here.

This module does NOT perform:
- Feature Transformation
- RUL Target Generation
- EDA
- Feature Selection
- Target Selection
- Scaling
- Sequence / Windowing
"""
from __future__ import annotations
from typing import Iterable

import math
import numpy as np
import pandas as pd

from .bearing_utils import META_COLUMNS
from .project_utils import assert_monotonic_order, assert_required_columns


# =============================================================================
# 1. Dataset Configuration
# =============================================================================

# ----- Prefix -----
PREFIX_DEFINITION_COLUMNS = [
    "bearing",
    "prefix_id",
    "prefix_no",
    "prefix_ratio",
    "end_idx",
    "end_time",
]

DEFAULT_PREFIX_START = 0.65
DEFAULT_PREFIX_END = 0.92
DEFAULT_PREFIX_STEP = 0.05

# ----- Missing Value -----
# Structurally missing feature modality.
# NaN is allowed only for selected Features matching these prefixes.
# Selected Target is never allowed to contain NaN.
ALLOWED_NAN_PREFIXES = ("temp_",)


# =============================================================================
# 2. Model-ready Dataset
# =============================================================================

# -------------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------------
def build_model_ready_dataset(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    selected_target: str | None = None,
) -> pd.DataFrame:
    """
    Build a model-ready dataset from an Engineered Dataset.

    Only the following columns are retained:

        META_COLUMNS
        candidate_features
        selected_target (Learning only)

    Parameters
    ----------
    df : pd.DataFrame
        02 Engineered Dataset.
    candidate_features : Iterable[str]
        Features selected by the 03 Selection step.
    selected_target : str or None, default=None
        Selected target for Learning. Must be None for Test.

    Returns
    -------
    pd.DataFrame
        Model-ready dataset.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    candidate_features = list(candidate_features)
    if len(candidate_features) != len(set(candidate_features)):
        raise ValueError("candidate_features contains duplicates.")

    if selected_target is not None and selected_target in candidate_features:
        raise ValueError(
            "selected_target must not also appear in candidate_features."
        )

    # Required metadata is part of the 04 input contract and must not
    # be silently omitted.
    assert_required_columns(df, META_COLUMNS)

    selected_columns = list(candidate_features)
    if selected_target is not None:
        selected_columns.append(selected_target)

    missing_selected = [
        column for column in selected_columns
        if column not in df.columns
    ]

    if missing_selected:
        raise KeyError(
            f"Selected column(s) missing in dataset: {missing_selected}"
        )

    keep_columns = [*META_COLUMNS, *candidate_features]
    if selected_target is not None:
        keep_columns.append(selected_target)

    result = df[keep_columns].copy()

    validate_model_ready_dataset(
        result,
        candidate_features=candidate_features,
        selected_target=selected_target,
    )

    return result

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def validate_model_ready_dataset(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    selected_target: str | None = None,
) -> None:
    """
    Validate a model-ready dataset.
    Validation is intentionally performed after column filtering so that
    unselected Feature Candidates do not affect the 04 stage.

    Checks:
        - Required metadata exists.
        - Time ordering is monotonic within each bearing.
        - Selected Features / Target are numeric.
        - NaN is allowed only for selected Features matching
          ALLOWED_NAN_PREFIXES.
        - Selected Target never allows NaN.
        - Inf is never allowed.
    """
    candidate_features = list(candidate_features)

    assert_required_columns(df, META_COLUMNS)
    assert_monotonic_order(df, group_col="bearing", order_col="elapsed_sec")

    model_columns = list(candidate_features)
    if selected_target is not None:
        model_columns.append(selected_target)

    if not model_columns:
        raise ValueError("At least one selected Feature is required.")

    assert_required_columns(df, model_columns)

    non_numeric = [
        column for column in model_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise TypeError(f"Model-ready columns must be numeric: {non_numeric}")

    model_df = df[model_columns]

    # -------------------------------------------------------------------------
    # NaN Validation
    # -------------------------------------------------------------------------
    nan_columns = [
        column for column in model_columns
        if model_df[column].isna().any()
    ]

    if nan_columns:
        feature_set = set(candidate_features)

        allowed_nan_columns = [
            column for column in nan_columns
            if column in feature_set
            and any(
                column.startswith(prefix)
                for prefix in ALLOWED_NAN_PREFIXES
            )
        ]

        unexpected_nan_columns = [
            column for column in nan_columns
            if column not in allowed_nan_columns
        ]

        if allowed_nan_columns:
            print("[Warning] Allowed NaN detected in temperature features")

            for column in allowed_nan_columns:
                nan_count = model_df[column].isna().sum()
                nan_ratio = model_df[column].isna().mean() * 100

                print(f"  {column}: {nan_count:,} rows ({nan_ratio:.2f}%)")

                for bearing, group in df.groupby("bearing"):
                    bearing_nan_count = group[column].isna().sum()

                    if bearing_nan_count == 0:
                        continue

                    if bearing_nan_count == len(group):
                        status = "No temperature file"
                    else:
                        status = (
                            "Partial Missing "
                            f"({bearing_nan_count}/{len(group)})"
                        )

                    print(f"    - {bearing:<12}: {status}")

        if unexpected_nan_columns:
            raise ValueError(
                "Unexpected NaN detected in selected model columns: "
                f"{unexpected_nan_columns}"
            )

    # -------------------------------------------------------------------------
    # Inf Validation
    # -------------------------------------------------------------------------
    inf_columns = [
        column for column in model_columns
        if np.isinf(model_df[column].to_numpy()).any()
    ]

    if inf_columns:
        raise ValueError(
            "Infinite values detected in selected model columns: "
            f"{inf_columns}"
        )


# =============================================================================
# 3. Prefix Evaluation Scenario
# =============================================================================

# -------------------------------------------------------------------------
# Definition
# -------------------------------------------------------------------------
def generate_prefix_ratios(
    start: float = DEFAULT_PREFIX_START,
    end: float = DEFAULT_PREFIX_END,
    step: float = DEFAULT_PREFIX_STEP,
    include_endpoint: bool = True,
) -> list[float]:
    """
    Generate prefix ratios from start to end.
    The endpoint is explicitly included when requested, even when the
    interval does not land exactly on it.
    """
    if not 0.0 < start <= 1.0:
        raise ValueError("start must be in the range (0, 1].")

    if not 0.0 < end <= 1.0:
        raise ValueError("end must be in the range (0, 1].")

    if start > end:
        raise ValueError("start must not be greater than end.")

    if step <= 0.0:
        raise ValueError("step must be greater than 0.")

    ratios: list[float] = []
    ratio = start

    while ratio < end - 1e-7:
        ratios.append(round(ratio, 4))
        ratio += step

    if include_endpoint and (not ratios or abs(ratios[-1] - end) > 1e-7):
        ratios.append(round(end, 4))

    return ratios

def calculate_prefix_indices(
    group: pd.DataFrame, ratios: Iterable[float]
) -> dict[float, int]:
    """
    Convert prefix ratios into zero-based end indices.
    The input group is expected to be ordered by elapsed_sec.
    """
    if group.empty:
        raise ValueError("Cannot create prefixes from an empty bearing group.")

    n = len(group)
    return {ratio: math.floor(ratio * (n - 1)) for ratio in ratios}

def build_prefix_definition_for_bearing(
    bearing: str, group: pd.DataFrame, ratios: Iterable[float]
) -> pd.DataFrame:
    """Build Prefix Definition rows for one bearing."""
    indices = calculate_prefix_indices(group, ratios)
    records = []
    for prefix_no, (ratio, end_idx) in enumerate(indices.items(), start=1):
        records.append(
            {
                "bearing": bearing,
                "prefix_id": f"{bearing}_P{int(ratio * 100):02d}",
                "prefix_no": prefix_no,
                "prefix_ratio": ratio,
                "end_idx": end_idx,
                "end_time": group.iloc[end_idx]["elapsed_sec"],
            }
        )

    return pd.DataFrame(records, columns=PREFIX_DEFINITION_COLUMNS)

def create_prefix_definition(
    dataset: pd.DataFrame,
    start: float = DEFAULT_PREFIX_START,
    end: float = DEFAULT_PREFIX_END,
    step: float = DEFAULT_PREFIX_STEP,
) -> pd.DataFrame:
    """
    Create Learning-only Prefix Evaluation Scenarios.

    This function depends only on:
        bearing
        elapsed_sec

    It does NOT generate or require:
        failure_time
        life_ratio
        RUL_sec
        RUL_norm
        remaining_sec
    """
    assert_required_columns(dataset, ["bearing", "elapsed_sec"])

    if dataset.empty:
        raise ValueError("Cannot create Prefix Definition from an empty dataset.")

    ratios = generate_prefix_ratios(start=start, end=end, step=step)

    prefix_tables = [
        build_prefix_definition_for_bearing(
            bearing, group.sort_values("elapsed_sec"), ratios
        ) for bearing, group in dataset.groupby("bearing", sort=True)
    ]

    if not prefix_tables:
        return pd.DataFrame(columns=PREFIX_DEFINITION_COLUMNS)

    return (pd.concat(prefix_tables, ignore_index=True)
        .sort_values(["bearing", "prefix_no"], ignore_index=True))

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------
def validate_prefix_definition(
    prefix_definition: pd.DataFrame, dataset: pd.DataFrame):
    """
    Validate a Prefix Definition against its Learning Dataset.

    Checks:
        1. Required columns.
        2. Same number of prefixes for every bearing.
        3. Prefix ratio is increasing within each bearing.
        4. Prefix end index is increasing within each bearing.
        5. End index stays within each bearing's valid row range.
        6. All dataset bearings are represented.
    """
    assert_required_columns(prefix_definition, PREFIX_DEFINITION_COLUMNS)
    assert_required_columns(dataset, ["bearing", "elapsed_sec"])

    if prefix_definition.empty:
        raise ValueError("Prefix Definition is empty.")

    bearing_counts = prefix_definition.groupby("bearing")["prefix_no"].count()

    if bearing_counts.nunique() != 1:
        raise ValueError(
            "Every bearing must have the same number of prefixes."
        )

    for bearing, group in prefix_definition.groupby("bearing", sort=True):
        if not group["prefix_ratio"].is_monotonic_increasing:
            raise ValueError(f"{bearing}: prefix_ratio is not increasing.")

        if not group["end_idx"].is_monotonic_increasing:
            raise ValueError(f"{bearing}: end_idx is not increasing.")

    max_indices = dataset.groupby("bearing").size() - 1

    for bearing, group in prefix_definition.groupby("bearing", sort=True):
        if bearing not in max_indices.index:
            raise ValueError(
                f"Prefix Definition contains unknown bearing: {bearing}"
            )

        if group["end_idx"].min() < 0:
            raise ValueError(
                f"{bearing}: negative end_idx detected."
            )

        if group["end_idx"].max() > max_indices[bearing]:
            raise ValueError(
                f"{bearing}: prefix end_idx exceeds dataset bounds."
            )

    dataset_bearings = set(dataset["bearing"].unique())
    prefix_bearings = set(prefix_definition["bearing"].unique())

    if dataset_bearings != prefix_bearings:
        raise ValueError(
            "Prefix Definition bearings do not match the Learning Dataset."
        )

    print("[Validation Passed] Prefix definition integrity verified.")