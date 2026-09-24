"""
feature_utils.py

Feature Transformation utilities for the 02 Feature Transformation stage.

Pipeline
--------
Raw Feature
    ↓
Shock Extraction
    ↓
Rectified Feature
    ↓
Cumulative Rectified Shock
    ↓
Relative Damage Index (RDI)

Responsibilities
----------------
Configuration
    - RDI transformation target configuration.

Primitive Operations
    - Shock extraction.
    - Cumulative shock calculation.
    - Rectified feature normalization.
    - Relative damage normalization.

Feature Transformation
    - Single-feature damage transformation.
    - Bearing-wise feature transformation.

This module does not perform:
- EDA
- Feature selection
- Target generation
- Blueprint management
- Dataset construction
- Reporting
"""
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 1. Transformation Configuration
# =============================================================================
# 기본 Transformation 대상 Feature.

# 주의:
# RDI_FEATURES는 Feature Selection 결과가 아니다.
# build_damage_features()의 features 인자에 대한 기본값이다.
RDI_FEATURES = ["rms", "p2p", "cf", "kurt"]


# =============================================================================
# 2. Transformation Primitives
# =============================================================================

def calculate_shock(series: pd.Series, threshold: float) -> pd.Series:
    """
    Calculate shock magnitude above the threshold.

    Parameters
    ----------
    series : pd.Series
        Raw feature time series.
    threshold : float
        Threshold applied to the feature.

    Returns
    -------
    pd.Series
        Shock series. Values below the threshold are zero.
    """
    return np.maximum(series - threshold, 0.0)

def calculate_cumulative_shock(shock_series: pd.Series) -> pd.Series:
    """
    Calculate cumulative shock.

    Parameters
    ----------
    shock_series : pd.Series
        Shock time series.

    Returns
    -------
    pd.Series
        Cumulative shock series.
    """
    return shock_series.cumsum()

def calculate_rectified_feature(
    shock: np.ndarray,
    baseline_std: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize shock magnitude by baseline standard deviation.

    Parameters
    ----------
    shock : np.ndarray
        Shock signal.
    baseline_std : float
        Standard deviation measured from the baseline region.
    eps : float, default=1e-12
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Rectified feature.
    """
    safe_std = max(float(baseline_std), eps)
    return shock / safe_std

def calculate_relative_damage(
    cum_shock_series: pd.Series, baseline_damage: float
) -> pd.Series:
    """
    Calculate Relative Damage Index (RDI).

    Parameters
    ----------
    cum_shock_series : pd.Series
        Cumulative rectified shock series.
    baseline_damage : float
        Cumulative damage at the end of the baseline region.

    Returns
    -------
    pd.Series
        Relative Damage Index.
    """
    safe_denominator = max(float(baseline_damage), np.finfo(float).eps)
    return cum_shock_series / safe_denominator


# =============================================================================
# 3. Feature Transformation
# =============================================================================

# ----- Single Feature -----
def build_damage_feature(
    series: pd.Series,
    baseline_ratio: float = 0.05,
    threshold_k: float = 0.0,
) -> dict[str, pd.Series]:
    """
    Transform one raw Feature into Rectified and RDI Features.

    Pipeline
    --------
    Raw
        ↓
    Baseline
        ↓
    Threshold
        ↓
    Shock
        ↓
    Rectified
        ↓
    Cumulative Rectified
        ↓
    RDI

    Parameters
    ----------
    series : pd.Series
        Single raw Feature time series.
    baseline_ratio : float, default=0.05
        Ratio of the initial series used as the baseline region.
    threshold_k : float, default=0.0
        Threshold coefficient.

        threshold = baseline_mean + threshold_k * baseline_std

    Returns
    -------
    dict[str, pd.Series]
        {"Rectified": rectified, "RDI": rdi}

    Notes
    -----
    Cumulative Rectified is an intermediate value used only for
    RDI calculation and is not returned as a Feature Candidate.

    Rectified itself is returned because it is a Feature Candidate.
    """
    if not 0.0 < baseline_ratio <= 1.0:
        raise ValueError(
            "baseline_ratio must be greater than 0 "
            "and less than or equal to 1.0."
        )

    if len(series) == 0:
        raise ValueError("Cannot transform an empty Feature series.")

    baseline_length = min(
        max(int(len(series) * baseline_ratio), 10), len(series)
    )

    baseline = series.iloc[:baseline_length]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std()

    threshold = baseline_mean + threshold_k * baseline_std

    # -------------------------------------------------------------------------
    # 1. Shock
    # -------------------------------------------------------------------------
    shock = calculate_shock(series, threshold)

    # -------------------------------------------------------------------------
    # 2. Rectified Feature
    # -------------------------------------------------------------------------
    rectified_values = calculate_rectified_feature(
        shock.to_numpy(), baseline_std
    )

    rectified = pd.Series(
        rectified_values, index=series.index, name="Rectified"
    )

    # -------------------------------------------------------------------------
    # 3. Cumulative Rectified Shock
    # -------------------------------------------------------------------------
    cumulative_rectified = calculate_cumulative_shock(rectified)

    # -------------------------------------------------------------------------
    # 4. Baseline Damage
    # -------------------------------------------------------------------------
    baseline_damage = cumulative_rectified.iloc[baseline_length - 1]

    # -------------------------------------------------------------------------
    # 5. Relative Damage Index
    # -------------------------------------------------------------------------
    rdi = calculate_relative_damage(
        cumulative_rectified, baseline_damage
    )
    rdi.name = "RDI"

    return {"Rectified": rectified, "RDI": rdi}

# ----- Multiple Features -----
def build_damage_features(
    df: pd.DataFrame,
    features: Iterable[str] = RDI_FEATURES,
    baseline_ratio: float = 0.05,
    threshold_k: float = 0.0,
    bearing_col: str = "bearing",
) -> pd.DataFrame:
    """
    Generate transformed Feature Candidates for each bearing.
    Each requested raw Feature is independently transformed into:

        Rectified_<feature>
        RDI_<feature>

    Parameters
    ----------
    df : pd.DataFrame
        Input Base Feature Dataset.
    features : Iterable[str], default=RDI_FEATURES
        Raw Features to transform.

        This is the replacement for the former Blueprint input.
        RDI_FEATURES is only the default Transformation target list;
        it does not represent Feature Selection.
    baseline_ratio : float, default=0.05
        Ratio of each bearing's series used as the baseline region.
    threshold_k : float, default=0.0
        Threshold coefficient.
    bearing_col : str, default="bearing"
        Column identifying each bearing.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with transformed Feature
        Candidates appended.

    Raises
    ------
    KeyError
        If bearing_col is missing.

    Notes
    -----
    Transformation is performed independently for each bearing.

    Learning and Test datasets use the same function independently.
    No dataset-type-specific branch is required.

    No Blueprint is read, modified, or generated.
    No Feature Selection is performed.
    """
    output = df.copy()

    if bearing_col not in output.columns:
        raise KeyError(f"'{bearing_col}' column is required.")

    # Materialize the iterable once so generators are also supported.
    target_features = list(features)

    # Only Features actually present in the input dataset can be transformed.
    target_features = [
        feature for feature in target_features
        if feature in output.columns
    ]

    for _, group in output.groupby(bearing_col, sort=False):
        for feature in target_features:
            transformed = build_damage_feature(
                group[feature],
                baseline_ratio=baseline_ratio,
                threshold_k=threshold_k,
            )

            rectified_name = f"Rectified_{feature}"
            rdi_name = f"RDI_{feature}"

            output.loc[
                group.index, rectified_name
            ] = transformed["Rectified"].to_numpy()

            output.loc[
                group.index, rdi_name
            ] = transformed["RDI"].to_numpy()

    return output