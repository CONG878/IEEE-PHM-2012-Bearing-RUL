"""
Exploratory Data Analysis utilities for the 03 EDA & Feature Selection stage.

Responsibilities
----------------
Visualization
    - Feature distribution.
    - Operating-condition comparison.
    - Bearing-wise degradation trend.
    - Feature-to-target relationship.

Feature Evidence
    - Monotonicity.
    - Trendability.
    - Prognosability.
    - Condition stability (CV).
    - Feature ranking.
    - Feature redundancy.

Target Evidence
    - Global Feature-to-Target correlation.
    - Bearing-wise Feature-to-Target correlation.

This module does not perform:
- Feature Transformation
- Target Generation
- Feature Selection
- Target Selection
- Blueprint management
- Dataset construction
- File output
"""
from __future__ import annotations
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

from scipy.stats import (
    kendalltau, pearsonr, probplot, spearmanr
)


# =============================================================================
# 1. EDA Visualization
# =============================================================================

# ----- Feature Distribution -----
def plot_distribution(
    df: pd.DataFrame,
    feature: str,
    bins: int = 30,
    figsize: tuple = (15, 4),
    kde: bool = True
) -> Figure:
    """
    Plot histogram, boxplot and Q-Q plot for one feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature to visualize.
    bins : int, default=30
        Number of histogram bins.
    figsize : tuple, default=(15, 4)
        Figure size.
    kde : bool, default=True
        Whether to display KDE on the histogram.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    values = df[feature].dropna()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 0.8, 1.0])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    sns.histplot(values, bins=bins, kde=kde, ax=ax1)
    ax1.set_title("Histogram")

    sns.boxplot(y=values, ax=ax2)
    ax2.set_title("Boxplot")

    probplot(values, dist="norm", plot=ax3)
    ax3.set_title("Normal Q-Q")

    fig.suptitle(feature, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig

# ----- Operating Condition -----
def plot_condition(
    df: pd.DataFrame,
    feature: str,
    condition_col: str = "condition",
    figsize: tuple = (6, 4)
) -> Figure:
    """
    Compare feature distributions across operating conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature to visualize.
    condition_col : str, default="condition"
        Operating-condition column.
    figsize : tuple, default=(6, 4)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=condition_col, y=feature, data=df, ax=ax)

    ax.set_title(f"{feature} by {condition_col}")
    fig.tight_layout()
    plt.show()

    return fig

# ----- Degradation Trend -----
def plot_bearing_trend(
    df: pd.DataFrame,
    feature: str,
    x_col: str = "life_ratio",
    bearing_col: str = "bearing",
    figsize: tuple = (14, 8),
):
    """
    Plot one Feature's degradation trend for every bearing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    feature : str
        Feature to visualize.

    x_col : str, default="life_ratio"
        Column used as the x-axis.
        Examples:
            "life_ratio"
            "elapsed_sec"

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    figsize : tuple, default=(14, 8)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Generated Figure.

    Notes
    -----
    The function does not determine which x-axis is appropriate.
    The caller explicitly specifies the x-axis through x_col.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found.")

    if x_col not in df.columns:
        raise ValueError(f"X-axis column '{x_col}' not found.")

    if bearing_col not in df.columns:
        raise ValueError(f"Bearing column '{bearing_col}' not found.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    bearings = sorted(df[bearing_col].dropna().unique())
    if not bearings:
        raise ValueError(
            f"No valid bearings found in '{bearing_col}'."
        )

    n = len(bearings)
    cols = 2
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, bearing in zip(axes, bearings):
        temp = df[df[bearing_col] == bearing].sort_values(x_col)

        ax.plot(temp[x_col], temp[feature], lw=1.5)

        ax.set_title(str(bearing))
        ax.set_xlabel(x_col)
        ax.set_ylabel(feature)
        ax.grid(True)

    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(f"{feature} Trend", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig

# ----- Feature–Target Relationship -----
def plot_feature_vs_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bearing_col: str = "bearing",
    figsize: tuple = (8, 6),
):
    """
    Plot the relationship between one Feature and one Target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    feature : str
        Feature to visualize on the x-axis.

    target : str
        Target Candidate to visualize on the y-axis.

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Generated Figure.

    Notes
    -----
    The function visualizes one explicit Feature–Target combination.

    It does not:
        - iterate over all Features,
        - iterate over all Targets,
        - perform Feature Selection,
        - calculate correlation coefficients,
        - generate degradation trend plots.

    Bearing identity is represented by separate scatter series.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found.")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    if bearing_col not in df.columns:
        raise ValueError(f"Bearing column '{bearing_col}' not found.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    plot_columns = [feature, target, bearing_col]
    plot_df = df[plot_columns].dropna(subset=[feature, target]).copy()

    fig, ax = plt.subplots(figsize=figsize)
    for bearing, group in plot_df.groupby(bearing_col, sort=True):
        ax.scatter(
            group[feature],
            group[target],
            s=12,
            alpha=0.6,
            label=str(bearing),
        )

    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target}")
    ax.grid(True)
    ax.legend(
        title=bearing_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()

    return fig


# =============================================================================
# 2. Feature Evidence
# =============================================================================

# -------------------------------------------------------------------------
# Feature Metrics
# -------------------------------------------------------------------------
def calculate_monotonicity(
    df: pd.DataFrame, feature: str, bearing_col: str = "bearing"
) -> float:
    """
    Compute monotonicity score.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature Candidate.
    bearing_col : str, default="bearing"
        Bearing identifier.

    Returns
    -------
    float
        Mean monotonicity over all bearings.
    """
    _validate_metric_input(df, feature, bearing_col)

    scores = []
    for _, group in df.groupby(bearing_col):
        signal = group[feature].dropna().to_numpy()

        if len(signal) < 3:
            continue

        diff = np.diff(signal)

        score = np.abs(np.sum(np.sign(diff))) / len(diff)
        scores.append(score)

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))

def calculate_trendability(
    df: pd.DataFrame,
    feature: str,
    bearing_col: str = "bearing",
    x_col: str = "life_ratio",
    num_points: int = 200,
) -> float:
    """
    Compute trendability score based on pairwise Spearman correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Input Learning Engineered Dataset.
    feature : str
        Feature Candidate.
    bearing_col : str, default="bearing"
        Bearing identifier.
    x_col : str, default="life_ratio"
        Normalized life axis.
    num_points : int, default=200
        Number of common interpolation points.

    Returns
    -------
    float
        Mean absolute pairwise Spearman correlation.
    """
    _validate_metric_input(df, feature, bearing_col)

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found.")

    bearings = sorted(df[bearing_col].unique())
    pairs = _generate_bearing_pairs(bearings)

    scores = []
    for b1, b2 in pairs:
        g1 = df[df[bearing_col] == b1].sort_values(x_col)
        g2 = df[df[bearing_col] == b2].sort_values(x_col)

        g1 = g1[[x_col, feature]].dropna()
        g2 = g2[[x_col, feature]].dropna()

        if len(g1) < 2 or len(g2) < 2:
            continue

        _, y1 = _resample_series(
            g1[x_col].to_numpy(),
            g1[feature].to_numpy(),
            num_points,
        )

        _, y2 = _resample_series(
            g2[x_col].to_numpy(),
            g2[feature].to_numpy(),
            num_points,
        )

        score = _pairwise_metric(y1, y2)
        if np.isfinite(score):
            scores.append(score)

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))

def calculate_prognosability(
    df: pd.DataFrame,
    feature: str,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> float:
    """
    Compute prognosability score.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature Candidate.
    bearing_col : str, default="bearing"
        Bearing identifier.
    time_col : str, default="elapsed_sec"
        Time column.

    Returns
    -------
    float
        Prognosability score.
    """
    _validate_metric_input(df, feature, bearing_col)

    values = _extract_failure_values(df, feature, bearing_col, time_col)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if np.isclose(mean, 0):
        return np.nan

    score = np.exp(-std / np.abs(mean))
    return float(score)

# -------------------------------------------------------------------------
# Feature Evaluation
# -------------------------------------------------------------------------
def normalize_metric_scores(
    metric_df: pd.DataFrame, metric_columns: Iterable[str]
) -> pd.DataFrame:
    """
    Min-Max normalize specified metric columns.

    Parameters
    ----------
    metric_df : pd.DataFrame
        Metric dataframe.
    metric_columns : Iterable[str]
        Columns to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized copy.
    """
    result = metric_df.copy()

    for col in metric_columns:
        x = result[col]
        xmin = x.min()
        xmax = x.max()

        if np.isclose(xmin, xmax):
            result[col] = 1.0
        else:
            result[col] = (x - xmin) / (xmax - xmin)

    return result

def find_redundant_features(
    df: pd.DataFrame,
    features: Iterable[str],
    threshold: float = 0.95,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Find highly correlated Feature Candidate pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : Iterable[str]
        Feature Candidates to compare.
    threshold : float, default=0.95
        Absolute correlation threshold.
    method : str, default="spearman"
        Correlation method supported by pandas.DataFrame.corr.

    Returns
    -------
    pd.DataFrame
        Redundancy Evidence with columns:

        - Feature1
        - Feature2
        - Correlation

    Notes
    -----
    This function produces Evidence only.
    It does not remove or select any feature.
    """
    features = list(features)

    if len(features) < 2:
        return pd.DataFrame(
            columns=["Feature1", "Feature2", "Correlation"]
        )

    corr = df[features].corr(method=method).abs()
    cols = corr.columns

    redundant = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]

            if value >= threshold:
                redundant.append({
                    "Feature1": cols[i],
                    "Feature2": cols[j],
                    "Correlation": value,
                })

    return pd.DataFrame(
        redundant, columns=["Feature1", "Feature2", "Correlation"],
    ).sort_values(
        "Correlation", ascending=False
    ).reset_index(drop=True)

def rank_features(
    df: pd.DataFrame,
    features: Iterable[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Rank Feature Candidates using EDA metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input Learning Engineered Dataset.
    features : Iterable[str]
        Feature Candidates to evaluate.
    weights : dict[str, float], optional
        Weights for the ranking score.

        Default
        -------
        {
            "Monotonicity": 0.35,
            "Trendability": 0.35,
            "Prognosability": 0.20,
            "CV": 0.10,
        }

    Returns
    -------
    pd.DataFrame
        Feature Ranking Evidence.

    Notes
    -----
    Ranking is Evidence only.
    It does not perform Feature Selection.
    """
    features = list(features)
    rows = []
    cv_map = _calculate_condition_cv(df, features)

    for feature in features:
        mono = calculate_monotonicity(df, feature)
        trend = calculate_trendability(df, feature)
        prog = calculate_prognosability(df, feature)
        cv = cv_map.get(feature, np.nan)

        rows.append({
            "Feature": feature,
            "Monotonicity": mono,
            "Trendability": trend,
            "Prognosability": prog,
            "CV": cv,
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Monotonicity",
                "Trendability",
                "Prognosability",
                "CV",
                "Score",
            ]
        )

    ranking = normalize_metric_scores(
        ranking, ["Monotonicity", "Trendability", "Prognosability"]
    )
    ranking["CV"] = 1 - normalize_metric_scores(ranking[["CV"]], ["CV"])["CV"]

    if weights is None:
        weights = {
            "Monotonicity": 0.35,
            "Trendability": 0.35,
            "Prognosability": 0.20,
            "CV": 0.10,
        }

    ranking["Score"] = (
        weights["Monotonicity"]
        * ranking["Monotonicity"]
        +
        weights["Trendability"]
        * ranking["Trendability"]
        +
        weights["Prognosability"]
        * ranking["Prognosability"]
        +
        weights["CV"]
        * ranking["CV"]
    )

    return ranking.sort_values(
        "Score", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. Target Evidence
# =============================================================================

def calculate_rul_correlation(
    df: pd.DataFrame,
    features: Iterable[str],
    targets: Iterable[str],
) -> pd.DataFrame:
    """
    Calculate global Feature-to-Target correlations.

    For every Feature Candidate and Target Candidate combination,
    Pearson, Spearman and Kendall correlations are calculated.

    Parameters
    ----------
    df : pd.DataFrame
        Input Learning Engineered Dataset.
    features : Iterable[str]
        Feature Candidates.
    targets : Iterable[str]
        Target Candidates.

    Returns
    -------
    pd.DataFrame
        Correlation Evidence with columns:

        - Feature
        - Target
        - Pearson
        - Spearman
        - Kendall

    Notes
    -----
    Correlation signs are preserved.

    This function does not select a Target or Feature.
    """
    features = list(features)
    targets = list(targets)

    rows = []
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found.")

        for target in targets:
            if target not in df.columns:
                raise ValueError(f"Target '{target}' not found.")

            pearson_value, spearman_value, kendall_value = (
                _calculate_correlation_values(df[feature], df[target])
            )

            rows.append({
                "Feature": feature,
                "Target": target,
                "Pearson": pearson_value,
                "Spearman": spearman_value,
                "Kendall": kendall_value,
            })

    return pd.DataFrame(
        rows, columns=[
            "Feature", "Target", "Pearson", "Spearman", "Kendall"
        ]
    )

def calculate_rul_correlation_by_bearing(
    df: pd.DataFrame,
    features: Iterable[str],
    targets: Iterable[str],
    bearing_col: str = "bearing",
) -> pd.DataFrame:
    """
    Calculate Feature-to-Target correlations independently
    for each bearing.

    Parameters
    ----------
    df : pd.DataFrame
        Input Learning Engineered Dataset.
    features : Iterable[str]
        Feature Candidates.
    targets : Iterable[str]
        Target Candidates.
    bearing_col : str, default="bearing"
        Bearing identifier.

    Returns
    -------
    pd.DataFrame
        Bearing-wise Correlation Evidence with columns:

        - bearing
        - Feature
        - Target
        - Pearson
        - Spearman
        - Kendall

    Notes
    -----
    Correlation signs are preserved.

    If a bearing does not contain enough valid observations
    for a correlation calculation, the corresponding values
    are NaN.

    This function does not select a Target or Feature.
    """
    features = list(features)
    targets = list(targets)

    if bearing_col not in df.columns:
        raise ValueError(f"Column '{bearing_col}' not found.")

    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found.")

    for target in targets:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")

    rows = []
    for bearing, group in df.groupby(bearing_col, sort=True):
        for feature in features:
            for target in targets:
                (
                    pearson_value, spearman_value, kendall_value
                ) = _calculate_correlation_values(
                    group[feature], group[target]
                )

                rows.append({
                    "bearing": bearing,
                    "Feature": feature,
                    "Target": target,
                    "Pearson": pearson_value,
                    "Spearman": spearman_value,
                    "Kendall": kendall_value,
                })

    return pd.DataFrame(
        rows, columns=[
            "bearing",
            "Feature",
            "Target",
            "Pearson",
            "Spearman",
            "Kendall"
        ]
    )


# =============================================================================
# 4. RUL Prediction Visualization
# =============================================================================

def plot_rul_prediction(
    df: pd.DataFrame,
    bearing,
    x_col: str = "life_ratio",
    actual_col: str = "actual_RUL_sec",
    predicted_col: str = "predicted_RUL_sec",
    actual_norm_col: str = "actual_RUL_norm",
    bearing_col: str = "bearing",
    prefix_x: Iterable[float] | None = None,
    figsize: tuple = (10, 6),
    ax=None,
):
    """
    Plot actual and predicted RUL trajectory for one bearing.

    Parameters
    ----------
    df : pd.DataFrame
        Full validation prediction table after RUL reconstruction.

    bearing
        Bearing identifier to visualize.

    x_col : str, default="life_ratio"
        X-axis column. If absent, it is derived as
        1.0 - actual_RUL_norm.

    actual_col : str, default="actual_RUL_sec"
        Actual RUL in seconds.

    predicted_col : str, default="predicted_RUL_sec"
        Predicted RUL in seconds.

    actual_norm_col : str, default="actual_RUL_norm"
        Normalized actual RUL used to derive life_ratio when needed.

    bearing_col : str, default="bearing"
        Bearing identifier column.

    prefix_x : Iterable[float] or None, default=None
        Prefix marker positions expressed in the same x-axis units as
        x_col. Multiple positions are supported.

    figsize : tuple, default=(10, 6)
        Figure size used only when ax is not supplied.

    ax : matplotlib.axes.Axes or None, default=None
        Existing Axes to draw on. Supplying an Axes allows the caller to
        arrange multiple bearing plots in a shared Figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot. When ax is supplied, its parent
        Figure is returned.

    Notes
    -----
    The function visualizes the complete validation trajectory. It does not
    apply TrainingRange or Prefix Definition filtering.
    """
    required_columns = [
        bearing_col,
        actual_col,
        predicted_col,
        actual_norm_col,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required column(s) missing from prediction data: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    bearing_df = (
        df[df[bearing_col] == bearing].copy()
        .sort_values(x_col if x_col in df.columns else actual_norm_col)
    )

    if bearing_df.empty:
        raise ValueError(f"Bearing '{bearing}' not found in prediction data.")

    if x_col not in bearing_df.columns:
        bearing_df[x_col] = 1.0 - bearing_df[actual_norm_col]

    plot_columns = [x_col, actual_col, predicted_col]
    plot_df = bearing_df[plot_columns].copy()

    for column in plot_columns:
        if not pd.api.types.is_numeric_dtype(plot_df[column]):
            raise TypeError(f"Column '{column}' must be numeric.")

    if not np.isfinite(plot_df.to_numpy(dtype=float)).all():
        raise ValueError("RUL prediction data contains non-finite values.")

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(
        plot_df[x_col],
        plot_df[actual_col],
        label="Actual RUL",
    )
    ax.plot(
        plot_df[x_col],
        plot_df[predicted_col],
        label="Predicted RUL",
    )

    if prefix_x is not None:
        prefix_positions = list(prefix_x)
        if not all(np.isfinite(float(x)) for x in prefix_positions):
            raise ValueError("prefix_x contains non-finite values.")

        for index, x_position in enumerate(prefix_positions):
            ax.axvline(
                float(x_position),
                linestyle="--",
                alpha=0.7,
                label="Prefix boundary" if index == 0 else None,
            )

    ax.set_xlabel("life_ratio")
    ax.set_ylabel("RUL_sec")
    ax.set_title(f"RUL Prediction - {bearing}")
    ax.grid(True)
    ax.legend()

    if created_figure:
        fig.tight_layout()
        plt.show()

    return fig


# =============================================================================
# 5. Internal Helpers
# =============================================================================

# ----- Validation -----
def _validate_metric_input(
    df: pd.DataFrame, feature: str, bearing_col: str = "bearing"
) -> None:
    """
    Validate dataframe and feature for metric calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature to evaluate.
    bearing_col : str, default="bearing"
        Column identifying each bearing.

    Raises
    ------
    ValueError
        If the feature, bearing column, or dataframe is invalid.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found.")

    if bearing_col not in df.columns:
        raise ValueError(f"Column '{bearing_col}' not found.")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

# ----- Feature Evidence -----
def _resample_series(
    x: np.ndarray, y: np.ndarray, num_points: int = 200
):
    """
    Resample one feature onto a common normalized axis.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Feature values.
    num_points : int, default=200
        Number of resampling points.

    Returns
    -------
    grid : np.ndarray
        Common interpolation axis.
    values : np.ndarray
        Interpolated feature values.
    """
    grid = np.linspace(np.min(x), np.max(x), num_points)
    values = np.interp(grid, x, y)

    return grid, values

def _pairwise_metric(a: np.ndarray, b: np.ndarray):
    """
    Calculate absolute pairwise Spearman correlation.

    Parameters
    ----------
    a, b : np.ndarray
        Resampled feature series.

    Returns
    -------
    float
        Absolute Spearman correlation.
    """
    score, _ = spearmanr(a, b)
    return float(np.abs(score))

def _extract_failure_values(
    df: pd.DataFrame,
    feature: str,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
):
    """
    Extract feature values at the final observed time of each bearing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Feature to evaluate.
    bearing_col : str, default="bearing"
        Bearing identifier.
    time_col : str, default="elapsed_sec"
        Time column.

    Returns
    -------
    np.ndarray
        Feature values at failure/end-of-life observation.
    """
    idx = df.groupby(bearing_col)[time_col].idxmax()
    return df.loc[idx, feature].to_numpy()

def _generate_bearing_pairs(bearings):
    """
    Generate all unique bearing pairs.

    Parameters
    ----------
    bearings : Iterable
        Bearing identifiers.

    Returns
    -------
    list[tuple]
        Unique bearing pairs.
    """
    return list(combinations(bearings, 2))

def _calculate_condition_cv(
    df: pd.DataFrame,
    features: Iterable[str],
    condition_col: str = "condition",
) -> dict[str, float]:
    """
    Calculate mean coefficient of variation for each feature
    across operating conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : Iterable[str]
        Feature Candidates to evaluate.
    condition_col : str, default="condition"
        Operating-condition column.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to mean CV.
    """
    cv_map = {}
    for feature in features:
        temp = df.groupby(condition_col)[feature].agg(["mean", "std"])

        cv = temp["std"] / temp["mean"].abs()
        cv_map[feature] = float(cv.mean())

    return cv_map

# ----- Target Evidence -----
def _calculate_correlation_values(
    x: pd.Series, y: pd.Series
) -> tuple[float, float, float]:
    """
    Calculate Pearson, Spearman and Kendall correlations
    using pairwise finite observations.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.

    Returns
    -------
    tuple[float, float, float]
        Pearson, Spearman and Kendall correlation coefficients.

    Notes
    -----
    Correlation signs are preserved.
    No absolute-value conversion is performed.

    If fewer than two valid observations remain, all metrics
    are returned as NaN.
    """
    pair = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf], np.nan).dropna()

    if len(pair) < 2:
        return np.nan, np.nan, np.nan

    x_values = pair["x"].to_numpy()
    y_values = pair["y"].to_numpy()

    # Constant series do not provide meaningful correlation.
    if np.all(x_values == x_values[0]) or np.all(y_values == y_values[0]):
        return np.nan, np.nan, np.nan

    pearson_value, _ = pearsonr(x_values, y_values)
    spearman_value, _ = spearmanr(x_values, y_values)
    kendall_value, _ = kendalltau(x_values, y_values)

    return (
        float(pearson_value),
        float(spearman_value),
        float(kendall_value),
    )