"""
project_utils.py

Common utility functions shared across the entire PHM Bearing RUL project.

Author
------
Bearing RUL Project

Notes
-----
This module intentionally depends only on NumPy and Pandas.
Visualization, statistics and EDA-specific functions belong to eda_utils.py.
"""

# =============================================================================
# Imports
# =============================================================================

from typing import Iterable, List, Tuple
import pandas as pd


# =============================================================================
# 1. DataFrame Utilities
# =============================================================================

# ----- Feature -----
def split_columns(
    df: pd.DataFrame, exclude_columns: [Iterable[str]]
) -> Tuple[List[str], List[str]]:
    """
    Split dataframe columns into excluded and remaining columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    exclude_columns : Iterable[str]

    Returns
    -------
    excluded_columns : list[str]
    remaining_columns : list[str]
    """
    excluded_columns = [c for c in exclude_columns if c in df.columns]
    remaining_columns = [
        c for c in df.columns
        if c not in excluded_columns
    ]
    
    return excluded_columns, remaining_columns

# ----- Time -----
def sort_by_time(
    df: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Sort dataframe by bearing and elapsed time.

    Returns
    -------
    pd.DataFrame
    """
    return df.sort_values([bearing_col, time_col]).reset_index(drop=True)


# =============================================================================
# 2. Validation Utilities
# =============================================================================

# ----- Schema -----
def assert_required_columns(
    df: pd.DataFrame, required_columns: Iterable[str]
) -> None:
    """
    Validate required columns.

    Raises
    ------
    ValueError
    """
    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ----- Ordering -----
def assert_monotonic_order(
    df: pd.DataFrame,
    group_col: str = "bearing",
    order_col: str = "elapsed_sec",
    increasing: bool = True,
) -> None:
    """
    Verify elapsed time is monotonically
    increasing/decreasing inside every bearing.
    """
    for group, sub_df in df.groupby(group_col):
        ordered = sub_df[order_col]
        if increasing:
            valid = ordered.is_monotonic_increasing
            direction = "increasing"
        else:
            valid = ordered.is_monotonic_decreasing
            direction = "decreasing"
        
        if not valid:
            raise ValueError(f"{group}: '{order_col}' is not monotonic {direction}.")