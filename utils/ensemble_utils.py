"""
Ensemble utilities for the PHM Bearing RUL project.

Responsibilities
----------------
- Validate an N x K prediction matrix.
- Validate EnsembleConfig.
- Combine ordered LOBO predictions within one ModelingConfig.
- Support arithmetic and harmonic mean.
- Support unweighted and weighted ensemble.

This module does not:
- train models
- perform LOBO splitting
- reconstruct RUL_sec
- evaluate predictions
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """
    Configuration for intra-config LOBO ensemble.

    Parameters
    ----------
    method : str, default="arithmetic"
        Ensemble method.

        Supported values:
        - "arithmetic"
        - "harmonic"

    weights : list[float] or None, default=None
        Optional model weights.

        The order must exactly match the ordered LOBO models in the
        ModelBundle.
    """

    method: str = "arithmetic"
    weights: list[float] | None = None


# =============================================================================
# 2. Internal Validation
# =============================================================================

def _validate_prediction_matrix(prediction_matrix) -> np.ndarray:
    """
    Validate an N x K prediction matrix.

    Parameters
    ----------
    prediction_matrix : array-like
        Matrix whose rows are Test bearings and whose columns are
        ordered LOBO model predictions.

    Returns
    -------
    np.ndarray
        Validated two-dimensional float array.
    """
    matrix = np.asarray(prediction_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Prediction matrix must be two-dimensional.")

    n_samples, n_models = matrix.shape
    if n_samples == 0:
        raise ValueError("Prediction matrix must contain at least one sample.")

    if n_models == 0:
        raise ValueError("Prediction matrix must contain at least one model.")

    if not np.isfinite(matrix).all():
        raise ValueError("Prediction matrix contains NaN or Inf values.")

    return matrix

def _validate_ensemble_config(config: EnsembleConfig, n_models: int):
    """
    Validate EnsembleConfig against the number of LOBO models.
    """
    if not isinstance(config, EnsembleConfig):
        raise TypeError("config must be an EnsembleConfig instance.")

    if config.method not in {"arithmetic", "harmonic"}:
        raise ValueError(
            "Unsupported ensemble method: "
            f"{config.method!r}. "
            "Expected 'arithmetic' or 'harmonic'."
        )

    if config.weights is None:
        return

    weights = np.asarray(config.weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Ensemble weights must be one-dimensional.")

    if len(weights) != n_models:
        raise ValueError(
            "Ensemble weight length mismatch: "
            f"expected {n_models}, got {len(weights)}."
        )

    if not np.isfinite(weights).all():
        raise ValueError("Ensemble weights contain NaN or Inf values.")

    if np.any(weights < 0):
        raise ValueError("Ensemble weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Ensemble weights must have a positive sum.")


# =============================================================================
# 3. Ensemble
# =============================================================================

def ensemble_predictions(prediction_matrix, config: EnsembleConfig) -> np.ndarray:
    """
    Ensemble ordered LOBO predictions into one prediction per sample.

    Parameters
    ----------
    prediction_matrix : array-like
        N x K matrix.

        - N: number of Test bearings
        - K: number of ordered LOBO models

    config : EnsembleConfig
        Ensemble method and optional model weights.

    Returns
    -------
    np.ndarray
        One-dimensional array of length N containing the ensembled
        RUL_norm prediction for each Test bearing.

    Notes
    -----
    Arithmetic ensemble:

        unweighted:
            mean(predictions)

        weighted:
            sum(w_k * prediction_k) / sum(w_k)

    Harmonic ensemble:

        unweighted:
            K / sum(1 / prediction_k)

        weighted:
            sum(w_k) / sum(w_k / prediction_k)

    Harmonic ensemble requires every prediction to be strictly positive.
    """
    matrix = _validate_prediction_matrix(prediction_matrix)
    n_models = matrix.shape[1]
    _validate_ensemble_config(config, n_models=n_models)

    weights = (
        None if config.weights is None
        else np.asarray(config.weights, dtype=float)
    )

    if config.method == "arithmetic":
        if weights is None:
            prediction = np.mean(matrix, axis=1)
        else:
            prediction = (
                np.sum(matrix * weights[np.newaxis, :], axis=1)
                / np.sum(weights)
            )

    elif config.method == "harmonic":
        if np.any(matrix <= 0):
            raise ValueError(
                "Harmonic ensemble requires all predictions "
                "to be strictly positive."
            )

        if weights is None:
            prediction = n_models / np.sum(1.0 / matrix, axis=1)
        else:
            prediction = (
                np.sum(weights)
                / np.sum(weights[np.newaxis, :] / matrix, axis=1)
            )

    else:
        # Defensive guard; method validity is already checked above.
        raise ValueError(f"Unsupported ensemble method: {config.method!r}.")

    if not np.isfinite(prediction).all():
        raise ValueError("Ensemble prediction contains NaN or Inf values.")

    return prediction