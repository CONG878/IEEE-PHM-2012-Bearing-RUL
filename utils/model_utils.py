"""
model_utils.py

Modeling utilities for the PHM Bearing RUL project.

Responsibilities
----------------
- Define modeling-related data contracts.
- Create supported regression models (Algorithm Factory).
- Generate deterministic LOBO folds and train LOBO models.
- Perform LOBO model inference for Validation (Stage 05) and Test (Stage 06).
- Apply and fit experimental RUL calibration functions.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

from .project_utils import assert_required_columns, sort_by_time


# =============================================================================
# 1. Constants & Data Contracts
# =============================================================================

SUPPORTED_ALGORITHMS = ("RandomForest", "LightGBM", "XGBoost", "CatBoost")
SUPPORTED_CALIBRATION_METHODS = ("power", "rational")
SUPPORTED_CALIBRATION_OBJECTIVES = (
    "RMSE", "MAE", "SMAPE", "R2", "PHM_Score"
)
CALIBRATION_LOG10_BOUNDS = (-8.0, 8.0)

@dataclass
class ModelingConfig:
    """
    Configuration for a regression algorithm.

    Parameters
    ----------
    algorithm : str
        Supported algorithm name.

    params : dict[str, Any]
        Keyword parameters passed to the corresponding model constructor.
    """
    algorithm: str
    params: dict[str, Any]

@dataclass
class LOBOFold:
    """
    Metadata for one Leave-One-Bearing-Out fold.

    Parameters
    ----------
    fold_no : int
        Zero-based fold number.

    held_out_bearing : str
        Bearing excluded from training and used for validation.

    train_bearings : list[str]
        Bearings included in model training.
    """
    fold_no: int
    held_out_bearing: str
    train_bearings: list[str]

@dataclass
class TrainingRange:
    """
    Training interval defined by life progression ratio.
    The life progression ratio is:
        life_ratio = 1.0 - RUL_norm

    Parameters
    ----------
    start_ratio : float, default=0.0
        Inclusive minimum life progression ratio.

    end_ratio : float, default=1.0
        Inclusive maximum life progression ratio.
    """
    start_ratio: float = 0.0
    end_ratio: float = 1.0

@dataclass
class ModelBundle:
    """
    Trained LOBO model bundle.

    The positional correspondence between ``models`` and ``fold_metadata``
    is an invariant:

        models[i] <-> fold_metadata[i]

    Parameters
    ----------
    modeling_config : ModelingConfig
        Configuration used to train all models.

    feature_columns : list[str]
        Ordered feature schema used during training.

    target_column : str
        Target column used during training.

    models : list[Any]
        Ordered trained LOBO models.

    fold_metadata : list[LOBOFold]
        Ordered metadata corresponding to ``models``.
    """
    modeling_config: ModelingConfig
    training_range: TrainingRange
    feature_columns: list[str]
    target_column: str
    models: list[Any]
    fold_metadata: list[LOBOFold]


# =============================================================================
# 2. Model Factory & LOBO Fold Generation
# =============================================================================

def create_model(config: ModelingConfig):
    """
    Create a regression model from ModelingConfig.

    Supported algorithms
    --------------------
    - RandomForest
    - LightGBM
    - XGBoost
    - CatBoost

    Raises
    ------
    ValueError
        If the algorithm is unsupported.

    ImportError
        If the required optional library is not installed.
    """
    _validate_modeling_config(config)

    algorithm = config.algorithm
    params = config.params.copy()

    if algorithm == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**params)

    if algorithm == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM requires the 'lightgbm' package.") from exc

        return LGBMRegressor(**params)

    if algorithm == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost requires the 'xgboost' package.") from exc

        return XGBRegressor(**params)

    if algorithm == "CatBoost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost requires the 'catboost' package.") from exc

        return CatBoostRegressor(**params)

    # Protected by _validate_modeling_config().
    raise ValueError(f"Unsupported algorithm: '{algorithm}'.")

def create_lobo_folds(dataset: pd.DataFrame, bearing_col: str = "bearing") -> list[LOBOFold]:
    """
    Create deterministic Leave-One-Bearing-Out folds.

    Bearings are sorted alphabetically before fold generation.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input learning dataset.

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    Returns
    -------
    list[LOBOFold]
        One fold for each sorted bearing.

    Raises
    ------
    ValueError
        If the bearing column is missing or no bearing exists.
    """
    assert_required_columns(dataset, [bearing_col])

    bearings = sorted(dataset[bearing_col].dropna().unique().tolist())
    if not bearings:
        raise ValueError("Cannot create LOBO folds: no valid bearings found.")

    folds = []
    for fold_no, held_out_bearing in enumerate(bearings):
        train_bearings = [
            bearing for bearing in bearings
            if bearing != held_out_bearing
        ]

        folds.append(
            LOBOFold(
                fold_no=fold_no,
                held_out_bearing=held_out_bearing,
                train_bearings=train_bearings,
            )
        )

    return folds


# =============================================================================
# 3. LOBO Model Training & Bundle Packaging
# =============================================================================

def train_lobo_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: ModelingConfig,
    training_range: TrainingRange,
    bearing_col: str = "bearing",
) -> ModelBundle:
    """
    Train one regression model for each LOBO fold.

    For every fold:

    1. The held-out bearing is excluded completely.
    2. The configured TrainingRange is applied independently to
       every remaining training bearing.
    3. Selected rows are used to train the fold model.
    4. The trained model and Feature Importance are validated.

    Parameters
    ----------
    dataset : pd.DataFrame
        Model-ready learning dataset.

    feature_cols : list[str]
        Ordered feature columns.

    target_col : str
        Target column.

    config : ModelingConfig
        Regression algorithm configuration.

    training_range : TrainingRange
        Per-bearing training interval.

    bearing_col : str, default="bearing"
        Bearing identifier column.

    Returns
    -------
    ModelBundle
        Ordered LOBO models and corresponding fold metadata.

    Raises
    ------
    ValueError
        If required columns are missing, the training range is invalid,
        no training data exists for a fold, or the trained model output
        is invalid.
    """
    _validate_modeling_config(config)
    _validate_training_range(training_range)

    required_columns = [bearing_col, *feature_cols, target_col]
    assert_required_columns(dataset, required_columns)
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")

    folds = create_lobo_folds(dataset, bearing_col=bearing_col)
    if len(folds) < 2:
        raise ValueError("LOBO training requires at least two bearings.")

    models = []
    for fold in folds:
        fold_dataset = dataset[dataset[bearing_col].isin(fold.train_bearings)]

        if fold_dataset.empty:
            raise ValueError(f"Fold {fold.fold_no} has no candidate training rows.")

        train_df = _select_training_range(
            dataset=fold_dataset,
            training_range=training_range,
            target_col=target_col,
        )

        if train_df.empty:
            raise ValueError(
                f"Fold {fold.fold_no} has no rows after "
                "TrainingRange selection."
            )

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        if len(X_train) != len(y_train):
            raise ValueError(
                f"Fold {fold.fold_no}: X_train and y_train row count mismatch."
            )

        model = create_model(config)
        model.fit(X_train, y_train)
        if model is None:
            raise ValueError(f"Fold {fold.fold_no}: trained model is None.")

        # Validate model-dependent Feature Importance when available.
        extract_feature_importance(model, feature_cols)
        models.append(model)

    bundle = ModelBundle(
        modeling_config=config,
        training_range=training_range,
        feature_columns=list(feature_cols),
        target_column=target_col,
        models=models,
        fold_metadata=folds,
    )

    _validate_model_bundle(bundle)
    return bundle

# ----- Model Bundle -----
def build_model_bundle(
    config: ModelingConfig,
    training_range: TrainingRange,
    feature_cols: list[str],
    target_col: str,
    models: list[Any],
    fold_metadata: list[LOBOFold],
) -> ModelBundle:
    """
    Construct and validate a ModelBundle.
    """
    _validate_modeling_config(config)
    _validate_training_range(training_range)

    bundle = ModelBundle(
        modeling_config=config,
        training_range=training_range,
        feature_columns=list(feature_cols),
        target_column=target_col,
        models=list(models),
        fold_metadata=list(fold_metadata),
    )

    _validate_model_bundle(bundle)
    return bundle

# ----- Training Result Utilities -----
def extract_feature_importance(model, feature_cols: list[str]) -> pd.Series | None:
    """
    Extract model-dependent Feature Importance.

    Parameters
    ----------
    model : Any
        Trained regression model.

    feature_cols : list[str]
        Ordered feature columns used during training.

    Returns
    -------
    pd.Series or None
        Feature importance indexed by feature name.
        Returns None when the model does not expose
        ``feature_importances_``.

    Raises
    ------
    ValueError
        If the extracted importance has an invalid shape,
        length, or contains non-finite values.

    Notes
    -----
    This function provides model interpretation evidence only.
    It does not perform Feature Selection.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importance = np.asarray(model.feature_importances_, dtype=float)

    if importance.ndim != 1:
        raise ValueError("Feature importance must be one-dimensional.")

    if len(importance) != len(feature_cols):
        raise ValueError(
            "Feature importance length mismatch: "
            f"expected {len(feature_cols)}, got {len(importance)}."
        )

    if not np.isfinite(importance).all():
        raise ValueError("Feature importance contains non-finite values.")

    return pd.Series(importance, index=feature_cols, name="Importance")


# =============================================================================
# 4. Inference & Prediction Pipelines
# =============================================================================

# -------------------------------------------------------------------------
# Stage 05: Validation Predictions
# -------------------------------------------------------------------------
def predict_lobo_prefixes(
    bundle: ModelBundle,
    dataset: pd.DataFrame,
    prefix_definition: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
    end_idx_col: str = "end_idx",
) -> pd.DataFrame:
    """
    Predict one validation sample for each prefix definition row.

    Each prefix selects exactly one row from its bearing using ``end_idx``.
    The selected sample is predicted only by the LOBO model whose
    corresponding fold held out that bearing.

    Parameters
    ----------
    bundle : ModelBundle
        Trained LOBO model bundle.

    dataset : pd.DataFrame
        Model-ready learning dataset.

    prefix_definition : pd.DataFrame
        Prefix definition containing at least bearing and end_idx.

    bearing_col : str, default="bearing"
        Bearing identifier column.

    time_col : str, default="elapsed_sec"
        Elapsed-time column used to establish deterministic row order.

    end_idx_col : str, default="end_idx"
        Zero-based row position within the time-sorted bearing.

    Returns
    -------
    pd.DataFrame
        Sample-level prediction table containing prefix information,
        actual normalized RUL, and predicted normalized RUL.

    Raises
    ------
    ValueError
        If prefix definitions are invalid, a bearing has no matching
        LOBO model, or a model produces an invalid prediction.
    """
    _validate_model_bundle(bundle)

    assert_required_columns(
        dataset,
        [bearing_col, time_col, *bundle.feature_columns, bundle.target_column]
    )
    assert_required_columns(prefix_definition, [bearing_col, end_idx_col])

    dataset_bearings = set(dataset[bearing_col].dropna().unique().tolist())
    prefix_bearings = set(prefix_definition[bearing_col].dropna().unique().tolist())

    unknown_bearings = sorted(prefix_bearings - dataset_bearings)
    if unknown_bearings:
        raise ValueError(
            "Prefix definition contains bearings not found in dataset: "
            f"{unknown_bearings}"
        )

    expected_bearings = {fold.held_out_bearing for fold in bundle.fold_metadata}

    missing_prefix_bearings = sorted(expected_bearings - prefix_bearings)
    if missing_prefix_bearings:
        raise ValueError(
            "Missing prefix definitions for LOBO validation bearings: "
            f"{missing_prefix_bearings}"
        )

    ordered_dataset = sort_by_time(
        dataset, bearing_col=bearing_col, time_col=time_col
    )

    prediction_rows = []

    for prefix_row in prefix_definition.itertuples(index=False):
        prefix_data = prefix_row._asdict()

        bearing = prefix_data[bearing_col]
        end_idx = prefix_data[end_idx_col]

        if pd.isna(bearing):
            raise ValueError("Prefix definition contains a missing bearing value.")

        if pd.isna(end_idx):
            raise ValueError(f"{bearing}: '{end_idx_col}' cannot be missing.")

        if not isinstance(end_idx, int):
            if isinstance(end_idx, float) and end_idx.is_integer():
                end_idx = int(end_idx)
            else:
                raise ValueError(
                    f"{bearing}: '{end_idx_col}' must be an integer. "
                    f"Received {end_idx!r}."
                )

        bearing_df = ordered_dataset[ordered_dataset[bearing_col] == bearing]

        if end_idx < 0 or end_idx >= len(bearing_df):
            raise ValueError(
                f"{bearing}: invalid '{end_idx_col}'={end_idx}. "
                f"Valid range is [0, {len(bearing_df) - 1}]."
            )

        model_index = _get_bundle_model_index(bundle, held_out_bearing=bearing)
        model = bundle.models[model_index]

        sample = bearing_df.iloc[[end_idx]]
        X_sample = sample[bundle.feature_columns]
        raw_prediction = model.predict(X_sample)

        prediction = _validate_prediction(raw_prediction, expected_length=1)
        predicted_rul_norm = prediction[0]

        result_row = dict(prefix_data)
        result_row[time_col] = sample[time_col].iloc[0]
        result_row["actual_RUL_norm"] = sample[bundle.target_column].iloc[0]

        result_row["predicted_RUL_norm"] = predicted_rul_norm
        prediction_rows.append(result_row)

    return pd.DataFrame(prediction_rows)

def predict_lobo_validation_bearings(
    bundle: ModelBundle,
    dataset: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Predict every row of each LOBO validation bearing.

    Each bearing is predicted only by the LOBO model whose fold held that
    bearing out. This prediction is intended for full-life trajectory
    visualization and is independent of Prefix Definition evaluation.

    Parameters
    ----------
    bundle : ModelBundle
        Trained LOBO model bundle.

    dataset : pd.DataFrame
        Model-ready learning dataset containing the complete validation
        bearing trajectories.

    bearing_col : str, default="bearing"
        Bearing identifier column.

    time_col : str, default="elapsed_sec"
        Elapsed-time column used to establish deterministic ordering.

    Returns
    -------
    pd.DataFrame
        Full validation prediction table containing:

        - bearing
        - elapsed_sec
        - actual_RUL_norm
        - predicted_RUL_norm

    Raises
    ------
    ValueError
        If the dataset does not match the LOBO bundle or a prediction is
        invalid.
    """
    _validate_model_bundle(bundle)

    assert_required_columns(
        dataset,
        [bearing_col, time_col, *bundle.feature_columns, bundle.target_column]
    )

    ordered_dataset = sort_by_time(
        dataset, bearing_col=bearing_col, time_col=time_col
    )

    dataset_bearings = set(ordered_dataset[bearing_col].dropna().unique().tolist())
    expected_bearings = {fold.held_out_bearing for fold in bundle.fold_metadata}

    missing_bearings = sorted(expected_bearings - dataset_bearings)
    if missing_bearings:
        raise ValueError(
            f"LOBO validation bearings missing from dataset: {missing_bearings}"
        )

    prediction_frames = []
    for fold in bundle.fold_metadata:
        bearing = fold.held_out_bearing
        bearing_df = ordered_dataset[ordered_dataset[bearing_col] == bearing].copy()
        if bearing_df.empty:
            raise ValueError(f"No validation rows found for bearing: '{bearing}'.")

        model = bundle.models[fold.fold_no]

        X_validation = bearing_df[bundle.feature_columns]
        prediction = np.asarray(model.predict(X_validation), dtype=float)
        if prediction.ndim != 1:
            raise ValueError(
                f"Prediction for bearing '{bearing}' must be one-dimensional."
            )

        if len(prediction) != len(bearing_df):
            raise ValueError(
                f"Prediction length mismatch for bearing '{bearing}': "
                f"expected {len(bearing_df)}, got {len(prediction)}."
            )

        if not np.isfinite(prediction).all():
            raise ValueError(
                f"Prediction for bearing '{bearing}' "
                "contains non-finite values."
            )

        prediction_frames.append(
            pd.DataFrame({
                bearing_col: bearing_df[bearing_col].to_numpy(),
                time_col: bearing_df[time_col].to_numpy(),
                "actual_RUL_norm": bearing_df[bundle.target_column].to_numpy(),
                "predicted_RUL_norm": prediction,
            })
        )

    result = pd.concat(prediction_frames, ignore_index=True)
    return result.sort_values([bearing_col, time_col]).reset_index(drop=True)

# -------------------------------------------------------------------------
# Stage 06: Test Predictions
# -------------------------------------------------------------------------
def extract_test_endpoints(
    dataset: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Extract the final observed row of each Test bearing.

    The endpoint is defined as the unique row having the maximum
    elapsed time within each bearing.

    Parameters
    ----------
    dataset : pd.DataFrame
        Model-ready Test dataset.

    bearing_col : str, default="bearing"
        Column identifying each bearing.

    time_col : str, default="elapsed_sec"
        Elapsed-time column used to determine the final observation.

    Returns
    -------
    pd.DataFrame
        One endpoint row per bearing, sorted by bearing and elapsed time.

    Raises
    ------
    ValueError
        If required columns are missing, the dataset is empty, a bearing
        value is missing, elapsed time is invalid, or a bearing has
        multiple rows tied for the maximum elapsed time.
    TypeError
        If the elapsed-time column is not numeric.
    """
    assert_required_columns(dataset, [bearing_col, time_col])

    if dataset.empty:
        raise ValueError("Test dataset must not be empty.")

    if dataset[bearing_col].isna().any():
        raise ValueError(
            f"'{bearing_col}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(dataset[time_col]):
        raise TypeError(
            f"'{time_col}' must be numeric."
        )

    time_values = dataset[time_col].to_numpy(dtype=float)

    if not np.isfinite(time_values).all():
        raise ValueError(
            f"'{time_col}' contains NaN or Inf values."
        )

    if (time_values < 0).any():
        raise ValueError(
            f"'{time_col}' must contain non-negative values."
        )

    max_time_by_bearing = dataset.groupby(bearing_col)[time_col].transform("max")
    endpoint_mask = dataset[time_col].eq(max_time_by_bearing)
    endpoint_counts = endpoint_mask.groupby(dataset[bearing_col]).sum()

    duplicated_endpoints = endpoint_counts[endpoint_counts != 1]
    if not duplicated_endpoints.empty:
        raise ValueError(
            "Each bearing must have exactly one endpoint. "
            "Multiple rows share the maximum elapsed time for: "
            f"{sorted(duplicated_endpoints.index.tolist())}"
        )

    endpoints = dataset.loc[endpoint_mask].copy()
    endpoints = (
        sort_by_time(
            endpoints,
            bearing_col=bearing_col,
            time_col=time_col,
        ).reset_index(drop=True)
    )

    return endpoints

def predict_lobo_test_endpoints(
    bundle: ModelBundle,
    endpoints: pd.DataFrame,
    bearing_col: str = "bearing",
    time_col: str = "elapsed_sec",
) -> pd.DataFrame:
    """
    Predict Test endpoints with every LOBO model.

    Each Test endpoint is passed through all models contained in the
    ModelBundle. The resulting columns form an N x K prediction matrix,
    where N is the number of Test bearings and K is the number of
    ordered LOBO models.

    Parameters
    ----------
    bundle : ModelBundle
        Trained LOBO model bundle.

    endpoints : pd.DataFrame
        One final observed endpoint row per Test bearing.

    bearing_col : str, default="bearing"
        Bearing identifier column.

    time_col : str, default="elapsed_sec"
        Elapsed-time column.

    Returns
    -------
    pd.DataFrame
        Endpoint metadata followed by one prediction column per LOBO model.
        Prediction columns are named:

            predicted_RUL_norm_model_0
            predicted_RUL_norm_model_1
            ...

        The numeric suffix follows the ModelBundle model ordering.

    Raises
    ------
    ValueError
        If the endpoint schema is invalid, duplicate bearings exist,
        the number of models and fold metadata differ, or a model
        produces an invalid prediction.
    """
    _validate_model_bundle(bundle)

    assert_required_columns(
        endpoints,
        [bearing_col, time_col, *bundle.feature_columns],
    )

    if endpoints.empty:
        raise ValueError("Test endpoints must not be empty.")

    if endpoints[bearing_col].isna().any():
        raise ValueError(
            f"'{bearing_col}' contains missing values."
        )

    if endpoints[bearing_col].duplicated().any():
        duplicated_bearings = (
            endpoints.loc[
                endpoints[bearing_col].duplicated(keep=False),
                bearing_col,
            ].unique().tolist()
        )
        raise ValueError(
            "Test endpoints must contain exactly one row per bearing. "
            f"Duplicated bearings: {duplicated_bearings}"
        )

    if not pd.api.types.is_numeric_dtype(endpoints[time_col]):
        raise TypeError(f"'{time_col}' must be numeric.")

    time_values = endpoints[time_col].to_numpy(dtype=float)

    if not np.isfinite(time_values).all():
        raise ValueError(f"'{time_col}' contains NaN or Inf values.")

    if (time_values < 0).any():
        raise ValueError(f"'{time_col}' must contain non-negative values.")

    if len(bundle.models) != len(bundle.fold_metadata):
        raise ValueError(
            "ModelBundle invariant violated: "
            "len(models) must equal len(fold_metadata)."
        )

    result = endpoints.copy().reset_index(drop=True)
    X_test = result[bundle.feature_columns]
    for model_index, model in enumerate(bundle.models):
        raw_prediction = model.predict(X_test)

        prediction = _validate_prediction(
            raw_prediction, expected_length=len(result)
        )

        result[f"predicted_RUL_norm_model_{model_index}"] = prediction

    return result


# =============================================================================
# 5. Experimental RUL Post-Calibration
# =============================================================================

# -------------------------------------------------------------------------
# Calibration Fitting
# -------------------------------------------------------------------------
def fit_rul_calibration(
    y_true_sec: Iterable[float],
    predicted_norm: Iterable[float],
    method: str,
    prediction_sec_fn: Callable[[np.ndarray], np.ndarray],
    objective_fn: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool,
    log10_bounds: tuple[float, float] = CALIBRATION_LOG10_BOUNDS,
) -> float:
    """
    Fit one boundary-preserving calibration parameter.

    Optimization is performed in log10(parameter) space so that the
    parameter remains positive while allowing a deliberately broad search.
    The objective is evaluated in RUL_sec space.

    Parameters
    ----------
    y_true_sec : array-like
        Actual RUL values in seconds.
    predicted_norm : array-like
        Raw predicted RUL_norm values.
    method : str
        Calibration method: "power" or "rational".
    prediction_sec_fn : callable
        Function converting calibrated normalized predictions to RUL_sec.
    objective_fn : callable
        Function receiving ``(y_true_sec, y_pred_sec)`` and returning the
        scalar objective value.
    maximize : bool
        Whether a larger objective value is better.
    log10_bounds : tuple[float, float], default=(-8, 8)
        Search bounds in log10(parameter) space.

    Returns
    -------
    float
        Fitted positive calibration parameter.
    """
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method: {method!r}. "
            f"Expected one of {SUPPORTED_CALIBRATION_METHODS}."
        )

    y_true = np.asarray(y_true_sec, dtype=float)
    raw = np.asarray(predicted_norm, dtype=float)

    if y_true.ndim != 1 or raw.ndim != 1:
        raise ValueError("Calibration inputs must be one-dimensional.")
    if len(y_true) != len(raw):
        raise ValueError("Calibration input lengths must match.")
    if len(raw) == 0:
        raise ValueError("Calibration inputs must not be empty.")
    if not np.isfinite(y_true).all():
        raise ValueError("y_true_sec contains NaN or Inf values.")
    if not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError(
            "predicted_norm must contain finite values within [0, 1]."
        )
    if not callable(prediction_sec_fn):
        raise TypeError("prediction_sec_fn must be callable.")
    if not callable(objective_fn):
        raise TypeError("objective_fn must be callable.")

    lower, upper = map(float, log10_bounds)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("log10_bounds must be finite and strictly increasing.")

    def objective(log10_parameter: float) -> float:
        parameter = 10.0 ** log10_parameter
        calibrated_norm = apply_rul_calibration(
            raw, method=method, parameter=parameter
        )
        predicted_sec = np.asarray(
            prediction_sec_fn(calibrated_norm), dtype=float
        )
        if predicted_sec.ndim != 1 or len(predicted_sec) != len(y_true):
            raise ValueError(
                "prediction_sec_fn must return a one-dimensional array "
                "with the same length as y_true_sec."
            )
        if not np.isfinite(predicted_sec).all():
            return np.inf
        score = float(objective_fn(y_true, predicted_sec))
        if not np.isfinite(score):
            return np.inf
        return -score if maximize else score

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10},
    )

    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(
            f"Calibration optimization failed for method '{method}': "
            f"{result.message}"
        )

    parameter = 10.0 ** float(result.x)
    if not np.isfinite(parameter) or parameter <= 0.0:
        raise RuntimeError("Calibration optimization produced an invalid parameter.")

    return float(parameter)

# -------------------------------------------------------------------------
# Calibration Application
# -------------------------------------------------------------------------
def apply_rul_calibration(
    prediction: Iterable[float],
    method: str,
    parameter: float,
) -> np.ndarray:
    """
    Apply a boundary-preserving normalized-RUL calibration.

    Parameters
    ----------
    prediction : array-like
        Predicted RUL_norm values.
    method : str
        Calibration method: "power" or "rational".
    parameter : float
        Power exponent ``gamma`` or rational coefficient ``a``.

    Returns
    -------
    np.ndarray
        Calibrated RUL_norm values.

    Raises
    ------
    ValueError
        If the method or parameter is invalid, or predictions are outside
        the normalized RUL domain.
    """
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method: {method!r}. "
            f"Expected one of {SUPPORTED_CALIBRATION_METHODS}."
        )

    if not np.isfinite(parameter) or parameter <= 0.0:
        raise ValueError("Calibration parameter must be finite and positive.")

    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1:
        raise ValueError("prediction must be one-dimensional.")
    if len(values) == 0:
        raise ValueError("prediction must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError("prediction contains NaN or Inf values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            "Calibration requires predicted RUL_norm values within [0, 1]."
        )

    if method == "power":
        calibrated = np.power(values, parameter)
    else:
        denominator = 1.0 + (parameter - 1.0) * values
        calibrated = parameter * values / denominator

    if not np.isfinite(calibrated).all():
        raise ValueError("Calibration produced NaN or Inf values.")

    return calibrated


# =============================================================================
# 6. Internal Validation & Helper Functions
# =============================================================================

# -------------------------------------------------------------------------
# Configuration & Bundle Validation
# -------------------------------------------------------------------------
def _validate_modeling_config(config: ModelingConfig) -> None:
    """
    Validate ModelingConfig.

    Raises
    ------
    TypeError
        If config is not a ModelingConfig instance.

    ValueError
        If the algorithm is unsupported.
    """
    if not isinstance(config, ModelingConfig):
        raise TypeError("config must be an instance of ModelingConfig.")

    if config.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: '{config.algorithm}'. "
            f"Supported algorithms: {list(SUPPORTED_ALGORITHMS)}"
        )

    if not isinstance(config.params, dict):
        raise TypeError("ModelingConfig.params must be a dictionary.")

def _validate_model_bundle(bundle: ModelBundle) -> None:
    """
    Validate ModelBundle invariants.

    Raises
    ------
    TypeError
        If bundle is not a ModelBundle instance.

    ValueError
        If model/fold correspondence is invalid.
    """
    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be an instance of ModelBundle.")

    if len(bundle.models) != len(bundle.fold_metadata):
        raise ValueError(
            "ModelBundle invariant violated: "
            "len(models) must equal len(fold_metadata)."
        )

    if not bundle.models:
        raise ValueError("ModelBundle contains no trained models.")

def _validate_training_range(training_range: TrainingRange) -> None:
    """
    Validate TrainingRange.

    Raises
    ------
    TypeError
        If training_range is not a TrainingRange instance.

    ValueError
        If range ratios are outside [0.0, 1.0] or
        start_ratio is greater than end_ratio.
    """
    if not isinstance(training_range, TrainingRange):
        raise TypeError(
            "training_range must be an instance of TrainingRange."
        )

    start_ratio = training_range.start_ratio
    end_ratio = training_range.end_ratio

    if not isinstance(start_ratio, (int, float)):
        raise TypeError(
            "TrainingRange.start_ratio must be numeric."
        )

    if not isinstance(end_ratio, (int, float)):
        raise TypeError(
            "TrainingRange.end_ratio must be numeric."
        )

    if not 0.0 <= start_ratio <= 1.0:
        raise ValueError(
            "TrainingRange.start_ratio must be within [0.0, 1.0]."
        )

    if not 0.0 <= end_ratio <= 1.0:
        raise ValueError(
            "TrainingRange.end_ratio must be within [0.0, 1.0]."
        )

    if start_ratio > end_ratio:
        raise ValueError(
            "TrainingRange.start_ratio must not exceed end_ratio."
        )

# -------------------------------------------------------------------------
# Prediction Validation
# -------------------------------------------------------------------------
def _validate_prediction(prediction, expected_length: int) -> np.ndarray:
    """
    Validate a model prediction array.

    Parameters
    ----------
    prediction : Any
        Raw output returned by ``model.predict()``.

    expected_length : int
        Expected number of predictions.

    Returns
    -------
    np.ndarray
        Validated one-dimensional float array.

    Raises
    ------
    ValueError
        If the prediction has an invalid shape, length, dtype,
        or contains NaN / Inf.
    """
    prediction_array = np.asarray(prediction, dtype=float)

    if prediction_array.ndim != 1:
        raise ValueError("Prediction must be one-dimensional.")

    if len(prediction_array) != expected_length:
        raise ValueError(
            "Prediction length mismatch: "
            f"expected {expected_length}, got {len(prediction_array)}."
        )

    if not np.isfinite(prediction_array).all():
        raise ValueError("Prediction contains non-finite values.")

    return prediction_array

# -------------------------------------------------------------------------
# Training & Bundle Helpers
# -------------------------------------------------------------------------
def _select_training_range(
    dataset: pd.DataFrame,
    training_range: TrainingRange,
    target_col: str,
) -> pd.DataFrame:
    """
    Select rows within the configured life progression interval.
    Supported target interpretations
    --------------------------------
    RUL_norm
        life_ratio = 1.0 - RUL_norm

    life_ratio
        life_ratio = life_ratio
    """
    _validate_training_range(training_range)

    if target_col not in dataset.columns:
        raise KeyError(f"Target column not found in dataset: {target_col}")

    if target_col == "RUL_norm":
        life_ratio = 1.0 - dataset[target_col]

    elif target_col == "life_ratio":
        life_ratio = dataset[target_col]

    else:
        raise ValueError(
            "TrainingRange supports only 'RUL_norm' or "
            f"'life_ratio' as target_col, got: {target_col!r}"
        )

    selection_mask = (
        (life_ratio >= training_range.start_ratio)
        & (life_ratio <= training_range.end_ratio)
    )

    selected = dataset.loc[selection_mask].copy()
    if selected.empty:
        raise ValueError("Training range selection produced no rows.")

    return selected

def _get_bundle_model_index(bundle: ModelBundle, held_out_bearing: str) -> int:
    """
    Find the model index corresponding to a held-out bearing.

    Raises
    ------
    ValueError
        If no matching LOBO fold exists.
    """
    matches = [
        index for index, fold in enumerate(bundle.fold_metadata)
        if fold.held_out_bearing == held_out_bearing
    ]

    if not matches:
        raise ValueError(f"No LOBO model found for bearing: '{held_out_bearing}'.")

    if len(matches) > 1:
        raise ValueError(
            f"Multiple LOBO folds found for bearing: '{held_out_bearing}'."
        )

    return matches[0]