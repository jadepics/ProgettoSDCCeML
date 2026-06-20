from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CLASSIFICATION_SCENARIOS: dict[str, dict[str, list[str] | None]] = {
    "baseline_original": {
        "drop_columns": [],
        "keep_columns": None,
    },
    "baseline_no_leakage": {
        "drop_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "keep_columns": None,
    },
    "no_diagnostic_features": {
        "drop_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
        ],
        "keep_columns": None,
    },
    "no_diagnostic_extended": {
        "drop_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
            "insulin_level",
        ],
        "keep_columns": None,
    },
    "clinical_only": {
        "drop_columns": [],
        "keep_columns": [
            "family_history_diabetes",
            "hypertension_history",
            "cardiovascular_history",
            "bmi",
            "waist_to_hip_ratio",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "cholesterol_total",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
            "insulin_level",
        ],
    },
    "glucose_only": {
        "drop_columns": [],
        "keep_columns": [
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
        ],
    },
}


@dataclass
class LocalBaselineConfig:
    dataset_url: str
    target_column: str
    task_type: str

    dataset_scenario: str = "baseline_original"
    extra_drop_columns: list[str] | None = None

    n_estimators: int = 48
    max_depth: int | None = 5
    max_features: str | float | None = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    classification_criterion: str = "gini"
    regression_criterion: str = "squared_error"

    bootstrap: bool = True
    class_weight: str | None = None

    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42

    n_jobs: int = 1
    output_json: str | None = None

    # NUOVO: salvataggio modello locale
    save_model: bool = True
    model_output_path: str | None = None

    # NUOVO: salvataggio degli split usati dalla baseline locale
    save_split_artifacts: bool = True
    split_artifacts_dir: str | None = None

@dataclass
class LocalSavedInferenceConfig:
    model_path: str
    features_path: str
    task_type: str

    labels_path: str | None = None
    split_name: str = "test"
    rows: int | None = None
    output_json: str | None = None

def resolve_input_path(path_value: str | Path) -> Path:
    """
    Input:
    - se assoluto, usa il path direttamente;
    - se relativo, prova prima dalla working directory;
    - se non esiste, prova dalla root del progetto.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    return project_candidate


def resolve_output_path(path_value: str | Path) -> Path:
    """
    Output:
    - se assoluto, usa il path direttamente;
    - se relativo, salva sempre rispetto alla root del progetto.

    Questo rende il comportamento stabile anche se la CLI viene lanciata da IDE.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

def sanitize_filename(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def default_local_model_path(config: LocalBaselineConfig) -> Path:
    filename = (
        f"local_{config.task_type}_"
        f"{sanitize_filename(config.dataset_scenario)}_"
        f"{sanitize_filename(config.target_column)}_"
        f"{config.n_estimators}_trees.joblib"
    )

    return PROJECT_ROOT / "local_baseline" / "models" / filename


def default_split_artifacts_dir(config: LocalBaselineConfig) -> Path:
    dirname = (
        f"local_{config.task_type}_"
        f"{sanitize_filename(config.dataset_scenario)}_"
        f"{sanitize_filename(config.target_column)}_"
        f"{config.n_estimators}_trees"
    )

    return PROJECT_ROOT / "local_baseline" / "prepared_splits" / dirname


def save_series_as_parquet(series: pd.Series, path: Path, column_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df = series.reset_index(drop=True).to_frame(name=column_name)
    df.to_parquet(path, index=False)


def save_dataframe_as_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)

def make_json_safe(value: Any) -> Any:
    """
    Converte tipi NumPy/Pandas in tipi Python standard serializzabili in JSON.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(make_json_safe(key)): make_json_safe(inner_value)
            for key, inner_value in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def load_dataset(path_value: str | Path) -> pd.DataFrame:
    resolved_path = resolve_input_path(path_value)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path_value}. "
            f"Resolved path: {resolved_path}"
        )

    suffix = resolved_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(resolved_path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved_path)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def apply_dataset_scenario(
    df: pd.DataFrame,
    target_column: str,
    scenario: str,
    extra_drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    scenario_config = CLASSIFICATION_SCENARIOS.get(
        scenario,
        {
            "drop_columns": [],
            "keep_columns": None,
        },
    )

    keep_columns = scenario_config.get("keep_columns")
    drop_columns = list(scenario_config.get("drop_columns") or [])
    drop_columns.extend(extra_drop_columns)

    drop_set = set(drop_columns)
    drop_set.add(target_column)

    y = df[target_column].copy()

    if keep_columns is not None:
        selected_features = [
            col
            for col in keep_columns
            if col in df.columns
            and col != target_column
            and col not in drop_set
        ]

        if not selected_features:
            raise ValueError(
                f"Scenario '{scenario}' produced no usable features"
            )

        X = df[selected_features].copy()
        return X, y, selected_features

    existing_drop_columns = [
        col for col in drop_set
        if col in df.columns
    ]

    X = df.drop(columns=existing_drop_columns).copy()

    if X.empty or len(X.columns) == 0:
        raise ValueError(
            f"Scenario '{scenario}' produced no usable features"
        )

    return X, y, list(X.columns)


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Compatibilità sklearn:
    - sklearn >= 1.2 usa sparse_output
    - sklearn < 1.2 usa sparse
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessing locale allineato il più possibile al distribuito.

    Numeriche:
    - imputazione median.

    Categoriche:
    - OneHotEncoder diretto, senza imputazione most_frequent.
      In questo modo i NaN possono diventare categoria separata,
      come visto negli artifact distribuiti: gender_nan, ethnicity_nan, ecc.
    """
    numeric_columns = [
        col
        for col in X.columns
        if is_numeric_dtype(X[col]) and not is_bool_dtype(X[col])
    ]

    categorical_columns = [
        col
        for col in X.columns
        if col not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("onehot", make_one_hot_encoder()),
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable feature columns found")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_model(config: LocalBaselineConfig):
    if config.task_type == "classification":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_features=config.max_features,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            criterion=config.classification_criterion,
            bootstrap=config.bootstrap,
            class_weight=config.class_weight,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
        )

    if config.task_type == "regression":
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_features=config.max_features,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            criterion=config.regression_criterion,
            bootstrap=config.bootstrap,
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
        )

    raise ValueError(f"Unsupported task_type: {config.task_type}")


def validate_target_for_task(
    task_type: str,
    y: pd.Series,
) -> pd.Series:
    if task_type == "classification":
        return y

    if task_type == "regression":
        numeric_y = pd.to_numeric(y, errors="coerce")

        if numeric_y.isna().any():
            raise ValueError(
                "Regression target contains non-numeric or missing values "
                "after conversion"
            )

        return numeric_y

    raise ValueError(f"Unsupported task_type: {task_type}")


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    if task_type == "classification":
        labels = sorted(pd.Series(y_true).dropna().unique().tolist())

        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
            ).tolist(),
            "labels": labels,
        }

        return make_json_safe(result)

    mse = mean_squared_error(y_true, y_pred)

    result = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }

    return make_json_safe(result)


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    try:
        names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return []


def run_local_baseline(config: LocalBaselineConfig) -> dict[str, Any]:
    total_start = time.perf_counter()

    dataset_path = resolve_input_path(config.dataset_url)
    df = load_dataset(dataset_path)

    extra_drop_columns = config.extra_drop_columns or []

    X, y, feature_names = apply_dataset_scenario(
        df=df,
        target_column=config.target_column,
        scenario=config.dataset_scenario,
        extra_drop_columns=extra_drop_columns,
    )

    y = validate_target_for_task(
        task_type=config.task_type,
        y=y,
    )

    stratify_y = y if config.task_type == "classification" else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        stratify=stratify_y,
    )

    validation_relative_ratio = (
        config.validation_ratio / (1.0 - config.test_ratio)
    )

    stratify_train_val = (
        y_train_val if config.task_type == "classification" else None
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_relative_ratio,
        random_state=config.random_seed,
        stratify=stratify_train_val,
    )

    preprocessing = build_preprocessing_pipeline(X_train)
    model = build_model(config)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    training_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - training_start

    transformed_feature_names = get_transformed_feature_names(pipeline)
    n_features_after_preprocessing = len(transformed_feature_names)

    validation_inference_start = time.perf_counter()
    validation_predictions = pipeline.predict(X_validation)
    validation_inference_time_seconds = (
        time.perf_counter() - validation_inference_start
    )

    test_inference_start = time.perf_counter()
    test_predictions = pipeline.predict(X_test)
    test_inference_time_seconds = time.perf_counter() - test_inference_start

    validation_metrics = compute_metrics(
        task_type=config.task_type,
        y_true=np.asarray(y_validation),
        y_pred=np.asarray(validation_predictions),
    )

    test_metrics = compute_metrics(
        task_type=config.task_type,
        y_true=np.asarray(y_test),
        y_pred=np.asarray(test_predictions),
    )

    model_path: Path | None = None

    if config.save_model:
        if config.model_output_path:
            model_path = resolve_output_path(config.model_output_path)
        else:
            model_path = default_local_model_path(config)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

    split_dir: Path | None = None
    split_paths: dict[str, str] = {}

    if config.save_split_artifacts:
        if config.split_artifacts_dir:
            split_dir = resolve_output_path(config.split_artifacts_dir)
        else:
            split_dir = default_split_artifacts_dir(config)

        split_dir.mkdir(parents=True, exist_ok=True)

        train_features_path = split_dir / "train_features.parquet"
        train_labels_path = split_dir / "train_labels.parquet"
        validation_features_path = split_dir / "validation_features.parquet"
        validation_labels_path = split_dir / "validation_labels.parquet"
        test_features_path = split_dir / "test_features.parquet"
        test_labels_path = split_dir / "test_labels.parquet"

        save_dataframe_as_parquet(X_train, train_features_path)
        save_series_as_parquet(y_train, train_labels_path, config.target_column)

        save_dataframe_as_parquet(X_validation, validation_features_path)
        save_series_as_parquet(
            y_validation,
            validation_labels_path,
            config.target_column,
        )

        save_dataframe_as_parquet(X_test, test_features_path)
        save_series_as_parquet(y_test, test_labels_path, config.target_column)

        split_paths = {
            "split_artifacts_dir": str(split_dir),
            "train_features_path": str(train_features_path),
            "train_labels_path": str(train_labels_path),
            "validation_features_path": str(validation_features_path),
            "validation_labels_path": str(validation_labels_path),
            "test_features_path": str(test_features_path),
            "test_labels_path": str(test_labels_path),
        }

    result = {
        "mode": "local_non_distributed",
        "dataset_path_resolved": str(dataset_path),
        "config": asdict(config),

        "local_model_path": str(model_path) if model_path is not None else None,

        "n_samples_total": int(len(df)),
        "n_features_input": int(len(feature_names)),
        "feature_names_input": feature_names,

        "n_features_after_preprocessing": int(
            n_features_after_preprocessing
        ),
        "feature_names_after_preprocessing": transformed_feature_names,

        "n_train": int(len(X_train)),
        "n_validation": int(len(X_validation)),
        "n_test": int(len(X_test)),

        "training_time_seconds": float(training_time_seconds),
        "validation_inference_time_seconds": float(
            validation_inference_time_seconds
        ),
        "test_inference_time_seconds": float(test_inference_time_seconds),
        "total_time_seconds": float(time.perf_counter() - total_start),

        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,

        "split_artifacts": split_paths,
    }

    result = make_json_safe(result)

    if config.output_json:
        output_path = resolve_output_path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    return result

def run_saved_local_inference(
    config: LocalSavedInferenceConfig,
) -> dict[str, Any]:
    model_path = resolve_input_path(config.model_path)
    features_path = resolve_input_path(config.features_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    pipeline = joblib.load(model_path)

    X = pd.read_parquet(features_path)

    if config.rows is not None:
        if config.rows <= 0:
            raise ValueError("rows must be positive or None")
        X = X.head(config.rows)

    X = X.reset_index(drop=True)

    inference_start = time.perf_counter()
    predictions = pipeline.predict(X)
    inference_time_seconds = time.perf_counter() - inference_start

    labels_path = (
        resolve_input_path(config.labels_path)
        if config.labels_path is not None
        else None
    )

    metrics = None

    if labels_path is not None:
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        y_df = pd.read_parquet(labels_path)

        if config.rows is not None:
            y_df = y_df.head(config.rows)

        y_true = y_df.to_numpy().reshape(-1)

        metrics = compute_metrics(
            task_type=config.task_type,
            y_true=np.asarray(y_true),
            y_pred=np.asarray(predictions),
        )

    split_name = str(config.split_name).strip().lower() or "test"

    result = {
        "mode": "local_saved_model_inference",
        "model_path": str(model_path),
        "features_path": str(features_path),
        "labels_path": str(labels_path) if labels_path is not None else None,
        "task_type": config.task_type,
        "split": split_name,
        "requested_rows": config.rows,
        "n_rows": int(len(X)),
        "inference_time_seconds": float(inference_time_seconds),

        # Campi compatibili con distributed_inference_benchmark.py
        f"{split_name}_inference_time_seconds": float(
            inference_time_seconds
        ),
        f"{split_name}_metrics": metrics,

        "predictions_preview": make_json_safe(
            np.asarray(predictions).reshape(-1)[:20]
        ),
        "created_at": time.time(),
    }

    result = make_json_safe(result)

    if config.output_json:
        output_path = resolve_output_path(config.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    return result