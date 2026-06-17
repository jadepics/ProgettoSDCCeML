from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def select_experiment_record_path(job_dir: Path, job_record: dict[str, Any]) -> Path:
    selected_experiment_id = job_record.get("selected_experiment_id")

    if selected_experiment_id:
        candidate = (
            job_dir
            / "experiments"
            / selected_experiment_id
            / "experiment_record.json"
        )

        if candidate.exists():
            return candidate

    experiment_records = sorted(
        job_dir.glob("experiments/*/experiment_record.json")
    )

    if not experiment_records:
        raise FileNotFoundError(
            f"No experiment_record.json found under {job_dir / 'experiments'}"
        )

    completed_records: list[Path] = []

    for path in experiment_records:
        try:
            data = load_json(path)
        except Exception:
            continue

        if data.get("status") == "COMPLETED":
            completed_records.append(path)

    if completed_records:
        return completed_records[0]

    return experiment_records[0]


def read_tree_metadata(experiment_dir: Path) -> list[dict[str, Any]]:
    tree_records: list[dict[str, Any]] = []

    for path in sorted((experiment_dir / "trees").glob("tree_*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue

        tree_records.append(data)

    return tree_records


def summarize_trees(experiment_dir: Path) -> dict[str, Any]:
    tree_records = read_tree_metadata(experiment_dir)

    saved_tree_json_count = len(tree_records)
    saved_tree_joblib_count = len(
        list((experiment_dir / "trees").glob("tree_*.joblib"))
    )

    worker_counter: Counter[str] = Counter()
    training_times: list[float] = []

    completed_tree_count_from_metadata = 0

    for tree in tree_records:
        if tree.get("status") == "COMPLETED":
            completed_tree_count_from_metadata += 1

        worker_id = tree.get("worker_id")

        if worker_id:
            worker_counter[str(worker_id)] += 1

        training_time = safe_float(tree.get("training_time_seconds"))

        if training_time is not None:
            training_times.append(training_time)

    if training_times:
        tree_training_time_sum_seconds = sum(training_times)
        tree_training_time_avg_seconds = (
            tree_training_time_sum_seconds / len(training_times)
        )
        tree_training_time_max_seconds = max(training_times)
        tree_training_time_min_seconds = min(training_times)
    else:
        tree_training_time_sum_seconds = None
        tree_training_time_avg_seconds = None
        tree_training_time_max_seconds = None
        tree_training_time_min_seconds = None

    return {
        "saved_tree_json_count": saved_tree_json_count,
        "saved_tree_joblib_count": saved_tree_joblib_count,
        "completed_tree_count_from_metadata": completed_tree_count_from_metadata,
        "tree_distribution_by_worker": dict(sorted(worker_counter.items())),
        "tree_training_time_sum_seconds": tree_training_time_sum_seconds,
        "tree_training_time_avg_seconds": tree_training_time_avg_seconds,
        "tree_training_time_max_seconds": tree_training_time_max_seconds,
        "tree_training_time_min_seconds": tree_training_time_min_seconds,
    }


def extract_metric_value(metrics: dict[str, Any], metric_name: str) -> float | None:
    value = metrics.get(metric_name)

    if value is None:
        return None

    return safe_float(value)


def summarize_validation_metrics(
    validation_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    if not validation_metrics:
        return {
            "validation_accuracy": None,
            "validation_r2": None,
            "validation_mae": None,
            "validation_mse": None,
            "validation_rmse": None,
            "validation_confusion_matrix": None,
            "validation_classification_report": None,
        }

    return {
        "validation_accuracy": extract_metric_value(
            validation_metrics,
            "accuracy",
        ),
        "validation_r2": extract_metric_value(
            validation_metrics,
            "r2",
        ),
        "validation_mae": extract_metric_value(
            validation_metrics,
            "mae",
        ),
        "validation_mse": extract_metric_value(
            validation_metrics,
            "mse",
        ),
        "validation_rmse": extract_metric_value(
            validation_metrics,
            "rmse",
        ),
        "validation_confusion_matrix": validation_metrics.get(
            "confusion_matrix"
        ),
        "validation_classification_report": validation_metrics.get(
            "classification_report"
        ),
    }


def extract_distributed_job_summary(job_dir_value: str | Path) -> dict[str, Any]:
    job_dir = Path(job_dir_value)

    if not job_dir.exists():
        raise FileNotFoundError(f"Distributed job directory not found: {job_dir}")

    job_record_path = job_dir / "job_record.json"
    job_record = load_json(job_record_path)

    experiment_record_path = select_experiment_record_path(
        job_dir=job_dir,
        job_record=job_record,
    )

    experiment_record = load_json(experiment_record_path)
    experiment_dir = experiment_record_path.parent

    forest_config = experiment_record.get("forest_config") or {}
    validation_metrics = experiment_record.get("validation_metrics") or {}

    training_request = job_record.get("training_request") or {}
    prepared_dataset = job_record.get("prepared_dataset") or {}
    prepared_schema = prepared_dataset.get("schema") or {}

    created_at = safe_float(job_record.get("created_at"))
    updated_at = safe_float(job_record.get("updated_at"))

    if created_at is not None and updated_at is not None:
        training_wall_time_seconds = max(0.0, updated_at - created_at)
    else:
        training_wall_time_seconds = None

    assigned_workers = experiment_record.get("assigned_workers") or []

    tree_summary = summarize_trees(experiment_dir)
    metric_summary = summarize_validation_metrics(validation_metrics)

    n_estimators = (
        forest_config.get("n_estimators")
        or training_request.get("n_estimators_total")
        or experiment_record.get("expected_tree_count")
    )

    summary = {
        "source": "distributed_job",
        "job_dir": str(job_dir),
        "job_id": job_record.get("job_id") or job_dir.name,
        "status": job_record.get("status"),
        "message": job_record.get("message"),
        "model_id": job_record.get("model_id"),
        "selected_experiment_id": job_record.get("selected_experiment_id"),
        "experiment_id": experiment_record.get("experiment_id"),
        "experiment_status": experiment_record.get("status"),

        "created_at": created_at,
        "updated_at": updated_at,
        "training_wall_time_seconds": training_wall_time_seconds,

        "dataset_uri": (
            training_request.get("dataset_uri")
            or prepared_schema.get("dataset_uri")
        ),
        "dataset_scenario": (
            training_request.get("dataset_scenario")
            or get_nested(
                prepared_dataset,
                ["preparation_metadata", "dataset_scenario"],
            )
        ),
        "target_column": (
            training_request.get("target_column")
            or prepared_schema.get("target_column")
        ),
        "task_type": (
            training_request.get("task_type")
            or prepared_schema.get("task_type")
            or forest_config.get("task_type")
        ),

        "n_train": prepared_dataset.get("n_train"),
        "n_validation": prepared_dataset.get("n_validation"),
        "n_test": prepared_dataset.get("n_test"),
        "n_features": prepared_dataset.get("n_features"),

        "n_estimators": safe_int(n_estimators),
        "max_depth": forest_config.get("max_depth"),
        "max_features": forest_config.get("max_features"),
        "min_samples_split": forest_config.get("min_samples_split"),
        "min_samples_leaf": forest_config.get("min_samples_leaf"),
        "criterion": forest_config.get("criterion"),
        "bootstrap": forest_config.get("bootstrap"),
        "random_seed": (
            forest_config.get("global_random_seed")
            or training_request.get("global_random_seed")
        ),

        "assigned_workers": assigned_workers,
        "worker_count": len(assigned_workers),
        "expected_tree_count": experiment_record.get("expected_tree_count"),
        "completed_tree_count": experiment_record.get("completed_tree_count"),

        **metric_summary,
        **tree_summary,
    }

    return summary