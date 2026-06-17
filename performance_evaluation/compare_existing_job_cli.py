from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from performance_evaluation.distributed_job_reader import (
    extract_distributed_job_summary,
)


DISTRIBUTED_JOBS_ROOT = (
    PROJECT_ROOT / "performance_evaluation" / "distributed_jobs"
)

RESULTS_ROOT = PROJECT_ROOT / "performance_evaluation" / "results"


LOCAL_BASELINE_RESULTS_ROOT = PROJECT_ROOT / "local_baseline" / "results"


def ask(label: str, default: str | None = None) -> str:
    if default is None:
        value = input(f"{label}: ").strip()
    else:
        value = input(f"{label} [{default}]: ").strip()

    if not value and default is not None:
        return default

    return value


def ask_choice(title: str, choices: list[tuple[str, str]]) -> str:
    print()
    print(title)
    print("=" * len(title))

    for index, (_, label) in enumerate(choices, start=1):
        print(f"{index} -> {label}")

    while True:
        raw = input("\nSelect option: ").strip()

        try:
            selected_index = int(raw)
        except ValueError:
            print("[ERROR] Insert a valid number")
            continue

        if 1 <= selected_index <= len(choices):
            return choices[selected_index - 1][0]

        print("[ERROR] Invalid option")


def load_json(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))

def safe_identifier(value: str) -> str:
    result = []

    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            result.append(char)
        else:
            result.append("_")

    safe_value = "".join(result).strip("_")
    return safe_value or "baseline"


def build_local_baseline_label(path: Path, local_result: dict[str, Any]) -> str:
    config = local_result.get("config") or {}

    task_type = config.get("task_type", "?")
    dataset_scenario = config.get("dataset_scenario", "?")
    target_column = config.get("target_column", "?")
    n_estimators = config.get("n_estimators", "?")

    relative_path = path.relative_to(PROJECT_ROOT).as_posix()

    return (
        f"{task_type} | "
        f"{dataset_scenario} | "
        f"target={target_column} | "
        f"trees={n_estimators} | "
        f"{relative_path}"
    )


def list_available_local_baselines() -> list[tuple[str, str]]:
    if not LOCAL_BASELINE_RESULTS_ROOT.exists():
        return []

    choices: list[tuple[str, str]] = []

    for path in sorted(LOCAL_BASELINE_RESULTS_ROOT.glob("*.json")):
        if not path.is_file():
            continue

        try:
            local_result = load_json(path)
        except Exception:
            continue

        config = local_result.get("config")

        if not isinstance(config, dict):
            continue

        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        label = build_local_baseline_label(path, local_result)

        choices.append((relative_path, label))

    return choices

def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def list_available_job_ids() -> list[str]:
    if not DISTRIBUTED_JOBS_ROOT.exists():
        return []

    job_ids: list[str] = []

    for path in sorted(DISTRIBUTED_JOBS_ROOT.iterdir()):
        if not path.is_dir():
            continue

        if (path / "job_record.json").exists():
            job_ids.append(path.name)

    return job_ids


def print_available_jobs() -> None:
    job_ids = list_available_job_ids()

    print()
    print("AVAILABLE DISTRIBUTED JOBS")
    print("=" * 80)

    if not job_ids:
        print(
            "No jobs found under: "
            f"{DISTRIBUTED_JOBS_ROOT}"
        )
        print()
        return

    for job_id in job_ids:
        print(f"- {job_id}")

    print()


def resolve_distributed_job_dir(job_id_or_path: str) -> Path:
    """
    Uso normale:
        job_af645...

    Risolve in:
        performance_evaluation/distributed_jobs/job_af645...

    Per comodità, se viene passato un path assoluto o relativo esistente,
    lo accetta comunque.
    """
    raw_path = Path(job_id_or_path)

    if raw_path.exists():
        return raw_path

    if raw_path.is_absolute():
        return raw_path

    candidate = DISTRIBUTED_JOBS_ROOT / job_id_or_path

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Distributed job not found.\n"
        f"Requested job id/path: {job_id_or_path}\n"
        f"Expected location: {candidate}\n\n"
        "Make sure the job directory exists under:\n"
        f"{DISTRIBUTED_JOBS_ROOT}"
    )


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def metric_name_for_task(task_type: str | None) -> str:
    if task_type == "classification":
        return "accuracy"

    if task_type == "regression":
        return "r2"

    return "metric"


def get_local_validation_metric(
    local_result: dict[str, Any],
    task_type: str | None,
) -> float | None:
    validation_metrics = local_result.get("validation_metrics") or {}

    if task_type == "classification":
        return safe_float(validation_metrics.get("accuracy"))

    if task_type == "regression":
        return safe_float(validation_metrics.get("r2"))

    return None


def get_local_test_metric(
    local_result: dict[str, Any],
    task_type: str | None,
) -> float | None:
    test_metrics = local_result.get("test_metrics") or {}

    if task_type == "classification":
        return safe_float(test_metrics.get("accuracy"))

    if task_type == "regression":
        return safe_float(test_metrics.get("r2"))

    return None


def get_distributed_validation_metric(
    distributed_result: dict[str, Any],
    task_type: str | None,
) -> float | None:
    if task_type == "classification":
        return safe_float(distributed_result.get("validation_accuracy"))

    if task_type == "regression":
        return safe_float(distributed_result.get("validation_r2"))

    return None


def get_local_criterion(
    local_config: dict[str, Any],
    task_type: str | None,
) -> Any:
    if task_type == "classification":
        return local_config.get("classification_criterion")

    if task_type == "regression":
        return local_config.get("regression_criterion")

    return None


def format_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def format_seconds(value: Any) -> str:
    numeric_value = safe_float(value)

    if numeric_value is None:
        return "-"

    return f"{numeric_value:.4f} s"


def build_default_output_path(
    job_id: str,
    local_option_key: str,
) -> str:
    filename = f"comparison_{local_option_key}_{job_id}.json"
    return str(Path("performance_evaluation") / "results" / filename)


def build_comparison(
    distributed_result: dict[str, Any],
    local_result: dict[str, Any],
) -> dict[str, Any]:
    local_config = local_result.get("config") or {}

    task_type = (
        local_config.get("task_type")
        or distributed_result.get("task_type")
    )

    metric_name = metric_name_for_task(task_type)

    local_validation_metric = get_local_validation_metric(
        local_result=local_result,
        task_type=task_type,
    )

    local_test_metric = get_local_test_metric(
        local_result=local_result,
        task_type=task_type,
    )

    distributed_validation_metric = get_distributed_validation_metric(
        distributed_result=distributed_result,
        task_type=task_type,
    )

    local_training_time = safe_float(
        local_result.get("training_time_seconds")
    )

    distributed_training_time = safe_float(
        distributed_result.get("training_wall_time_seconds")
    )

    if (
        local_training_time is not None
        and distributed_training_time is not None
        and distributed_training_time > 0
    ):
        training_time_ratio_local_over_distributed = (
            local_training_time / distributed_training_time
        )
    else:
        training_time_ratio_local_over_distributed = None

    if (
        local_validation_metric is not None
        and distributed_validation_metric is not None
    ):
        validation_metric_delta = (
            distributed_validation_metric - local_validation_metric
        )
    else:
        validation_metric_delta = None

    local_criterion = get_local_criterion(
        local_config=local_config,
        task_type=task_type,
    )

    comparison = {
        "comparison_type": "offline_existing_distributed_job_vs_local_baseline",
        "metric_name": metric_name,

        "compatibility_check": {
            "same_task_type": (
                local_config.get("task_type")
                == distributed_result.get("task_type")
            ),
            "same_target_column": (
                local_config.get("target_column")
                == distributed_result.get("target_column")
            ),
            "same_dataset_scenario": (
                local_config.get("dataset_scenario")
                == distributed_result.get("dataset_scenario")
            ),
            "same_n_estimators": (
                local_config.get("n_estimators")
                == distributed_result.get("n_estimators")
            ),
            "same_max_depth": (
                local_config.get("max_depth")
                == distributed_result.get("max_depth")
            ),
            "same_max_features": (
                local_config.get("max_features")
                == distributed_result.get("max_features")
            ),
            "same_criterion": (
                local_criterion == distributed_result.get("criterion")
            ),
            "same_bootstrap": (
                local_config.get("bootstrap")
                == distributed_result.get("bootstrap")
            ),
            "same_train_size": (
                local_result.get("n_train")
                == distributed_result.get("n_train")
            ),
            "same_validation_size": (
                local_result.get("n_validation")
                == distributed_result.get("n_validation")
            ),
            "same_test_size": (
                local_result.get("n_test")
                == distributed_result.get("n_test")
            ),
        },

        "local": {
            "mode": local_result.get("mode"),
            "dataset_url": local_config.get("dataset_url"),
            "dataset_path_resolved": local_result.get("dataset_path_resolved"),
            "dataset_scenario": local_config.get("dataset_scenario"),
            "target_column": local_config.get("target_column"),
            "task_type": local_config.get("task_type"),
            "n_estimators": local_config.get("n_estimators"),
            "max_depth": local_config.get("max_depth"),
            "max_features": local_config.get("max_features"),
            "criterion": local_criterion,
            "bootstrap": local_config.get("bootstrap"),
            "class_weight": local_config.get("class_weight"),
            "n_jobs": local_config.get("n_jobs"),
            "n_train": local_result.get("n_train"),
            "n_validation": local_result.get("n_validation"),
            "n_test": local_result.get("n_test"),
            "n_features_input": local_result.get("n_features_input"),
            "n_features_after_preprocessing": local_result.get(
                "n_features_after_preprocessing"
            ),
            "training_time_seconds": local_training_time,
            "validation_inference_time_seconds": local_result.get(
                "validation_inference_time_seconds"
            ),
            "test_inference_time_seconds": local_result.get(
                "test_inference_time_seconds"
            ),
            "validation_metric": local_validation_metric,
            "test_metric": local_test_metric,
            "validation_metrics": local_result.get("validation_metrics"),
            "test_metrics": local_result.get("test_metrics"),
        },

        "distributed": {
            "mode": "distributed",
            "job_id": distributed_result.get("job_id"),
            "job_dir": distributed_result.get("job_dir"),
            "job_status": distributed_result.get("status"),
            "experiment_id": distributed_result.get("experiment_id"),
            "experiment_status": distributed_result.get("experiment_status"),
            "model_id": distributed_result.get("model_id"),
            "dataset_uri": distributed_result.get("dataset_uri"),
            "dataset_scenario": distributed_result.get("dataset_scenario"),
            "target_column": distributed_result.get("target_column"),
            "task_type": distributed_result.get("task_type"),
            "n_estimators": distributed_result.get("n_estimators"),
            "max_depth": distributed_result.get("max_depth"),
            "max_features": distributed_result.get("max_features"),
            "criterion": distributed_result.get("criterion"),
            "bootstrap": distributed_result.get("bootstrap"),
            "worker_count": distributed_result.get("worker_count"),
            "assigned_workers": distributed_result.get("assigned_workers"),
            "n_train": distributed_result.get("n_train"),
            "n_validation": distributed_result.get("n_validation"),
            "n_test": distributed_result.get("n_test"),
            "n_features": distributed_result.get("n_features"),
            "training_wall_time_seconds": distributed_training_time,
            "validation_metric": distributed_validation_metric,
            "validation_accuracy": distributed_result.get(
                "validation_accuracy"
            ),
            "validation_r2": distributed_result.get("validation_r2"),
            "validation_mae": distributed_result.get("validation_mae"),
            "validation_mse": distributed_result.get("validation_mse"),
            "validation_rmse": distributed_result.get("validation_rmse"),
            "validation_confusion_matrix": distributed_result.get(
                "validation_confusion_matrix"
            ),
            "expected_tree_count": distributed_result.get(
                "expected_tree_count"
            ),
            "completed_tree_count": distributed_result.get(
                "completed_tree_count"
            ),
            "saved_tree_joblib_count": distributed_result.get(
                "saved_tree_joblib_count"
            ),
            "tree_distribution_by_worker": distributed_result.get(
                "tree_distribution_by_worker"
            ),
            "tree_training_time_sum_seconds": distributed_result.get(
                "tree_training_time_sum_seconds"
            ),
            "tree_training_time_avg_seconds": distributed_result.get(
                "tree_training_time_avg_seconds"
            ),
            "tree_training_time_max_seconds": distributed_result.get(
                "tree_training_time_max_seconds"
            ),
        },

        "comparison": {
            "local_validation_metric": local_validation_metric,
            "distributed_validation_metric": distributed_validation_metric,
            "validation_metric_delta_distributed_minus_local": (
                validation_metric_delta
            ),
            "local_training_time_seconds": local_training_time,
            "distributed_training_wall_time_seconds": distributed_training_time,
            "training_time_ratio_local_over_distributed": (
                training_time_ratio_local_over_distributed
            ),
        },
    }

    return comparison


def print_compatibility_warnings(comparison: dict[str, Any]) -> None:
    compatibility = comparison["compatibility_check"]

    failed_checks = [
        key
        for key, passed in compatibility.items()
        if passed is not True
    ]

    if not failed_checks:
        print()
        print("[OK] Local and distributed configurations look compatible.")
        return

    print()
    print("[WARN] Some local/distributed configuration checks differ:")

    for key in failed_checks:
        print(f"  - {key}: {compatibility[key]}")


def print_comparison_summary(comparison: dict[str, Any]) -> None:
    local = comparison["local"]
    distributed = comparison["distributed"]
    comparison_values = comparison["comparison"]
    metric_name = comparison["metric_name"]

    print()
    print("LOCAL VS DISTRIBUTED COMPARISON")
    print("=" * 80)

    print()
    print("CONFIGURATION")
    print("-" * 80)
    print(f"task_type:          {local.get('task_type')}")
    print(f"target_column:      {local.get('target_column')}")
    print(f"dataset_scenario:   {local.get('dataset_scenario')}")
    print(f"n_estimators:       {local.get('n_estimators')}")
    print(f"max_depth:          {local.get('max_depth')}")
    print(f"max_features:       {local.get('max_features')}")
    print(f"criterion:          {local.get('criterion')}")
    print(f"bootstrap:          {local.get('bootstrap')}")
    print(f"metric:             {metric_name}")

    print()
    print("RESULTS")
    print("-" * 80)
    print(
        f"{'Field':35s} | {'Local':20s} | {'Distributed':20s}"
    )
    print("-" * 80)

    print(
        f"{'mode':35s} | "
        f"{str(local.get('mode')):20s} | "
        f"{'distributed':20s}"
    )

    print(
        f"{'workers/jobs':35s} | "
        f"{str(local.get('n_jobs')):20s} | "
        f"{str(distributed.get('worker_count')):20s}"
    )

    print(
        f"{'n_train':35s} | "
        f"{str(local.get('n_train')):20s} | "
        f"{str(distributed.get('n_train')):20s}"
    )

    print(
        f"{'n_validation':35s} | "
        f"{str(local.get('n_validation')):20s} | "
        f"{str(distributed.get('n_validation')):20s}"
    )

    print(
        f"{'n_test':35s} | "
        f"{str(local.get('n_test')):20s} | "
        f"{str(distributed.get('n_test')):20s}"
    )

    print(
        f"{'n_features after preprocessing':35s} | "
        f"{str(local.get('n_features_after_preprocessing')):20s} | "
        f"{str(distributed.get('n_features')):20s}"
    )

    print(
        f"{'training time':35s} | "
        f"{format_seconds(local.get('training_time_seconds')):20s} | "
        f"{format_seconds(distributed.get('training_wall_time_seconds')):20s}"
    )

    print(
        f"{'validation ' + metric_name:35s} | "
        f"{format_value(local.get('validation_metric')):20s} | "
        f"{format_value(distributed.get('validation_metric')):20s}"
    )

    print(
        f"{'test ' + metric_name:35s} | "
        f"{format_value(local.get('test_metric')):20s} | "
        f"{'-':20s}"
    )

    print(
        f"{'trees completed':35s} | "
        f"{str(local.get('n_estimators')):20s} | "
        f"{str(distributed.get('completed_tree_count')):20s}"
    )

    print()
    print("DELTA")
    print("-" * 80)
    print(
        "distributed_validation_metric - local_validation_metric: "
        f"{format_value(comparison_values.get('validation_metric_delta_distributed_minus_local'))}"
    )

    ratio = comparison_values.get("training_time_ratio_local_over_distributed")

    if ratio is not None:
        print(
            "local_training_time / distributed_training_time: "
            f"{ratio:.6f}"
        )

    print()
    print("DISTRIBUTED TREE DISTRIBUTION BY WORKER")
    print("-" * 80)
    tree_distribution = distributed.get("tree_distribution_by_worker") or {}

    if not tree_distribution:
        print("-")
    else:
        for worker_id, tree_count in tree_distribution.items():
            print(f"{worker_id}: {tree_count}")

    print()


def main() -> None:
    print()
    print("===================================")
    print("COMPARE EXISTING DISTRIBUTED JOB")
    print("WITH LOCAL BASELINE")
    print("OFFLINE MODE")
    print("===================================")

    print()
    print(f"Distributed jobs root: {DISTRIBUTED_JOBS_ROOT}")

    print_available_jobs()

    local_choices = list_available_local_baselines()

    if local_choices:
        local_choices.append(("custom", "custom local JSON path"))
    else:
        print()
        print(
            "[WARN] No local baseline JSON files found under: "
            f"{LOCAL_BASELINE_RESULTS_ROOT}"
        )
        local_choices = [("custom", "custom local JSON path")]

    local_option_key = ask_choice(
        "CHOOSE LOCAL BASELINE",
        local_choices,
    )

    if local_option_key == "custom":
        local_json_path = ask("Local baseline JSON")
    else:
        local_json_path = local_option_key

    default_job_id = None
    available_job_ids = list_available_job_ids()

    if available_job_ids:
        default_job_id = available_job_ids[0]

    job_id = ask("Distributed job id", default_job_id)

    output_default = build_default_output_path(
        job_id=job_id,
        local_option_key=local_option_key,
    )

    output_json_path = ask(
        "Output comparison JSON",
        output_default,
    )

    distributed_job_dir = resolve_distributed_job_dir(job_id)

    distributed_result = extract_distributed_job_summary(
        distributed_job_dir
    )

    local_result = load_json(local_json_path)

    comparison = build_comparison(
        distributed_result=distributed_result,
        local_result=local_result,
    )

    print_compatibility_warnings(comparison)
    print_comparison_summary(comparison)

    output_path = resolve_output_path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print(f"Saved comparison JSON to: {output_path}")
    print()


if __name__ == "__main__":
    main()