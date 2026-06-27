from __future__ import annotations

import json
from typing import Any


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_error(message: str) -> None:
    print()
    print("[ERROR]")
    print(message)


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_master_attempt(address: str) -> None:
    print(f"[CLIENT] Trying master {address}...")


def print_submit_training_response(response: object) -> None:
    print_section("SubmitTraining response")

    job_id = getattr(response, "job_id", "")
    status = getattr(response, "status", "")
    message = getattr(response, "message", "")

    if job_id:
        print(f"job_id:  {job_id}")

    if status != "":
        print(f"status:  {status}")

    if message:
        print(f"message: {message}")

    if not job_id and status == "" and not message:
        print(response)


def print_resume_training_response(response: object) -> None:
    print_section("ResumeTraining response")

    job_id = getattr(response, "job_id", "")
    status = getattr(response, "status", "")
    message = getattr(response, "message", "")

    if job_id:
        print(f"job_id:  {job_id}")

    if status != "":
        print(f"status:  {status}")

    if message:
        print(f"message: {message}")

    if not job_id and status == "" and not message:
        print(response)


def print_job_status_response(response: object) -> None:
    print_section("Job status")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    print(f"job_id:                 {getattr(response, 'job_id', '')}")
    print(f"status:                 {getattr(response, 'status', '')}")
    print(f"message:                {getattr(response, 'message', '')}")
    print(f"selected_experiment_id: {getattr(response, 'selected_experiment_id', '')}")
    print(f"model_id:               {getattr(response, 'model_id', '')}")


def print_list_experiments_response(response: object) -> None:
    print_section("Experiments")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    print(f"job_id: {getattr(response, 'job_id', '')}")
    print()

    experiments = list(getattr(response, "experiments", []))

    if not experiments:
        print("No experiments found.")
        return

    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}] {getattr(experiment, 'experiment_id', '')}")
        print(f"    status:               {getattr(experiment, 'status', '')}")
        print(
            "    expected_tree_count:  "
            f"{getattr(experiment, 'expected_tree_count', 0)}"
        )
        print(
            "    completed_tree_count: "
            f"{getattr(experiment, 'completed_tree_count', 0)}"
        )
        print(
            "    assigned_workers:     "
            f"{list(getattr(experiment, 'assigned_workers', []))}"
        )
        print()


def print_count_saved_trees_response(response: object) -> None:
    print_section("Saved trees")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    print(f"job_id: {getattr(response, 'job_id', '')}")
    print()

    counts = list(getattr(response, "counts", []))

    if not counts:
        print("No saved trees found.")
        return

    for item in counts:
        print(
            f"{getattr(item, 'experiment_id', '')}: "
            f"{getattr(item, 'saved_tree_count', 0)} saved trees"
        )


def print_validation_metrics_response(response: object) -> None:
    print_section("Validation metrics")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    print(f"job_id:                 {getattr(response, 'job_id', '')}")
    print(f"experiment_id:          {getattr(response, 'experiment_id', '')}")
    print(f"status:                 {getattr(response, 'status', '')}")
    print(f"expected_tree_count:    {getattr(response, 'expected_tree_count', 0)}")
    print(f"completed_tree_count:   {getattr(response, 'completed_tree_count', 0)}")
    print(f"assigned_workers:       {list(getattr(response, 'assigned_workers', []))}")
    print()

    metrics_json = getattr(response, "validation_metrics_json", "{}") or "{}"

    print("validation_metrics:")

    try:
        metrics = json.loads(metrics_json)
        print(json.dumps(metrics, indent=2, sort_keys=True))
    except Exception:
        print(metrics_json)


def print_run_inference_on_model_split_response(response: object) -> None:
    print_section("Inference result")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    print(f"model_id:       {getattr(response, 'model_id', '')}")
    print(f"split_name:     {getattr(response, 'split_name', '')}")
    print(f"task_type:      {getattr(response, 'task_type', '')}")
    print(f"prediction_uri: {getattr(response, 'prediction_uri', '')}")
    print(f"n_rows:         {getattr(response, 'n_rows', 0)}")
    print(f"n_cols:         {getattr(response, 'n_cols', 0)}")

    if getattr(response, "has_local_accuracy", False):
        print(f"local_accuracy: {getattr(response, 'local_accuracy', 0.0):.6f}")

    predicted_values = list(getattr(response, "predicted_values", []))
    expected_values = list(getattr(response, "expected_values", []))

    print()

    if not predicted_values:
        print("No predictions returned.")
        return

    print("Preview:")
    print("-" * 80)

    if expected_values:
        for index, predicted in enumerate(predicted_values):
            expected = expected_values[index] if index < len(expected_values) else ""
            print(f"{index + 1}. predicted={predicted} | expected={expected}")
    else:
        for index, predicted in enumerate(predicted_values):
            print(f"{index + 1}. predicted={predicted}")


def print_submit_inference_response(response: object) -> None:
    print_section("SubmitInference response")

    prediction_uri = getattr(response, "prediction_uri", "")
    task_type = getattr(response, "task_type", "")
    n_rows = getattr(response, "n_rows", 0)
    n_cols = getattr(response, "n_cols", 0)

    if prediction_uri:
        print(f"prediction_uri: {prediction_uri}")

    if task_type:
        print(f"task_type:      {task_type}")

    print(f"n_rows:         {n_rows}")
    print(f"n_cols:         {n_cols}")


def print_download_model_response(response: object) -> None:
    print_section("DownloadModel response")

    success = getattr(response, "success", True)
    error = getattr(response, "error", "")

    if success is False:
        print_error(error or "Download failed")
        return

    model_id = getattr(response, "model_id", "")
    model_uri = getattr(response, "model_uri", "")
    local_path = getattr(response, "local_path", "")
    format_value = getattr(response, "format", "")
    size_bytes = getattr(response, "size_bytes", 0)

    if model_id:
        print(f"model_id:   {model_id}")

    if format_value:
        print(f"format:     {format_value}")

    if model_uri:
        print(f"model_uri:  {model_uri}")

    if local_path:
        print(f"local_path: {local_path}")

    if size_bytes:
        print(f"size_bytes: {size_bytes}")

    if not model_id and not model_uri and not local_path:
        print(response)


def print_reset_shared_artifacts_response(response: object) -> None:
    print_section("Reset shared artifacts")

    if not _response_success(response):
        print_error(_response_error(response))
        return

    message = getattr(response, "message", "")

    if message:
        print(message)
    else:
        print("Shared artifacts reset completed.")


def print_generic_response(response: object) -> None:
    print_section("Response")

    if hasattr(response, "success") and not _response_success(response):
        print_error(_response_error(response))
        return

    print(response)


def _response_success(response: object) -> bool:
    return bool(getattr(response, "success", True))


def _response_error(response: object) -> str:
    return str(getattr(response, "error", "") or "Operation failed")


def print_key_value_table(values: dict[str, Any]) -> None:
    if not values:
        return

    max_key_length = max(len(str(key)) for key in values.keys())

    for key, value in values.items():
        print(f"{key:<{max_key_length}} : {value}")