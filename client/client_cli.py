from __future__ import annotations

import shutil
from pathlib import Path
from typing import NoReturn

import grpc

from client.config import ClientConfig, load_client_config
from client.master_client import (
    NoLeaderAvailableError,
    count_saved_trees,
    download_model,
    get_job_status,
    get_validation_metrics,
    list_experiments,
    reset_shared_artifacts,
    resume_training,
    run_inference_on_model_split,
    submit_training,
)
from client.printer import (
    print_count_saved_trees_response,
    print_download_model_response,
    print_error,
    print_info,
    print_job_status_response,
    print_list_experiments_response,
    print_reset_shared_artifacts_response,
    print_resume_training_response,
    print_run_inference_on_model_split_response,
    print_section,
    print_submit_training_response,
    print_validation_metrics_response,
)
from client.request_builder import (
    TrainingRequestConfig,
    build_submit_training_request,
)
from client.training_presets import (
    TaskType,
    TrainingPreset,
    build_custom_training_preset,
    get_training_presets_by_task,
)


def main() -> None:
    config = _load_config_or_exit()

    while True:
        print_section("SDCC/ML CLIENT")
        print("1 -> Submit training")
        print("2 -> See job status")
        print("3 -> See experiments")
        print("4 -> Count saved trees")
        print("5 -> See validation metrics")
        print("6 -> Submit inference")
        print("7 -> Resume training job")
        print("8 -> Download trained model")
        print("9 -> Eliminate shared artifacts")
        print("10 -> Show client configuration")
        print("0 -> Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            submit_training_launcher(config)

        elif choice == "2":
            see_job_status_launcher(config)

        elif choice == "3":
            see_experiments_launcher(config)

        elif choice == "4":
            count_saved_trees_launcher(config)

        elif choice == "5":
            see_validation_metrics_launcher(config)

        elif choice == "6":
            submit_inference_launcher(config)

        elif choice == "7":
            resume_training_launcher(config)

        elif choice == "8":
            download_model_launcher(config)

        elif choice == "9":
            reset_shared_artifacts_launcher(config)

        elif choice == "10":
            show_client_configuration(config)

        elif choice == "0":
            print()
            print("Closing client...")
            print()
            break

        else:
            print_error("Invalid option")


# =========================================================
# 1. SUBMIT TRAINING
# =========================================================

def submit_training_launcher(config: ClientConfig) -> None:
    task_type = choose_task_type()

    if task_type is None:
        return

    preset = choose_training_preset(task_type)

    if preset is None:
        return

    n_estimators_total = ask_positive_int(
        prompt="Insert total number of trees [0 to go back]: ",
        allow_zero=True,
    )

    if n_estimators_total is None or n_estimators_total == 0:
        return

    request_config = TrainingRequestConfig()

    try:
        request = build_submit_training_request(
            preset=preset,
            n_estimators_total=n_estimators_total,
            config=request_config,
        )

        print_section("SubmitTraining request")
        print(f"dataset_url:        {request.dataset_url}")
        print(f"target_column:      {request.target_column}")
        print(f"task_type:          {request.task_type}")
        print(f"dataset_scenario:  {request.dataset_scenario}")
        print(f"n_estimators_total: {request.n_estimators_total}")
        print(f"leakage_columns:    {list(request.leakage_columns)}")

        if not ask_yes_no("\nSend request to master? [y/N]: ", default=False):
            print_info("SubmitTraining cancelled")
            return

        response = submit_training(
            request=request,
            config=config,
        )

        print_submit_training_response(response)

    except NoLeaderAvailableError as exc:
        print_error(str(exc))

    except grpc.RpcError as exc:
        print_error(f"gRPC error: {exc}")

    except Exception as exc:
        print_error(str(exc))


def choose_task_type() -> TaskType | None:
    while True:
        print_section("Choose training task")
        print("1 -> Regression")
        print("2 -> Classification")
        print("0 -> Go back")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            return "regression"

        if choice == "2":
            return "classification"

        if choice == "0":
            return None

        print_error("Invalid option")


def choose_training_preset(task_type: TaskType) -> TrainingPreset | None:
    presets = get_training_presets_by_task(task_type)

    while True:
        print_section(f"Choose {task_type} dataset preset")

        for index, preset in enumerate(presets, start=1):
            print(f"{index} -> {preset.label}")

        custom_index = len(presets) + 1
        back_index = len(presets) + 2

        print(f"{custom_index} -> CUSTOM")
        print(f"{back_index} -> GO BACK")

        choice = input("\nSelect option: ").strip()

        try:
            selected_index = int(choice)
        except ValueError:
            print_error("Insert a valid number")
            continue

        if 1 <= selected_index <= len(presets):
            return presets[selected_index - 1]

        if selected_index == custom_index:
            return ask_custom_training_preset(task_type)

        if selected_index == back_index:
            return None

        print_error("Invalid option")


def ask_custom_training_preset(task_type: TaskType) -> TrainingPreset | None:
    print_section("Custom training preset")

    dataset_path = input(
        "Dataset path visible by master "
        "[example: /mnt/efs/gp_artifacts/datasets/file.csv]: "
    ).strip()

    if not dataset_path:
        print_error("Dataset path cannot be empty")
        return None

    target_column = input("Target column: ").strip()

    if not target_column:
        print_error("Target column cannot be empty")
        return None

    dataset_scenario = input(
        "Dataset scenario [baseline_original]: "
    ).strip()

    if not dataset_scenario:
        dataset_scenario = "baseline_original"

    leakage_columns_raw = input(
        "Leakage/drop columns separated by comma [empty]: "
    ).strip()

    leakage_columns = [
        item.strip()
        for item in leakage_columns_raw.split(",")
        if item.strip()
    ]

    default_criterion = "gini" if task_type == "classification" else "squared_error"

    criterion = input(
        f"Criterion [{default_criterion}]: "
    ).strip()

    if not criterion:
        criterion = default_criterion

    try:
        return build_custom_training_preset(
            dataset_path_value=dataset_path,
            task_type=task_type,
            target_column=target_column,
            dataset_scenario=dataset_scenario,
            leakage_columns=leakage_columns,
            criterion=criterion,
        )

    except Exception as exc:
        print_error(str(exc))
        return None


# =========================================================
# 2. SEE JOB STATUS
# =========================================================

def see_job_status_launcher(config: ClientConfig) -> None:
    print_section("SEE JOB STATUS")

    job_id = ask_required_string("Insert job_id: ")

    if job_id is None:
        return

    try:
        response = get_job_status(
            job_id=job_id,
            config=config,
        )

        print_job_status_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 3. SEE EXPERIMENTS
# =========================================================

def see_experiments_launcher(config: ClientConfig) -> None:
    print_section("SEE EXPERIMENTS")

    job_id = ask_required_string("Insert job_id: ")

    if job_id is None:
        return

    try:
        response = list_experiments(
            job_id=job_id,
            config=config,
        )

        print_list_experiments_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 4. COUNT SAVED TREES
# =========================================================

def count_saved_trees_launcher(config: ClientConfig) -> None:
    print_section("COUNT SAVED TREES")

    job_id = ask_required_string("Insert job_id: ")

    if job_id is None:
        return

    try:
        response = count_saved_trees(
            job_id=job_id,
            config=config,
        )

        print_count_saved_trees_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 5. SEE VALIDATION METRICS
# =========================================================

def see_validation_metrics_launcher(config: ClientConfig) -> None:
    print_section("SEE VALIDATION METRICS")

    job_id = ask_required_string("Insert job_id: ")

    if job_id is None:
        return

    experiment_id = ask_required_string("Insert experiment_id: ")

    if experiment_id is None:
        return

    try:
        response = get_validation_metrics(
            job_id=job_id,
            experiment_id=experiment_id,
            config=config,
        )

        print_validation_metrics_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 6. SUBMIT INFERENCE
# =========================================================

def submit_inference_launcher(config: ClientConfig) -> None:
    print_section("SUBMIT INFERENCE")

    print("This client-side option does not read EFS directly.")
    print("It asks the master to run inference on a saved model split.")
    print()

    model_id = ask_required_string("Insert model_id: ")

    if model_id is None:
        return

    split_name = choose_split_name()

    if split_name is None:
        return

    rows = ask_positive_int(
        prompt="Rows to preview [default 5]: ",
        default=5,
        allow_zero=False,
    )

    if rows is None:
        return

    try:
        response = run_inference_on_model_split(
            model_id=model_id,
            split_name=split_name,
            rows=rows,
            config=config,
        )

        print_run_inference_on_model_split_response(response)

    except Exception as exc:
        print_error(str(exc))


def choose_split_name() -> str | None:
    while True:
        print_section("Choose split")
        print("1 -> train")
        print("2 -> validation")
        print("3 -> test")
        print("0 -> Go back")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            return "train"

        if choice == "2":
            return "validation"

        if choice == "3":
            return "test"

        if choice == "0":
            return None

        print_error("Invalid option")


# =========================================================
# 7. RESUME TRAINING JOB
# =========================================================

def resume_training_launcher(config: ClientConfig) -> None:
    print_section("RESUME TRAINING JOB")

    job_id = ask_required_string("Insert job_id: ")

    if job_id is None:
        return

    try:
        response = resume_training(
            job_id=job_id,
            config=config,
        )

        print_resume_training_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 8. DOWNLOAD TRAINED MODEL
# =========================================================

def download_model_launcher(config: ClientConfig) -> None:
    print_section("DOWNLOAD TRAINED MODEL")

    model_id = ask_required_string("Insert model_id: ")

    if model_id is None:
        return

    print()
    print("CHOOSE FORMAT")
    print("1 -> Pickle/Joblib bundle")
    print("0 -> GO BACK")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        model_format = "pickle"
    elif choice == "0":
        return
    else:
        print_error("Invalid option")
        return

    output_dir_raw = input("\nOutput folder [./downloaded_models]: ").strip()
    output_dir = Path(output_dir_raw or "downloaded_models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_bytes = ask_yes_no(
        "\nDownload bytes through gRPC? [y/N]: ",
        default=False,
    )

    overwrite = ask_yes_no(
        "\nRegenerate export if already exists? [y/N]: ",
        default=False,
    )

    try:
        response = download_model(
            model_id=model_id,
            model_format=model_format,
            include_bytes=include_bytes,
            overwrite=overwrite,
            config=config,
        )

        print_download_model_response(response)

        if not getattr(response, "success", True):
            return

        _save_downloaded_model_if_possible(
            response=response,
            output_dir=output_dir,
        )

    except Exception as exc:
        print_error(str(exc))


def _save_downloaded_model_if_possible(
    *,
    response: object,
    output_dir: Path,
) -> None:
    filename = getattr(response, "filename", "") or ""
    payload = getattr(response, "payload", b"") or b""
    artifact_uri = getattr(response, "artifact_uri", "") or ""

    if not filename:
        filename = "downloaded_model.bin"

    destination = output_dir / filename

    if payload:
        destination.write_bytes(payload)
        print()
        print("downloaded_file:")
        print(destination)
        return

    source_path = _path_from_file_uri(artifact_uri)

    if source_path is not None and source_path.exists():
        shutil.copy2(source_path, destination)
        print()
        print("downloaded_file:")
        print(destination)
        return

    print()
    print("[INFO] Export created, but it was not copied to the client filesystem.")
    print("Use include_bytes=true to transfer the payload through gRPC,")
    print("or copy the artifact manually from the machine/storage that can access it.")

    if artifact_uri:
        print()
        print("artifact_uri:")
        print(artifact_uri)


def _path_from_file_uri(uri: str) -> Path | None:
    if not uri:
        return None

    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))

    if uri.startswith("/"):
        return Path(uri)

    return None


# =========================================================
# 9. ELIMINATE SHARED ARTIFACTS
# =========================================================

def reset_shared_artifacts_launcher(config: ClientConfig) -> None:
    print_section("ELIMINATE SHARED ARTIFACTS")

    print("This operation deletes shared artifacts managed by the master.")
    print("It should be used only for debug/test cleanup.")
    print()
    print("Folders affected on the master/shared storage:")
    print("- jobs")
    print("- models")
    print("- debug_inference_inputs")

    confirm_text = input(
        "\nType DELETE to confirm artifact reset, anything else to cancel: "
    ).strip()

    if confirm_text != "DELETE":
        print_info("Reset cancelled")
        return

    try:
        response = reset_shared_artifacts(
            confirm=True,
            config=config,
        )

        print_reset_shared_artifacts_response(response)

    except Exception as exc:
        print_error(str(exc))


# =========================================================
# 10. SHOW CONFIGURATION
# =========================================================

def show_client_configuration(config: ClientConfig) -> None:
    print_section("Client configuration")
    print("Master candidates:")

    for index, address in enumerate(config.master_addresses, start=1):
        print(f"{index}. {address}")

    print()
    print(f"gRPC max message length: {config.grpc_max_message_length_mb} MiB")
    print(f"gRPC timeout:            {config.grpc_timeout_seconds} seconds")


# =========================================================
# COMMON INPUT HELPERS
# =========================================================

def ask_required_string(prompt: str) -> str | None:
    value = input(f"\n{prompt}").strip()

    if not value:
        print_error("Value cannot be empty")
        return None

    return value


def ask_positive_int(
    *,
    prompt: str,
    default: int | None = None,
    allow_zero: bool = False,
) -> int | None:
    while True:
        raw_value = input(prompt).strip()

        if not raw_value and default is not None:
            return default

        try:
            value = int(raw_value)
        except ValueError:
            print_error("Insert a valid integer")
            continue

        if value == 0 and allow_zero:
            return value

        if value <= 0:
            print_error("Value must be > 0")
            continue

        return value


def ask_yes_no(prompt: str, *, default: bool = False) -> bool:
    value = input(prompt).strip().lower()

    if not value:
        return default

    return value in {"y", "yes"}


def _load_config_or_exit() -> ClientConfig:
    try:
        return load_client_config()

    except Exception as exc:
        print_error(f"Invalid client configuration: {exc}")
        _exit_with_error()


def _exit_with_error() -> NoReturn:
    raise SystemExit(1)


if __name__ == "__main__":
    main()