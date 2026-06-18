import time
from pathlib import Path
import json
import os
import shutil
from typing import Optional, Callable, Any

import grpc
import numpy as np
import pandas as pd

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

import submit_training_classification
import submit_training_regression
from common.grpc_config import GRPC_OPTIONS

from common.prediction_io import load_prediction_array, path_to_file_uri

# =========================================================
# CONFIG
# =========================================================
def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            os.environ.setdefault(key, value)

ARTIFACT_ROOT = Path("/mnt/efs/gp_artifacts").resolve()
dataset_path = Path(ARTIFACT_ROOT / "datasets" / "diabetes_dataset.csv").resolve()

DATASETS_ROOT = ARTIFACT_ROOT / "datasets"

DISTRIBUTED_TRAINING_PRESETS = {
    "real_classification": {
        "label": "real diabetes classification",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "baseline_no_leakage",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },    "real_classification_no_diagnostic_features": {
        "label": "real diabetes classification - no diagnostic features",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "no_diagnostic_features",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_no_diagnostic_extended": {
        "label": "real diabetes classification - no diagnostic extended",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "no_diagnostic_extended",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_clinical_only": {
        "label": "real diabetes classification - clinical only",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "clinical_only",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_glucose_only": {
        "label": "real diabetes classification - glucose only",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "glucose_only",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_noise_10": {
        "label": "real diabetes classification - diagnostic noise 10%",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "diagnostic_noise_10pct",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_noise_25": {
        "label": "real diabetes classification - diagnostic noise 25%",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "diagnostic_noise_25pct",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_noise_50": {
        "label": "real diabetes classification - diagnostic noise 50%",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "diagnostic_noise_50pct",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_imbalance_positive_80": {
        "label": "real diabetes classification - 80% positives",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "imbalance_positive_80",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_imbalance_positive_90": {
        "label": "real diabetes classification - 90% positives",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "imbalance_positive_90",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_classification_imbalance_negative_80": {
        "label": "real diabetes classification - 80% negatives",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diagnosed_diabetes",
        "dataset_scenario": "imbalance_negative_80",
        "leakage_columns": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_stage_multiclass": {
        "label": "real diabetes stage multi-class classification",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "classification",
        "target_column": "diabetes_stage",
        "dataset_scenario": "stage_multiclass_no_leakage",
        "leakage_columns": [
            "diagnosed_diabetes",
            "diabetes_risk_score",
        ],
        "criterion": "gini",
    },
    "real_regression": {
        "label": "real diabetes regression",
        "dataset_path": DATASETS_ROOT / "diabetes_dataset.csv",
        "task_type": "regression",
        "target_column": "diabetes_risk_score",
        "dataset_scenario": "baseline_no_leakage",
        "leakage_columns": [
            "diagnosed_diabetes",
            "diabetes_stage",
        ],
        "criterion": "squared_error",
    },
    "synthetic_classification": {
        "label": "synthetic classification 100000x40",
        "dataset_path": DATASETS_ROOT / "synthetic_classification_100000_samples_40_features.csv",
        "task_type": "classification",
        "target_column": "target",
        "dataset_scenario": "baseline_original",
        "leakage_columns": [],
        "criterion": "gini",
    },
    "synthetic_regression": {
        "label": "synthetic regression 100000x40",
        "dataset_path": DATASETS_ROOT / "synthetic_regression_100000_samples_40_features.csv",
        "task_type": "regression",
        "target_column": "target",
        "dataset_scenario": "baseline_original",
        "leakage_columns": [],
        "criterion": "squared_error",
    },
}

DEFAULT_MASTER_HOST = "172.31.37.47"
DEFAULT_MASTER_PORT = "50051"

DATASET_SCENARIO_ORIGINAL = "baseline_original"
DATASET_SCENARIO_NO_LEAKAGE = "baseline_no_leakage"
DEFAULT_LEAKAGE_COLUMNS = ["diabetes_stage"]

# =========================================================
# MASTER LEADER DISCOVERY
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
load_env_file(PROJECT_ROOT / ".env.client")

def load_master_addresses() -> list[str]:
    raw_seeds = os.getenv("MASTER_SEEDS", "").strip()

    if raw_seeds:
        addresses = [
            item.strip()
            for item in raw_seeds.split(",")
            if item.strip()
        ]

        if addresses:
            return list(dict.fromkeys(addresses))

    master_host = os.getenv("MASTER_HOST", DEFAULT_MASTER_HOST).strip()
    master_port = os.getenv("MASTER_PORT", DEFAULT_MASTER_PORT).strip()

    return [f"{master_host}:{master_port}"]


def is_not_leader_message(message: str) -> bool:
    normalized = str(message or "").lower()
    return (
        "not leader" in normalized
        or "operation allowed only" in normalized
        or "leader-only operation rejected" in normalized
    )


def submit_training_with_leader_discovery(
    submit_function: Callable[..., Any],
    *submit_args,
) -> Optional[rf_pb2.SubmitTrainingResponse]:
    last_error = None

    for master_address in load_master_addresses():
        try:
            print()
            print(f"[CLIENT] Trying SubmitTraining on master {master_address}")

            response = submit_function(
                master_address,
                *submit_args,
            )

            if response is None:
                raise RuntimeError(
                    "submit_training module returned None. "
                    "For leader discovery to work correctly, "
                    "submit_training_classification.main(...) and "
                    "submit_training_regression.main(...) must return "
                    "the SubmitTrainingResponse received from gRPC."
                )

            message = getattr(response, "message", "")

            if getattr(response, "job_id", ""):
                print(f"[CLIENT] Training accepted by leader {master_address}")
                return response

            if is_not_leader_message(message):
                print(
                    f"[CLIENT] Master {master_address} is not leader. "
                    "Trying next candidate..."
                )
                last_error = RuntimeError(message)
                continue

            return response

        except Exception as exc:
            last_error = exc
            print(f"[CLIENT] SubmitTraining failed on {master_address}: {exc}")
            continue

    print()
    print("[ERROR] No master leader accepted SubmitTraining")
    print(last_error)
    print()

    return None


def submit_inference_with_leader_discovery(
    request: rf_pb2.SubmitInferenceRequest,
) -> Optional[rf_pb2.SubmitInferenceResponse]:
    last_error = None

    for master_address in load_master_addresses():
        try:
            print()
            print(f"[CLIENT] Trying SubmitInference on master {master_address}")

            with grpc.insecure_channel(
                master_address,
                options=GRPC_OPTIONS,
            ) as channel:
                stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

                response = stub.SubmitInference(
                    request,
                    timeout=120,
                )

            if response.success:
                print(f"[CLIENT] Inference accepted by leader {master_address}")
                return response

            if is_not_leader_message(response.error):
                print(
                    f"[CLIENT] Master {master_address} is not leader. "
                    "Trying next candidate..."
                )
                last_error = RuntimeError(response.error)
                continue

            return response

        except Exception as exc:
            last_error = exc
            print(f"[CLIENT] SubmitInference failed on {master_address}: {exc}")
            continue

    print()
    print("[ERROR] No master leader accepted SubmitInference")
    print(last_error)
    print()

    return None


def download_model_with_leader_discovery(
    request: rf_pb2.DownloadModelRequest,
) -> Optional[rf_pb2.DownloadModelResponse]:
    last_error = None

    for master_address in load_master_addresses():
        try:
            print()
            print(f"[CLIENT] Trying DownloadModel on master {master_address}")

            with grpc.insecure_channel(
                master_address,
                options=GRPC_OPTIONS,
            ) as channel:
                stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

                response = stub.DownloadModel(
                    request,
                    timeout=120,
                )

            if response.success:
                print(f"[CLIENT] DownloadModel accepted by leader {master_address}")
                return response

            if is_not_leader_message(response.error):
                print(
                    f"[CLIENT] Master {master_address} is not leader. "
                    "Trying next candidate..."
                )
                last_error = RuntimeError(response.error)
                continue

            return response

        except Exception as exc:
            last_error = exc
            print(f"[CLIENT] DownloadModel failed on {master_address}: {exc}")
            continue

    print()
    print("[ERROR] No master leader accepted DownloadModel")
    print(last_error)
    print()

    return None

def resume_training_launcher():
    print()
    print("===================================")
    print("RESUME TRAINING JOB")
    print("===================================")

    job_id = input("\nInsert job_id: ").strip()

    if not job_id:
        print()
        print("[ERROR] job_id cannot be empty")
        print()
        return

    response = resume_training_with_leader_discovery(job_id)

    if response is None:
        return

    print()
    print("job_id:")
    print(response.job_id)

    print()
    print("status:")
    print(response.status)

    print()
    print("message:")
    print(response.message)

    print()

# =========================================================
# INPUT HELPERS
# =========================================================

def _n_estimators_total() -> int:
    try:
        print("INSERIRE IL NUMERO TOTALE DI ALBERI")

        n_estimators_total = int(
            input("\nSelect option: ").strip()
        )

        return n_estimators_total

    except Exception:
        return 0

def choose_distributed_training_preset(task_type: str) -> Optional[dict[str, Any]]:
    matching_presets = [
        (key, config)
        for key, config in DISTRIBUTED_TRAINING_PRESETS.items()
        if config["task_type"] == task_type
    ]

    print()
    print("CHOOSE DISTRIBUTED TRAINING DATASET")
    print("=" * 80)

    for index, (_, config) in enumerate(matching_presets, start=1):
        print(f"{index} -> {config['label']}")

    print(f"{len(matching_presets) + 1} -> CUSTOM")
    print(f"{len(matching_presets) + 2} -> GO BACK")

    choice = input("\nSelect option: ").strip()

    try:
        selected_index = int(choice)
    except ValueError:
        print()
        print("[ERROR] Insert a valid number")
        print()
        return None

    if 1 <= selected_index <= len(matching_presets):
        return matching_presets[selected_index - 1][1]

    if selected_index == len(matching_presets) + 1:
        dataset_path_value = input(
            "\nDataset path on EFS "
            "[/mnt/efs/gp_artifacts/datasets/...]: "
        ).strip()

        target_column = input("Target column: ").strip()

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

        if task_type == "classification":
            criterion = "gini"
        else:
            criterion = "squared_error"

        return {
            "label": "custom",
            "dataset_path": Path(dataset_path_value),
            "task_type": task_type,
            "target_column": target_column,
            "dataset_scenario": dataset_scenario,
            "leakage_columns": leakage_columns,
            "criterion": criterion,
        }

    if selected_index == len(matching_presets) + 2:
        return None

    print()
    print("[ERROR] Invalid option")
    print()
    return None

def print_submit_training_response(
    response: Optional[rf_pb2.SubmitTrainingResponse],
) -> None:
    if response is None:
        return

    print()
    print("job_id:")
    print(response.job_id)

    print()
    print("status:")
    print(response.status)

    print()
    print("message:")
    print(response.message)

    print()


# =========================================================
# SUBMIT TRAINING
# =========================================================

def _submit_training_regression():
    preset = choose_distributed_training_preset("regression")

    if preset is None:
        return

    n_estimators_total = _n_estimators_total()

    if n_estimators_total <= 0:
        print("INSERT A NUMBER OF USABLE TREE, >0")
        return

    response = submit_training_with_leader_discovery(
        submit_training_regression.main,
        preset["dataset_path"],
        n_estimators_total,
        preset["dataset_scenario"],
        preset["leakage_columns"],
        preset["target_column"],
        preset["criterion"],
    )

    print_submit_training_response(response)


def _submit_training_classification():
    preset = choose_distributed_training_preset("classification")

    if preset is None:
        return

    n_estimators_total = _n_estimators_total()

    if n_estimators_total <= 0:
        print("INSERT A NUMBER OF USABLE TREE, >0")
        return

    response = submit_training_with_leader_discovery(
        submit_training_classification.main,
        preset["dataset_path"],
        n_estimators_total,
        preset["dataset_scenario"],
        preset["leakage_columns"],
        preset["target_column"],
        preset["criterion"],
    )

    print_submit_training_response(response)

def submit_training_launcher():
    print()
    print("===================================")
    print("SUBMIT TRAINING. CHOOSE BETWEEN:")
    print("1 -> REGRESSION")
    print("2 -> CLASSIFICATION")
    print("3 -> GO BACK")
    print("===================================")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        _submit_training_regression()

    elif choice == "2":
        _submit_training_classification()

    elif choice == "3":
        return

    else:
        print()
        print("[ERROR] Invalid option")
        print()

def resume_training_with_leader_discovery(job_id: str):
    last_error = None

    request = rf_pb2.ResumeTrainingRequest(
        job_id=job_id,
    )

    for master_address in load_master_addresses():
        try:
            print()
            print(f"[CLIENT] Trying ResumeTraining on master {master_address}")

            with grpc.insecure_channel(
                master_address,
                options=GRPC_OPTIONS,
            ) as channel:
                stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

                response = stub.ResumeTraining(
                    request,
                    timeout=30,
                )

            message = getattr(response, "message", "")

            if response.status == rf_pb2.RUNNING:
                print(f"[CLIENT] Resume accepted by leader {master_address}")
                return response

            if response.status == rf_pb2.COMPLETED:
                print(f"[CLIENT] Job already completed on leader {master_address}")
                return response

            if is_not_leader_message(message):
                print(
                    f"[CLIENT] Master {master_address} is not leader. "
                    "Trying next candidate..."
                )
                last_error = RuntimeError(message)
                continue

            return response

        except Exception as exc:
            last_error = exc
            print(f"[CLIENT] ResumeTraining failed on {master_address}: {exc}")
            continue

    print()
    print("[ERROR] No master leader accepted ResumeTraining")
    print(last_error)
    print()

    return None

# =========================================================
# JOB STATUS
# =========================================================

def see_job_status(job_id: str):
    job_record_path = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "job_record.json"
    )

    if not job_record_path.exists():
        print()
        print("[ERROR] job_record.json not found")
        print(job_record_path)
        print()
        return

    with open(
        job_record_path,
        "r",
        encoding="utf-8",
    ) as f:
        job_record = json.load(f)

    print()
    print("status:")
    print(job_record.get("status"))

    print()
    print("message:")
    print(job_record.get("message"))

    print()
    print("selected_experiment_id:")
    print(job_record.get("selected_experiment_id"))

    print()
    print("model_id:")
    print(job_record.get("model_id"))
    print()


def see_job_status_launcher():
    print()
    print("===================================")
    print("SEE JOB STATUS")
    print("===================================")

    job_id = input("\nInsert job_id: ").strip()

    see_job_status(job_id)


# =========================================================
# SEE EXPERIMENTS
# =========================================================

def see_experiments(job_id: str):
    experiments_root = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
    )

    if not experiments_root.exists():
        print()
        print("[ERROR] experiments folder not found")
        print(experiments_root)
        print()
        return

    print()

    for experiment_dir in experiments_root.iterdir():
        if not experiment_dir.is_dir():
            continue

        experiment_record_path = (
            experiment_dir
            / "experiment_record.json"
        )

        if not experiment_record_path.exists():
            continue

        with open(
            experiment_record_path,
            "r",
            encoding="utf-8",
        ) as f:
            experiment_record = json.load(f)

        print("===================================")

        print("experiment_id:")
        print(experiment_record.get("experiment_id"))

        print()
        print("status:")
        print(experiment_record.get("status"))

        print()
        print("expected_tree_count:")
        print(experiment_record.get("expected_tree_count"))

        print()
        print("completed_tree_count:")
        print(experiment_record.get("completed_tree_count"))

        print()
        print("assigned_workers:")
        print(experiment_record.get("assigned_workers"))

        print()


def see_experiments_launcher():
    print()
    print("===================================")
    print("SEE EXPERIMENTS")
    print("===================================")

    job_id = input("\nInsert job_id: ").strip()

    see_experiments(job_id)


# =========================================================
# COUNT TREES
# =========================================================

def count_saved_trees(job_id: str):
    experiments_root = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
    )

    if not experiments_root.exists():
        print()
        print("[ERROR] experiments folder not found")
        print(experiments_root)
        print()
        return

    print()

    for experiment_dir in experiments_root.iterdir():
        if not experiment_dir.is_dir():
            continue

        trees_dir = (
            experiment_dir
            / "trees"
        )

        tree_count = len(
            list(
                trees_dir.glob("*.joblib")
            )
        )

        print("===================================")

        print("Experiment:")
        print(experiment_dir.name)

        print()
        print("Saved trees:")
        print(tree_count)

        print()


def count_saved_trees_launcher():
    print()
    print("===================================")
    print("COUNT SAVED TREES")
    print("===================================")

    job_id = input("\nInsert job_id: ").strip()

    count_saved_trees(job_id)


# =========================================================
# SEE METRICS
# =========================================================

def see_validation_metrics(
    job_id: str,
    experiment_id: str,
):
    experiment_record_path = (
        ARTIFACT_ROOT
        / "jobs"
        / job_id
        / "experiments"
        / experiment_id
        / "experiment_record.json"
    )

    if not experiment_record_path.exists():
        print()
        print("[ERROR] experiment_record.json not found")
        print(experiment_record_path)
        print()
        return

    with open(
        experiment_record_path,
        "r",
        encoding="utf-8",
    ) as f:
        experiment_record = json.load(f)

    print()

    print("status:")
    print(experiment_record.get("status"))

    print()
    print("expected_tree_count:")
    print(experiment_record.get("expected_tree_count"))

    print()
    print("completed_tree_count:")
    print(experiment_record.get("completed_tree_count"))

    print()
    print("assigned_workers:")
    print(experiment_record.get("assigned_workers"))

    print()
    print("validation_metrics:")
    print(experiment_record.get("validation_metrics"))

    print()


def see_validation_metrics_launcher():
    print()
    print("===================================")
    print("SEE VALIDATION METRICS")
    print("===================================")

    job_id = input("\nInsert job_id: ").strip()

    experiment_id = input("Insert experiment_id: ").strip()

    see_validation_metrics(
        job_id,
        experiment_id,
    )


# =========================================================
# SUBMIT INFERENCE
# =========================================================

def path_from_file_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))
    return Path(uri)


def matrix_to_proto(X: np.ndarray) -> rf_pb2.DenseMatrix:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix")

    return rf_pb2.DenseMatrix(
        values=X.ravel().tolist(),
        n_rows=X.shape[0],
        n_cols=X.shape[1],
    )


def load_manifest_by_model_id(model_id: str) -> dict:
    manifest_path = (
        ARTIFACT_ROOT
        / "models"
        / model_id
        / "manifest.json"
    )

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for candidate in ARTIFACT_ROOT.rglob("manifest.json"):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if manifest.get("model_id") == model_id:
                return manifest

        except Exception:
            continue

    raise FileNotFoundError(
        f"Manifest not found for model_id={model_id}"
    )


def select_features_uri_from_manifest(
    manifest: dict,
    split_name: str,
) -> str:
    if split_name == "train":
        return manifest["train_features_uri"]

    if split_name == "validation":
        return manifest["validation_features_uri"]

    if split_name == "test":
        return manifest["test_features_uri"]

    raise ValueError(f"Unsupported split_name: {split_name}")


def select_labels_uri_from_manifest(
    manifest: dict,
    split_name: str,
) -> Optional[str]:
    key = f"{split_name}_labels_uri"
    return manifest.get(key)


def submit_inference(
    model_id: str,
    split_name: str,
    rows: int,
):
    try:
        manifest = load_manifest_by_model_id(model_id)

    except Exception as exc:
        print()
        print("[ERROR] Manifest loading failed")
        print(exc)
        print()
        return

    try:
        features_uri = select_features_uri_from_manifest(
            manifest,
            split_name,
        )

        features_path = path_from_file_uri(features_uri)

        if not features_path.exists():
            print()
            print("[ERROR] features file not found")
            print(features_path)
            print()
            return

        X_df = pd.read_parquet(features_path)

        feature_names = manifest.get("feature_names") or []

        if feature_names:
            missing_features = [
                feature
                for feature in feature_names
                if feature not in X_df.columns
            ]

            if missing_features:
                print()
                print("[ERROR] Some manifest features are missing from features parquet")
                print(missing_features)
                print()
                return

            X_df = X_df[feature_names]

        X_debug_df = X_df.head(rows).reset_index(drop=True)

        inference_input_dir = ARTIFACT_ROOT / "debug_inference_inputs" / model_id
        inference_input_dir.mkdir(parents=True, exist_ok=True)

        inference_input_path = (
            inference_input_dir
            / f"{split_name}_head_{rows}_{int(time.time())}.parquet"
        )

        X_debug_df.to_parquet(inference_input_path, index=False)

        inference_features_uri = path_to_file_uri(inference_input_path)

        request = rf_pb2.SubmitInferenceRequest(
            model_id=model_id,
            features_uri=inference_features_uri,
        )

        print()
        print("features_uri:")
        print(inference_features_uri)

        response = submit_inference_with_leader_discovery(request)

        if response is None:
            return

    except Exception as exc:
        print()
        print("[ERROR] SubmitInference RPC failed")
        print(exc)
        print()
        return

    print()
    print("success:")
    print(response.success)

    print()
    print("error:")
    print(response.error)

    print()
    print("task_type:")
    print(response.task_type)

    print()
    print("prediction_uri:")
    print(response.prediction_uri)

    print()
    print("n_rows:")
    print(response.n_rows)

    print()
    print("n_cols:")
    print(response.n_cols)

    predictions = None

    if response.success and response.prediction_uri:
        try:
            predictions = load_prediction_array(response.prediction_uri)

            print()
            print("predictions:")
            print(predictions.reshape(-1).tolist())

        except Exception as exc:
            print()
            print("[ERROR] Could not load prediction_uri")
            print(exc)

    labels_uri = select_labels_uri_from_manifest(
        manifest,
        split_name,
    )

    if labels_uri is not None:
        labels_path = path_from_file_uri(labels_uri)

        if labels_path.exists():
            y_df = pd.read_parquet(labels_path)
            y_values = y_df.head(rows).values.reshape(-1).tolist()

            print()
            print("expected_values_from_split:")
            print(y_values)

            if predictions is not None and response.task_type == "classification":
                predicted_values = predictions.reshape(-1).tolist()

                correct = sum(
                    1
                    for pred, true in zip(predicted_values, y_values)
                    if str(pred) == str(true)
                )

                accuracy = correct / len(y_values) if y_values else 0.0

                print()
                print("local_accuracy:")
                print(round(accuracy, 4))

    print()

def submit_inference_launcher():
    print()
    print("===================================")
    print("SUBMIT INFERENCE")
    print("===================================")

    model_id = input("\nInsert model_id: ").strip()

    if not model_id:
        print()
        print("[ERROR] model_id cannot be empty")
        print()
        return

    print()
    print("CHOOSE SPLIT")
    print("1 -> VALIDATION")
    print("2 -> TEST")
    print("3 -> TRAIN")
    print("4 -> GO BACK")

    split_choice = input("\nSelect option: ").strip()

    if split_choice == "1":
        split_name = "validation"

    elif split_choice == "2":
        split_name = "test"

    elif split_choice == "3":
        split_name = "train"

    elif split_choice == "4":
        return

    else:
        print()
        print("[ERROR] Invalid split option")
        print()
        return

    rows_raw = input("\nHow many rows? Default 5: ").strip()

    if rows_raw == "":
        rows = 5
    else:
        try:
            rows = int(rows_raw)
        except ValueError:
            print()
            print("[ERROR] rows must be an integer")
            print()
            return

    if rows <= 0:
        print()
        print("[ERROR] rows must be > 0")
        print()
        return

    submit_inference(
        model_id=model_id,
        split_name=split_name,
        rows=rows,
    )



# =========================================================
# DOWNLOAD MODEL
# =========================================================

def download_model_launcher():
    print()
    print("===================================")
    print("DOWNLOAD TRAINED MODEL")
    print("===================================")

    model_id = input("\nInsert model_id: ").strip()
    if not model_id:
        print()
        print("[ERROR] model_id cannot be empty")
        print()
        return

    print()
    print("CHOOSE FORMAT")
    print("1 -> Pickle/Joblib bundle")
    print("2 -> GO BACK")

    choice = input("\nSelect option: ").strip()
    if choice == "1":
        export_format = "pickle"
    elif choice == "2":
        return
    else:
        print()
        print("[ERROR] Invalid option")
        print()
        return

    output_dir_raw = input("\nOutput folder [./downloaded_models]: ").strip()
    output_dir = Path(output_dir_raw or "downloaded_models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_bytes = input("\nDownload bytes through gRPC? [y/N]: ").strip().lower() in {"y", "yes"}
    overwrite = input("\nRegenerate export if already exists? [y/N]: ").strip().lower() in {"y", "yes"}

    request = rf_pb2.DownloadModelRequest(
        model_id=model_id,
        format=export_format,
        include_bytes=include_bytes,
        overwrite=overwrite,
    )

    response = download_model_with_leader_discovery(request)
    if response is None:
        return

    if not response.success:
        print()
        print("[ERROR] DownloadModel failed")
        print(response.error)
        print()
        return

    print()
    print("artifact_uri:")
    print(response.artifact_uri)

    print()
    print("size_bytes:")
    print(response.size_bytes)

    destination = output_dir / response.filename

    if response.payload:
        destination.write_bytes(response.payload)
        print()
        print("downloaded_file:")
        print(destination)
        print()
        return

    source_path = path_from_file_uri(response.artifact_uri)
    if source_path.exists():
        shutil.copy2(source_path, destination)
        print()
        print("downloaded_file:")
        print(destination)
        print()
        return

    print()
    print("[INFO] Export created but not copied locally.")
    print("The returned artifact_uri is not accessible from this client filesystem.")
    print("Use scp/rsync from the machine that can access the shared storage.")
    print()

# =========================================================
# RESET ARTIFACTS
# =========================================================

def reset_shared_artifacts() -> None:
    folders_to_clean = [
        ARTIFACT_ROOT / "jobs",
        ARTIFACT_ROOT / "models",
    ]

    for folder in folders_to_clean:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

        folder.mkdir(parents=True, exist_ok=True)


def reset_shared_artifacts_launcher():
    print()
    print("===================================")
    print("ELIMINATING PAST SHARED ARTIFACTS")
    print("===================================")
    print("1 -> ELIMINATE ARTIFACTS")
    print("2 -> GO BACK TO MENU")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        reset_shared_artifacts()

    elif choice == "2":
        return

    else:
        print()
        print("[ERROR] Invalid option")
        print()


# =========================================================
# MAIN MENU
# =========================================================

def main():
    while True:
        print()
        print("===================================")
        print("TRAINING DEBUG CLI")
        print("===================================")

        print("1 -> Submit training")
        print("2 -> See job status")
        print("3 -> See experiments")
        print("4 -> Count saved trees")
        print("5 -> See validation metrics")
        print("6 -> Submit inference")
        print("7 -> Resume training job")
        print("8 -> Download trained model")
        print("9 -> Eliminate shared artifacts")
        print("0 -> Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            submit_training_launcher()

        elif choice == "2":
            see_job_status_launcher()

        elif choice == "3":
            see_experiments_launcher()

        elif choice == "4":
            count_saved_trees_launcher()

        elif choice == "5":
            see_validation_metrics_launcher()

        elif choice == "6":
            submit_inference_launcher()

        elif choice == "7":
            resume_training_launcher()

        elif choice == "8":
            download_model_launcher()

        elif choice == "9":
            reset_shared_artifacts_launcher()

        elif choice == "0":
            print()
            print("Closing CLI...")
            print()
            break

        else:
            print()
            print("[ERROR] Invalid option")
            print()


if __name__ == "__main__":
    main()