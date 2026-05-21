from pathlib import Path
import json
import shutil

import grpc
import numpy as np
import pandas as pd
from typing import Optional

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

import submit_training_classification
import submit_training_regression

# =========================================================
# CONFIG
# =========================================================

##########################################################
#
#utilizzare l'ip privato del master per fare l'allenamento
#Correggere per rendere tutto automatico
#
###########################################################
MASTER_ADDRESS = "172.31.37.47:50051"

###########################################################
#
#Modificare, se non funziona, con il mount comune
#
###########################################################
ARTIFACT_ROOT = Path("/mnt/efs/gp_artifacts").resolve()
dataset_path = Path(ARTIFACT_ROOT / "datasets" / "diabetes_dataset.csv").resolve()


DATASET_SCENARIO_ORIGINAL = "baseline_original"
DATASET_SCENARIO_NO_LEAKAGE = "baseline_no_leakage"
DEFAULT_LEAKAGE_COLUMNS = ["diabetes_stage"]

GRPC_MAX_MESSAGE_LENGTH = 64 * 1024 * 1024

GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
]



# =========================================================
# SUBMIT TRAINING
# =========================================================

def submit_training():

    print("CHOOSE TRAINING TYPE")
    print("1 -> CLASSIFICATION - BASELINE ORIGINAL")
    print("2 -> CLASSIFICATION - BASELINE NO LEAKAGE")
    print("3 -> CLASSIFICATION - NO DIAGNOSTIC FEATURES")
    print("4 -> CLASSIFICATION - NO DIAGNOSTIC EXTENDED")
    print("5 -> CLASSIFICATION - CLINICAL ONLY")
    print("6 -> CLASSIFICATION - GLUCOSE ONLY")
    print("7 -> REGRESSION")
    print("8 -> GO BACK")

    choice = input(
        "\nSelect option: "
    ).strip()

    if choice == "1":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="baseline_original",
            leakage_columns=[],
        )

    elif choice == "2":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="baseline_no_leakage",
            leakage_columns=["diabetes_stage"],
        )

    elif choice == "3":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="no_diagnostic_features",
            leakage_columns=[],
        )

    elif choice == "4":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="no_diagnostic_extended",
            leakage_columns=[],
        )

    elif choice == "5":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="clinical_only",
            leakage_columns=[],
        )

    elif choice == "6":
        submit_training_classification.main(
            MASTER_ADDRESS,
            dataset_path,
            dataset_scenario="glucose_only",
            leakage_columns=[],
        )

    elif choice == "7":
        submit_training_regression.main(
            MASTER_ADDRESS,
            dataset_path,
        )

    elif choice == "8":
        return

    else:
        print()
        print("[ERROR] Invalid option")
        print()

def submit_training_launcher():

    print()
    print("===================================")
    print("SUBMIT TRAINING")
    print("===================================")

    submit_training()

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
        encoding="utf-8"
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

    job_id = input(
        "\nInsert job_id: "
    ).strip()

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
            encoding="utf-8"
        ) as f:

            experiment_record = json.load(f)

        print("===================================")

        print("experiment_id:")
        print(
            experiment_record.get(
                "experiment_id"
            )
        )

        print()
        print("status:")
        print(
            experiment_record.get(
                "status"
            )
        )

        print()
        print("expected_tree_count:")
        print(
            experiment_record.get(
                "expected_tree_count"
            )
        )

        print()
        print("completed_tree_count:")
        print(
            experiment_record.get(
                "completed_tree_count"
            )
        )

        print()
        print("assigned_workers:")
        print(
            experiment_record.get(
                "assigned_workers"
            )
        )

        print()


def see_experiments_launcher():

    print()
    print("===================================")
    print("SEE EXPERIMENTS")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

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

    job_id = input(
        "\nInsert job_id: "
    ).strip()

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
        encoding="utf-8"
    ) as f:

        experiment_record = json.load(f)

    print()

    print("status:")
    print(
        experiment_record.get(
            "status"
        )
    )

    print()
    print("expected_tree_count:")
    print(
        experiment_record.get(
            "expected_tree_count"
        )
    )

    print()
    print("completed_tree_count:")
    print(
        experiment_record.get(
            "completed_tree_count"
        )
    )

    print()
    print("assigned_workers:")
    print(
        experiment_record.get(
            "assigned_workers"
        )
    )

    print()
    print("validation_metrics:")
    print(
        experiment_record.get(
            "validation_metrics"
        )
    )

    print()


def see_validation_metrics_launcher():

    print()
    print("===================================")
    print("SEE VALIDATION METRICS")
    print("===================================")

    job_id = input(
        "\nInsert job_id: "
    ).strip()

    experiment_id = input(
        "Insert experiment_id: "
    ).strip()

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

    # fallback più robusto: cerca tutti i manifest e trova quello col model_id corretto
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

        X = X_df.head(rows).to_numpy(dtype=float)

        request = rf_pb2.SubmitInferenceRequest(
            model_id=model_id,
            features=matrix_to_proto(X),
        )

        with grpc.insecure_channel(
            MASTER_ADDRESS,
            options=GRPC_OPTIONS,
        ) as channel:

            stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

            response = stub.SubmitInference(
                request,
                timeout=120,
            )

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

    if response.predicted_labels:
        print()
        print("predicted_labels:")
        print(list(response.predicted_labels))

    if response.predicted_values:
        print()
        print("predicted_values:")
        print(list(response.predicted_values))

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

            if response.predicted_labels:
                correct = sum(
                    1
                    for pred, true in zip(response.predicted_labels, y_values)
                    if str(pred) == str(true)
                )

                accuracy = correct / len(y_values)

                print()
                print("local_accuracy:")
                print(round(accuracy, 4))

    print()


def submit_inference_launcher():

    print()
    print("===================================")
    print("SUBMIT INFERENCE")
    print("===================================")

    model_id = input(
        "\nInsert model_id: "
    ).strip()

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

    split_choice = input(
        "\nSelect option: "
    ).strip()

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

    rows_raw = input(
        "\nHow many rows? Default 5: "
    ).strip()

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


#RIVEDERE QUESTO CODICE PERCHé NON è PIù CONFORME CON IL PATHING DELL'ARCH
"""
def reset_shared_artifacts(root_path: str = "./shared_artifacts") -> None:
  
    Rimuove completamente la directory shared_artifacts
    e la ricrea vuota.

    Equivalente Python di:

        Remove-Item -Recurse -Force .\\shared_artifacts
        New-Item -ItemType Directory -Force .\\shared_artifacts

    Sicuro da chiamare prima di un test end-to-end.
  

    root = Path(root_path)

    # rimozione completa se esiste
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    # ricreazione directory
    root.mkdir(parents=True, exist_ok=True)
 """
def reset_shared_artifacts() -> None:
    """
    Rimuove gli artifact generati dai job, lasciando intatti i dataset.
    TODO MODIFICA CON ROOT DI EFS
    """

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

    choice = input(
        "\nSelect option: "
    ).strip()

    if choice == "1":
        reset_shared_artifacts()

    elif choice == "2":
        return

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
        print("7 -> Eliminate shared artifacts")
        print("0 -> Exit")

        choice = input(
            "\nSelect option: "
        ).strip()

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