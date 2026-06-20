from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc
from common.grpc_config import GRPC_OPTIONS
from common.prediction_io import (
    load_prediction_array,
    normalize_uri_to_path,
    path_to_file_uri,
)

ARTIFACT_ROOT = Path(
    os.getenv("SHARED_STORAGE_ROOT", "/mnt/efs/gp_artifacts")
).resolve()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )
def path_from_file_uri(uri: str) -> Path:
    uri = str(uri)

    if uri.startswith("file://"):
        return Path(uri.replace("file://", "", 1))

    return Path(uri)


def load_json(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path)

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved_path}")

    return json.loads(resolved_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved_path = Path(str(path))

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path.write_text(
        json.dumps(make_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def make_json_safe(value: Any) -> Any:
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

    return value


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

    master_host = os.getenv("MASTER_HOST", "127.0.0.1").strip()
    master_port = os.getenv("MASTER_PORT", "50051").strip()

    return [f"{master_host}:{master_port}"]

def normalized_grpc_options() -> list[tuple[str, int | str]]:
    """
    gRPC vuole option name stringa e value stringa/int.
    Questa normalizzazione evita errori tipo:
    'PosixPath' object has no attribute 'decode'
    """
    normalized: list[tuple[str, int | str]] = []

    for key, value in GRPC_OPTIONS:
        normalized_key = str(key)

        if isinstance(value, Path):
            normalized_value: int | str = str(value)
        else:
            normalized_value = value

        normalized.append((normalized_key, normalized_value))

    return normalized

def is_not_leader_message(message: str) -> bool:
    return "not leader" in str(message or "").lower()


def submit_inference_with_leader_discovery(
    model_id: str,
    features_uri: str,
    timeout_seconds: float,
) -> tuple[rf_pb2.SubmitInferenceResponse, str]:
    model_id = str(model_id)
    features_uri = str(features_uri)

    request = rf_pb2.SubmitInferenceRequest(
        model_id=model_id,
        features_uri=features_uri,
    )

    last_error: Exception | None = None

    for raw_master_address in load_master_addresses():
        master_address = str(raw_master_address).strip()

        if not master_address:
            continue

        try:
            print(
                "[distributed_inference_benchmark] Trying master "
                f"{master_address}",
                flush=True,
            )

            with grpc.insecure_channel(
                master_address,
                options=normalized_grpc_options(),
            ) as channel:
                stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

                response = stub.SubmitInference(
                    request,
                    timeout=float(timeout_seconds),
                )

            if response.success:
                print(
                    "[distributed_inference_benchmark] SubmitInference "
                    f"accepted by {master_address}",
                    flush=True,
                )
                return response, master_address

            if is_not_leader_message(response.error):
                last_error = RuntimeError(response.error)
                print(
                    "[distributed_inference_benchmark] Master is not leader: "
                    f"{master_address} -> {response.error}",
                    flush=True,
                )
                continue

            raise RuntimeError(response.error or "SubmitInference failed")

        except Exception as exc:
            last_error = exc
            print(
                "[distributed_inference_benchmark] SubmitInference failed on "
                f"{master_address}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    raise RuntimeError(
        f"No master leader accepted SubmitInference: {last_error}"
    )


def find_manifest_by_model_id(model_id: str) -> Path:
    direct_path = ARTIFACT_ROOT / "models" / model_id / "manifest.json"

    if direct_path.exists():
        return direct_path

    models_root = ARTIFACT_ROOT / "models"

    if models_root.exists():
        for manifest_path in models_root.glob("*/manifest.json"):
            try:
                payload = load_json(manifest_path)
            except Exception:
                continue

            if payload.get("model_id") == model_id:
                return manifest_path

    raise FileNotFoundError(f"Manifest not found for model_id={model_id}")


def select_features_uri(manifest: dict[str, Any], split: str) -> str:
    key = f"{split}_features_uri"
    value = str(manifest.get(key) or "").strip()

    if not value:
        raise ValueError(f"Manifest does not contain {key}")

    return value


def select_labels_uri(manifest: dict[str, Any], split: str) -> str:
    key = f"{split}_labels_uri"
    value = str(manifest.get(key) or "").strip()

    if not value:
        raise ValueError(f"Manifest does not contain {key}")

    return value


def load_feature_frame(features_uri: str) -> pd.DataFrame:
    return pd.read_parquet(normalize_uri_to_path(features_uri))


def load_label_values(labels_uri: str, rows: int | None) -> np.ndarray:
    labels_df = pd.read_parquet(path_from_file_uri(str(labels_uri)))
    if rows is not None:
        labels_df = labels_df.head(rows)

    return labels_df.to_numpy().reshape(-1)


def build_benchmark_features_uri(
    model_id: str,
    split: str,
    source_features_uri: str,
    rows: int | None,
    manifest: dict[str, Any],
) -> tuple[str, int, bool]:
    """
    Costruisce l'input di inferenza nello stesso modo di training_debug_cli.py:
    - legge il parquet delle feature preparate;
    - riordina le colonne secondo manifest["feature_names"];
    - materializza un nuovo parquet su EFS;
    - passa al master un file:// URI pulito.
    """
    model_id = str(model_id)
    split = str(split)
    source_features_uri = str(source_features_uri)

    features_path = path_from_file_uri(source_features_uri)

    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {features_path}")

    X_df = pd.read_parquet(features_path)

    feature_names = manifest.get("feature_names") or []

    if feature_names:
        missing_features = [
            feature
            for feature in feature_names
            if feature not in X_df.columns
        ]

        if missing_features:
            raise ValueError(
                "Some manifest features are missing from features parquet: "
                f"{missing_features}"
            )

        X_df = X_df[feature_names]

    if rows is not None:
        if rows <= 0:
            raise ValueError("rows must be positive or 'all'")

        X_df = X_df.head(rows)

    X_df = X_df.reset_index(drop=True)

    output_dir = ARTIFACT_ROOT / "benchmark_inference_inputs" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_label = "all" if rows is None else str(rows)

    output_path = (
        output_dir
        / f"{split}_head_{rows_label}_{int(time.time())}.parquet"
    )

    X_df.to_parquet(output_path, index=False)

    return path_to_file_uri(output_path), int(len(X_df)), True

def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction/label length mismatch: "
            f"{len(y_pred)} predictions, {len(y_true)} labels"
        )

    if task_type == "classification":
        y_true_str = np.asarray([str(value) for value in y_true])
        y_pred_str = np.asarray([str(value) for value in y_pred])

        labels = sorted(
            pd.Series(y_true_str).dropna().unique().tolist()
        )

        return make_json_safe(
            {
                "accuracy": accuracy_score(y_true_str, y_pred_str),
                "classification_report": classification_report(
                    y_true_str,
                    y_pred_str,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    y_true_str,
                    y_pred_str,
                    labels=labels,
                ).tolist(),
                "labels": labels,
            }
        )

    if task_type == "regression":
        y_true_float = y_true.astype(float)
        y_pred_float = y_pred.astype(float)

        mse = mean_squared_error(y_true_float, y_pred_float)

        return make_json_safe(
            {
                "r2": r2_score(y_true_float, y_pred_float),
                "mae": mean_absolute_error(y_true_float, y_pred_float),
                "mse": mse,
                "rmse": np.sqrt(mse),
            }
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def load_inference_metadata_from_prediction_uri(
    prediction_uri: str,
) -> dict[str, Any] | None:
    try:
        prediction_path = normalize_uri_to_path(prediction_uri)
        metadata_path = prediction_path.parent / "metadata.json"

        if metadata_path.exists():
            return load_json(metadata_path)

    except Exception:
        return None

    return None


def extract_local_inference_time(
    local_baseline: dict[str, Any] | None,
    split: str,
) -> float | None:
    if local_baseline is None:
        return None

    key = f"{split}_inference_time_seconds"
    value = local_baseline.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest_json:
        manifest_path = Path(args.manifest_json)

        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path

    else:
        manifest_path = find_manifest_by_model_id(args.model_id)

    manifest = load_json(manifest_path)

    model_id = str(manifest.get("model_id") or args.model_id)
    task_type = str(
        manifest.get("model_type")
        or manifest.get("task_type")
        or ""
    )

    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"Unsupported or missing model task type: {task_type}"
        )

    source_features_uri = select_features_uri(manifest, args.split)
    labels_uri = select_labels_uri(manifest, args.split)

    rows = None if args.rows == "all" else int(args.rows)

    benchmark_features_uri, n_input_rows, materialized_slice = (
        build_benchmark_features_uri(
            model_id=model_id,
            split=args.split,
            source_features_uri=source_features_uri,
            rows=rows,
            manifest=manifest,
        )
    )

    y_true = load_label_values(labels_uri, rows=rows)

    start = time.perf_counter()

    response, accepted_master = submit_inference_with_leader_discovery(
        model_id=model_id,
        features_uri=benchmark_features_uri,
        timeout_seconds=args.timeout_seconds,
    )

    distributed_elapsed = time.perf_counter() - start

    predictions = load_prediction_array(response.prediction_uri).reshape(-1)

    metrics = compute_metrics(
        task_type=task_type,
        y_true=y_true,
        y_pred=predictions,
    )

    inference_metadata = load_inference_metadata_from_prediction_uri(
        response.prediction_uri
    )

    local_baseline = (
        load_json(args.local_baseline_json)
        if args.local_baseline_json
        else None
    )

    local_inference_time = extract_local_inference_time(
        local_baseline,
        args.split,
    )

    if local_inference_time is not None and distributed_elapsed > 0:
        local_over_distributed = (
            local_inference_time / distributed_elapsed
        )
    else:
        local_over_distributed = None

    result = {
        "benchmark_type": "distributed_submit_inference_end_to_end",
        "model_id": model_id,
        "manifest_path": str(manifest_path),
        "split": args.split,
        "task_type": task_type,
        "accepted_master": accepted_master,
        "source_features_uri": source_features_uri,
        "benchmark_features_uri": benchmark_features_uri,
        "labels_uri": labels_uri,
        "materialized_input_slice": materialized_slice,
        "requested_rows": args.rows,
        "n_input_rows": n_input_rows,
        "response": {
            "success": bool(response.success),
            "error": response.error,
            "task_type": response.task_type,
            "prediction_uri": response.prediction_uri,
            "n_rows": int(response.n_rows),
            "n_cols": int(response.n_cols),
        },
        "distributed": {
            "inference_time_seconds": float(distributed_elapsed),
            "metrics": metrics,
        },
        "local": {
            "baseline_json": args.local_baseline_json,
            "inference_time_seconds": local_inference_time,
        },
        "comparison": {
            "local_over_distributed_inference_time_ratio": (
                local_over_distributed
            ),
        },
        "inference_metadata": inference_metadata,
        "created_at": time.time(),
    }

    if args.output_json:
        write_json(args.output_json, result)

    return make_json_safe(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end dell'inferenza distribuita via SubmitInference."
        )
    )

    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest-json", default=None)

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )

    parser.add_argument(
        "--rows",
        default="all",
        help="Numero righe da usare oppure 'all'.",
    )

    parser.add_argument("--local-baseline-json", default=None)

    parser.add_argument(
        "--output-json",
        default=(
            "performance_evaluation/results/inference/"
            "distributed_inference_benchmark.json"
        ),
    )

    parser.add_argument("--timeout-seconds", type=float, default=600.0)

    return parser.parse_args()


def main() -> None:
    load_env_file(PROJECT_ROOT / ".env.client")

    args = parse_args()
    result = run_benchmark(args)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()