from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from common.contracts import TreeArtifactMetadata, ValidationMetrics
from common.prediction_io import (
    load_prediction_array,
    normalize_uri_to_path,
    path_to_file_uri,
)


@dataclass(slots=True)
class ValidationResult:
    metrics: ValidationMetrics
    predicted_labels: list[str] | None
    predicted_values: list[float] | None


class WorkerLike(Protocol):
    worker_id: str
    host: str
    port: int


class WorkerRegistryLike(Protocol):
    def alive_workers(self) -> list[WorkerLike]:
        ...

    def get_retry_candidate(
        self,
        exclude_worker_id: str | None = None,
    ) -> Optional[WorkerLike]:
        ...


class ValidationCoordinator:
    """
    Responsabilità:
    - leggere il validation split persistito
    - distribuire gli alberi ai worker vivi
    - raccogliere predizioni parziali
    - aggregare il risultato finale
    - produrre ValidationMetrics

    Nuovo flusso scalabile:
    - il master non invia più X_val dentro gRPC;
    - il master invia validation_features_uri;
    - ogni worker legge X_val da storage condiviso;
    - ogni worker salva le predizioni parziali su EFS;
    - il master legge prediction_uri e aggrega come prima.

    La strategia distribuita resta tree-parallel:
    ogni worker valuta un sottoinsieme di alberi su tutte le righe.
    """

    def __init__(
        self,
        leadership_guard,
        worker_registry: WorkerRegistryLike,
        worker_client,
        storage_layout,
        max_parallel_requests: int | None = None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.worker_client = worker_client
        self.storage_layout = storage_layout
        self.max_parallel_requests = max_parallel_requests

    def validate_experiment(
        self,
        job_id: str,
        experiment_id: str,
        task_type: str,
        validation_features_uri: str,
        validation_labels_uri: str,
        tree_artifacts: list[TreeArtifactMetadata],
        class_labels: Sequence[str] | None = None,
    ) -> ValidationResult:
        self.leadership_guard.require_leader()

        if not job_id:
            raise ValueError("job_id is required for validation")

        if not tree_artifacts:
            raise ValueError(
                f"Experiment '{experiment_id}' has no tree artifacts for validation"
            )

        # Il master legge le feature solo per controlli di shape e nomi colonna.
        # Non le passa più ai worker via gRPC.
        X_val_df = self._read_parquet_dataframe(validation_features_uri)
        feature_names = list(X_val_df.columns)
        n_samples = int(X_val_df.shape[0])
        n_features = int(X_val_df.shape[1])

        y_val = self._read_target_vector(validation_labels_uri)

        if X_val_df.ndim != 2:
            raise ValueError("Validation features must be a 2D matrix")
        if n_samples == 0:
            raise ValueError("Validation split is empty")
        if y_val.shape[0] != n_samples:
            raise ValueError(
                "Validation features/labels size mismatch: "
                f"{n_samples} rows vs {y_val.shape[0]} labels"
            )

        alive_workers = self.worker_registry.alive_workers()
        if not alive_workers:
            raise RuntimeError("No alive workers available for validation")

        tree_uris = [artifact.artifact_uri for artifact in tree_artifacts]
        assignments = self._assign_tree_uris_to_workers(
            workers=alive_workers,
            tree_uris=tree_uris,
        )
        if not assignments:
            raise RuntimeError("No validation assignments could be built")

        prediction_dir = self.storage_layout.prediction_dir(
            job_id=job_id,
            experiment_id=experiment_id,
            phase="validation",
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_output_dir = path_to_file_uri(prediction_dir)

        responses = self._collect_prediction_responses(
            experiment_id=experiment_id,
            task_type=task_type,
            features_uri=validation_features_uri,
            prediction_output_dir=prediction_output_dir,
            class_labels=class_labels,
            assignments=assignments,
        )

        if task_type == "classification":
            return self._build_classification_result(
                experiment_id=experiment_id,
                y_true=y_val,
                responses=responses,
                class_labels=class_labels,
                tree_artifacts=tree_artifacts,
                n_features=n_features,
                feature_names=feature_names,
            )

        if task_type == "regression":
            return self._build_regression_result(
                experiment_id=experiment_id,
                y_true=y_val,
                responses=responses,
                tree_count=len(tree_artifacts),
                tree_artifacts=tree_artifacts,
                n_features=n_features,
                feature_names=feature_names,
            )

        raise ValueError(f"Unsupported task_type '{task_type}'")

    def _collect_prediction_responses(
        self,
        experiment_id: str,
        task_type: str,
        features_uri: str,
        prediction_output_dir: str,
        class_labels: Sequence[str] | None,
        assignments: list[tuple[WorkerLike, list[str]]],
    ) -> list:
        responses = []
        max_workers = self.max_parallel_requests or len(assignments)
        max_workers = min(max_workers, len(assignments))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for index, (worker, uri_shard) in enumerate(assignments):
                prediction_id = (
                    f"{experiment_id}_validation_pred_{index:06d}"
                )

                future = pool.submit(
                    self.worker_client.predict_shard_from_uri,
                    worker.host,
                    worker.port,
                    experiment_id,  # model_id temporaneo per validation
                    experiment_id,
                    task_type,
                    features_uri,
                    prediction_output_dir,
                    prediction_id,
                    uri_shard,
                    class_labels,
                )
                future_map[future] = (worker, uri_shard, prediction_id)

            for future in as_completed(future_map):
                worker, uri_shard, prediction_id = future_map[future]
                result = future.result()

                if not result.success:
                    retry_worker = self.worker_registry.get_retry_candidate(
                        exclude_worker_id=worker.worker_id
                    )

                    if retry_worker is not None:
                        result = self.worker_client.predict_shard_from_uri(
                            retry_worker.host,
                            retry_worker.port,
                            experiment_id,  # model_id temporaneo per validation
                            experiment_id,
                            task_type,
                            features_uri,
                            prediction_output_dir,
                            prediction_id,
                            uri_shard,
                            class_labels,
                        )

                if not result.success:
                    raise RuntimeError(
                        f"Validation shard failed on worker {worker.worker_id}: "
                        f"{result.error_message}"
                    )

                responses.append(result)

        return responses

    def _build_classification_result(
        self,
        experiment_id: str,
        y_true: np.ndarray,
        responses: list,
        class_labels: Sequence[str] | None,
        tree_artifacts: Sequence[TreeArtifactMetadata],
        n_features: int,
        feature_names: Sequence[str],
    ) -> ValidationResult:
        resolved_class_labels = self._resolve_class_labels(y_true, class_labels)
        n_samples = y_true.shape[0]
        n_classes = len(resolved_class_labels)

        aggregated_votes = np.zeros((n_samples, n_classes), dtype=float)

        for response in responses:
            values = self._response_values(response)

            # Formato corretto atteso: voti parziali per classe.
            if values.shape == aggregated_votes.shape:
                aggregated_votes += values
                continue

            # Fallback temporaneo: classe locale già decisa dal worker.
            if values.shape == (n_samples, 1):
                predicted_indices = np.rint(values[:, 0]).astype(int)
                if np.any(predicted_indices < 0) or np.any(predicted_indices >= n_classes):
                    raise ValueError(
                        "Invalid classification shard response values: "
                        f"indices out of range for {n_classes} classes"
                    )
                aggregated_votes[np.arange(n_samples), predicted_indices] += 1.0
                continue

            raise ValueError(
                "Invalid classification shard response shape: "
                f"expected {(n_samples, n_classes)} or {(n_samples, 1)}, got {values.shape}"
            )

        final_indices = np.argmax(aggregated_votes, axis=1)
        predicted_labels: list[str] = [
            resolved_class_labels[int(index)]
            for index in final_indices.tolist()
        ]

        if self._is_integer_encoded_labels(y_true, n_classes):
            y_true_for_metrics = y_true.astype(int)
            y_pred_for_metrics = final_indices.astype(int)

            report = self._metric_dict(
                classification_report(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    output_dict=True,
                    zero_division=0,
                )
            )
            confusion = cast(
                list[list[int]],
                confusion_matrix(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                ).astype(int).tolist(),
            )
            accuracy = float(accuracy_score(y_true_for_metrics, y_pred_for_metrics))
            balanced_accuracy = float(
                balanced_accuracy_score(y_true_for_metrics, y_pred_for_metrics)
            )
            macro_f1 = float(
                f1_score(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    average="macro",
                    zero_division=0,
                )
            )
            weighted_f1 = float(
                f1_score(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    average="weighted",
                    zero_division=0,
                )
            )
        else:
            y_true_for_metrics = [str(item) for item in y_true.tolist()]
            y_pred_for_metrics = predicted_labels

            report = self._metric_dict(
                classification_report(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    output_dict=True,
                    zero_division=0,
                )
            )
            confusion = cast(
                list[list[int]],
                confusion_matrix(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                ).astype(int).tolist(),
            )
            accuracy = float(accuracy_score(y_true_for_metrics, y_pred_for_metrics))
            balanced_accuracy = float(
                balanced_accuracy_score(y_true_for_metrics, y_pred_for_metrics)
            )
            macro_f1 = float(
                f1_score(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    average="macro",
                    zero_division=0,
                )
            )
            weighted_f1 = float(
                f1_score(
                    y_true_for_metrics,
                    y_pred_for_metrics,
                    average="weighted",
                    zero_division=0,
                )
            )

        feature_importances = self._aggregate_feature_importances(
            tree_artifacts=tree_artifacts,
            n_features=n_features,
        )
        feature_importances_by_name = self._map_feature_importances_by_name(
            feature_names=feature_names,
            feature_importances=feature_importances,
        )

        metrics = ValidationMetrics(
            experiment_id=experiment_id,
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            classification_report=report,
            confusion_matrix=confusion,
            feature_importances=feature_importances,
            feature_importances_by_name=feature_importances_by_name,
            evaluated_at=time.time(),
        )

        return ValidationResult(
            metrics=metrics,
            predicted_labels=predicted_labels,
            predicted_values=None,
        )

    def _metric_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return cast(dict[str, Any], value)

        raise TypeError(
            f"Expected sklearn classification_report output_dict=True to return dict, "
            f"got {type(value).__name__}"
        )

    def _scalar_metric(self, value: Any) -> float:
        arr = np.asarray(value, dtype=float)

        if arr.size != 1:
            raise ValueError(f"Expected scalar metric, got shape={arr.shape}")

        return float(arr.reshape(-1)[0])

    def _build_regression_result(
        self,
        experiment_id: str,
        y_true: np.ndarray,
        responses: list,
        tree_count: int,
        tree_artifacts: Sequence[TreeArtifactMetadata],
        n_features: int,
        feature_names: Sequence[str],
    ) -> ValidationResult:
        n_samples = y_true.shape[0]
        aggregated_sum = np.zeros((n_samples, 1), dtype=float)

        for response in responses:
            values = self._response_values(response)

            if values.shape != aggregated_sum.shape:
                raise ValueError(
                    "Invalid regression shard response shape: "
                    f"expected {aggregated_sum.shape}, got {values.shape}"
                )

            aggregated_sum += values

        y_true_float = np.asarray(y_true, dtype=float).reshape(-1)
        predicted_values_array = aggregated_sum[:, 0] / float(tree_count)
        predicted_values = predicted_values_array.tolist()

        mae = self._scalar_metric(
            mean_absolute_error(y_true_float, predicted_values_array)
        )
        mse = self._scalar_metric(
            mean_squared_error(y_true_float, predicted_values_array)
        )
        rmse = float(np.sqrt(mse))
        r2 = self._scalar_metric(
            r2_score(y_true_float, predicted_values_array)
        )

        feature_importances = self._aggregate_feature_importances(
            tree_artifacts=tree_artifacts,
            n_features=n_features,
        )
        feature_importances_by_name = self._map_feature_importances_by_name(
            feature_names=feature_names,
            feature_importances=feature_importances,
        )
        metrics = ValidationMetrics(
            experiment_id=experiment_id,

            accuracy=None,
            classification_report=None,
            confusion_matrix=None,

            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,

            feature_importances=feature_importances,
            feature_importances_by_name=feature_importances_by_name,
            evaluated_at=time.time(),
        )

        return ValidationResult(
            metrics=metrics,
            predicted_labels=None,
            predicted_values=predicted_values,
        )

    def _response_values(self, response) -> np.ndarray:
        """
        Rende uniforme il vecchio e il nuovo percorso.

        Vecchio:
          response.values contiene già la matrice.

        Nuovo:
          response.prediction_uri punta a un file .npy salvato dal worker.
        """
        values = getattr(response, "values", None)

        if values is not None:
            arr = np.asarray(values, dtype=float)

            if arr.size > 0:
                if arr.ndim == 1:
                    if response.n_rows > 0 and response.n_cols > 0:
                        return arr.reshape(response.n_rows, response.n_cols)
                    return arr.reshape(-1, 1)

                if arr.ndim == 2:
                    return arr

                raise ValueError(
                    f"Invalid in-memory prediction values ndim={arr.ndim}"
                )

        prediction_uri_raw = getattr(response, "prediction_uri", None)
        prediction_uri = str(prediction_uri_raw) if prediction_uri_raw else ""

        if prediction_uri:
            arr = load_prediction_array(prediction_uri)

            expected_rows = int(getattr(response, "n_rows", 0) or 0)
            expected_cols = int(getattr(response, "n_cols", 0) or 0)

            if expected_rows > 0 and expected_cols > 0:
                expected_shape = (expected_rows, expected_cols)
                if arr.shape != expected_shape:
                    raise ValueError(
                        "Prediction URI shape mismatch: "
                        f"expected {expected_shape}, got {arr.shape}"
                    )

            return arr.astype(float, copy=False)

        raise ValueError(
            "Prediction response has neither in-memory values nor prediction_uri"
        )

    def _resolve_class_labels(
        self,
        y_true: np.ndarray,
        class_labels: Sequence[str] | None,
    ) -> list[str]:
        if class_labels:
            return [str(label) for label in class_labels]

        if self._is_integer_encoded_labels(y_true):
            max_label = int(np.max(y_true))
            return [str(index) for index in range(max_label + 1)]

        raise ValueError(
            "class_labels are required for classification when validation labels "
            "are not integer-encoded"
        )

    def _is_integer_encoded_labels(
        self,
        y_true: np.ndarray,
        n_classes: int | None = None,
    ) -> bool:
        if y_true.ndim != 1:
            return False

        if not np.issubdtype(y_true.dtype, np.number):
            return False

        rounded = np.rint(y_true).astype(int)
        if not np.allclose(y_true, rounded):
            return False

        if np.any(rounded < 0):
            return False

        if n_classes is not None and np.any(rounded >= n_classes):
            return False

        return True

    def _read_target_vector(self, uri: str) -> np.ndarray:
        df = self._read_parquet_dataframe(uri)

        if df.shape[1] != 1:
            raise ValueError(
                f"Expected a single target column in '{uri}', found {df.shape[1]}"
            )

        return df.iloc[:, 0].to_numpy()

    def _assign_tree_uris_to_workers(
        self,
        workers: list[WorkerLike],
        tree_uris: list[str],
    ) -> list[tuple[WorkerLike, list[str]]]:
        if not workers or not tree_uris:
            return []

        ordered_workers = sorted(workers, key=lambda worker: worker.worker_id)
        shard_count = min(len(ordered_workers), len(tree_uris))
        buckets: list[list[str]] = [[] for _ in range(shard_count)]

        for index, uri in enumerate(tree_uris):
            buckets[index % shard_count].append(uri)

        assignments: list[tuple[WorkerLike, list[str]]] = []
        for worker, bucket in zip(ordered_workers[:shard_count], buckets):
            if bucket:
                assignments.append((worker, bucket))

        return assignments

    def _read_parquet_dataframe(self, uri: str) -> pd.DataFrame:
        path = normalize_uri_to_path(uri)
        return pd.read_parquet(path)

    def _aggregate_feature_importances(
        self,
        tree_artifacts: Sequence[TreeArtifactMetadata],
        n_features: int,
    ) -> list[float]:
        vectors: list[np.ndarray] = []

        for artifact in tree_artifacts:
            raw = getattr(artifact, "feature_importances", None)
            if not raw:
                continue

            vec = np.asarray(raw, dtype=float).ravel()

            if vec.size != n_features:
                padded = np.zeros(n_features, dtype=float)
                limit = min(n_features, vec.size)
                padded[:limit] = vec[:limit]
                vec = padded

            vectors.append(vec)

        if not vectors:
            return [0.0] * n_features

        return np.mean(np.vstack(vectors), axis=0).tolist()

    def _map_feature_importances_by_name(
        self,
        feature_names: Sequence[str],
        feature_importances: Sequence[float],
    ) -> dict[str, float]:
        mapped: dict[str, float] = {}

        limit = min(len(feature_names), len(feature_importances))

        for index in range(limit):
            mapped[str(feature_names[index])] = float(feature_importances[index])

        return dict(
            sorted(
                mapped.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )