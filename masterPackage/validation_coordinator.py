from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)

from common.contracts import TreeArtifactMetadata, ValidationMetrics


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

    Nota:
    questa versione usa i tree artifact già addestrati.

    Per classificazione supporta due formati di risposta dal worker:
    1) formato corretto/atteso:
       matrice (n_samples, n_classes) con voti parziali
    2) fallback temporaneo:
       matrice (n_samples, 1) con classe predetta localmente dal worker
       -> utile finché il lato worker non viene riallineato del tutto
    """

    def __init__(
        self,
        leadership_guard,
        worker_registry: WorkerRegistryLike,
        worker_client,
        max_parallel_requests: int | None = None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.worker_client = worker_client
        self.max_parallel_requests = max_parallel_requests

    def validate_experiment(
        self,
        experiment_id: str,
        task_type: str,
        validation_features_uri: str,
        validation_labels_uri: str,
        tree_artifacts: list[TreeArtifactMetadata],
        class_labels: Sequence[str] | None = None,
    ) -> ValidationResult:
        self.leadership_guard.require_leader()

        if not tree_artifacts:
            raise ValueError(
                f"Experiment '{experiment_id}' has no tree artifacts for validation"
            )

        X_val = self._read_parquet_dataframe(validation_features_uri).to_numpy(dtype=float)
        y_val = self._read_target_vector(validation_labels_uri)

        if X_val.ndim != 2:
            raise ValueError("Validation features must be a 2D matrix")
        if X_val.shape[0] == 0:
            raise ValueError("Validation split is empty")
        if y_val.shape[0] != X_val.shape[0]:
            raise ValueError(
                "Validation features/labels size mismatch: "
                f"{X_val.shape[0]} rows vs {y_val.shape[0]} labels"
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

        responses = self._collect_prediction_responses(
            experiment_id=experiment_id,
            task_type=task_type,
            features=X_val,
            class_labels=class_labels,
            assignments=assignments,
        )

        if task_type == "classification":
            return self._build_classification_result(
                experiment_id=experiment_id,
                y_true=y_val,
                responses=responses,
                class_labels=class_labels,
                n_features=X_val.shape[1],
            )

        if task_type == "regression":
            return self._build_regression_result(
                experiment_id=experiment_id,
                y_true=y_val,
                responses=responses,
                tree_count=len(tree_artifacts),
                n_features=X_val.shape[1],
            )

        raise ValueError(f"Unsupported task_type '{task_type}'")

    def _collect_prediction_responses(
        self,
        experiment_id: str,
        task_type: str,
        features: np.ndarray,
        class_labels: Sequence[str] | None,
        assignments: list[tuple[WorkerLike, list[str]]],
    ) -> list:
        responses = []
        max_workers = self.max_parallel_requests or len(assignments)
        max_workers = min(max_workers, len(assignments))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for worker, uri_shard in assignments:
                future = pool.submit(
                    self.worker_client.predict_shard,
                    worker.host,
                    worker.port,
                    experiment_id,   # model_id temporaneo per validation
                    experiment_id,
                    task_type,
                    features,
                    uri_shard,
                    class_labels,
                )
                future_map[future] = (worker, uri_shard)

            for future in as_completed(future_map):
                worker, uri_shard = future_map[future]
                result = future.result()

                if not result.success:
                    retry_worker = self.worker_registry.get_retry_candidate(
                        exclude_worker_id=worker.worker_id
                    )
                    if retry_worker is not None:
                        result = self.worker_client.predict_shard(
                            retry_worker.host,
                            retry_worker.port,
                            experiment_id,   # model_id temporaneo per validation
                            experiment_id,
                            task_type,
                            features,
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
        n_features: int,
    ) -> ValidationResult:
        resolved_class_labels = self._resolve_class_labels(y_true, class_labels)
        n_samples = y_true.shape[0]
        n_classes = len(resolved_class_labels)

        aggregated_votes = np.zeros((n_samples, n_classes), dtype=float)

        for response in responses:
            values = response.values

            # formato corretto atteso: voti parziali per classe
            if values.shape == aggregated_votes.shape:
                aggregated_votes += values
                continue

            # fallback temporaneo: classe locale già decisa dal worker
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
        predicted_labels = [resolved_class_labels[index] for index in final_indices]

        if self._is_integer_encoded_labels(y_true, n_classes):
            y_true_for_metrics = y_true.astype(int)
            y_pred_for_metrics = final_indices.astype(int)

            report = classification_report(
                y_true_for_metrics,
                y_pred_for_metrics,
                output_dict=True,
                zero_division=0,
            )
            confusion = confusion_matrix(
                y_true_for_metrics,
                y_pred_for_metrics,
            ).tolist()
            accuracy = float(accuracy_score(y_true_for_metrics, y_pred_for_metrics))
        else:
            y_true_for_metrics = [str(item) for item in y_true.tolist()]
            y_pred_for_metrics = predicted_labels

            report = classification_report(
                y_true_for_metrics,
                y_pred_for_metrics,
                output_dict=True,
                zero_division=0,
            )
            confusion = confusion_matrix(
                y_true_for_metrics,
                y_pred_for_metrics,
            ).tolist()
            accuracy = float(accuracy_score(y_true_for_metrics, y_pred_for_metrics))

        metrics = ValidationMetrics(
            experiment_id=experiment_id,
            accuracy=accuracy,
            classification_report=report,
            confusion_matrix=confusion,
            feature_importances=[0.0] * n_features,
            evaluated_at=time.time(),
        )

        return ValidationResult(
            metrics=metrics,
            predicted_labels=predicted_labels,
            predicted_values=None,
        )

    def _build_regression_result(
        self,
        experiment_id: str,
        y_true: np.ndarray,
        responses: list,
        tree_count: int,
        n_features: int,
    ) -> ValidationResult:
        n_samples = y_true.shape[0]
        aggregated_sum = np.zeros((n_samples, 1), dtype=float)

        for response in responses:
            values = response.values
            if values.shape != aggregated_sum.shape:
                raise ValueError(
                    "Invalid regression shard response shape: "
                    f"expected {aggregated_sum.shape}, got {values.shape}"
                )
            aggregated_sum += values

        predicted_values = (aggregated_sum[:, 0] / tree_count).tolist()

        mse = float(mean_squared_error(y_true, predicted_values))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, predicted_values))

        metrics = ValidationMetrics(
            experiment_id=experiment_id,
            accuracy=0.0,
            classification_report={
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            },
            confusion_matrix=[],
            feature_importances=[0.0] * n_features,
            evaluated_at=time.time(),
        )

        return ValidationResult(
            metrics=metrics,
            predicted_labels=None,
            predicted_values=predicted_values,
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
        path = self._normalize_uri(uri)
        return pd.read_parquet(path)

    def _normalize_uri(self, uri: str) -> str:
        if uri.startswith("file://"):
            return uri.replace("file://", "", 1)
        return uri