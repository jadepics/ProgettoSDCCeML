from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol

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

    def get_retry_candidate(self, exclude_worker_id: str | None = None) -> Optional[WorkerLike]:
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
        class_labels: list[str] | None = None,
    ) -> ValidationResult:
        self.leadership_guard.require_leader()

        task_type = task_type.strip().lower()
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")

        X = self._read_parquet_dataframe(validation_features_uri).to_numpy(dtype=float)
        y_df = self._read_parquet_dataframe(validation_labels_uri)

        if X.ndim != 2:
            raise ValueError("Validation features must be a 2D matrix")
        if X.shape[0] == 0:
            raise ValueError("Validation set is empty")
        if y_df.shape[1] != 1:
            raise ValueError("Validation labels parquet must contain exactly one column")

        y_true = y_df.iloc[:, 0].to_numpy()

        alive_workers = self.worker_registry.alive_workers()
        if not alive_workers:
            raise RuntimeError("No alive workers available for validation")

        tree_uris = [artifact.artifact_uri for artifact in tree_artifacts]
        if not tree_uris:
            raise RuntimeError("No tree artifacts available for validation")

        worker_shards = self._assign_tree_uris_to_workers(alive_workers, tree_uris)
        if not worker_shards:
            raise RuntimeError("No worker shards could be built for validation")

        responses = []
        max_workers = self.max_parallel_requests or len(worker_shards)
        max_workers = min(max_workers, len(worker_shards))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for worker, uri_shard in worker_shards:
                future = pool.submit(
                    self.worker_client.predict_shard,
                    worker.host,
                    worker.port,
                    "validation",
                    experiment_id,
                    task_type,
                    X,
                    uri_shard,
                    class_labels or [],
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
                            "validation",
                            experiment_id,
                            task_type,
                            X,
                            uri_shard,
                            class_labels or [],
                        )

                if not result.success:
                    raise RuntimeError(
                        f"Validation shard failed on worker {worker.worker_id}: "
                        f"{result.error_message}"
                    )

                responses.append(result)

        if task_type == "classification":
            effective_class_labels = list(class_labels or [])
            if not effective_class_labels:
                effective_class_labels = sorted(np.unique(y_true.astype(str)).tolist())

            aggregated_votes = np.zeros((X.shape[0], len(effective_class_labels)), dtype=float)

            for response in responses:
                if response.values.shape != aggregated_votes.shape:
                    raise ValueError(
                        "Invalid classification validation shard response shape: "
                        f"expected {aggregated_votes.shape}, got {response.values.shape}"
                    )
                aggregated_votes += response.values

            predicted_indices = np.argmax(aggregated_votes, axis=1)
            predicted_labels = [effective_class_labels[index] for index in predicted_indices]
            y_true_labels = y_true.astype(str)

            metrics = ValidationMetrics(
                experiment_id=experiment_id,
                accuracy=float(accuracy_score(y_true_labels, predicted_labels)),
                classification_report=classification_report(
                    y_true_labels,
                    predicted_labels,
                    output_dict=True,
                    zero_division=0,
                ),
                confusion_matrix=confusion_matrix(
                    y_true_labels,
                    predicted_labels,
                    labels=effective_class_labels,
                ).tolist(),
                feature_importances=[],
                evaluated_at=self._now_ts(),
            )

            return ValidationResult(
                metrics=metrics,
                predicted_labels=predicted_labels,
                predicted_values=None,
            )

        aggregated_sum = np.zeros((X.shape[0], 1), dtype=float)

        for response in responses:
            if response.values.shape != aggregated_sum.shape:
                raise ValueError(
                    "Invalid regression validation shard response shape: "
                    f"expected {aggregated_sum.shape}, got {response.values.shape}"
                )
            aggregated_sum += response.values

        predicted_values = (aggregated_sum[:, 0] / len(tree_artifacts)).tolist()
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
            feature_importances=[],
            evaluated_at=self._now_ts(),
        )

        return ValidationResult(
            metrics=metrics,
            predicted_labels=None,
            predicted_values=predicted_values,
        )

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

    def _now_ts(self) -> float:
        import time
        return time.time()