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

from common.contracts import TreeArtifactMetadata, ValidationMetrics, ExperimentRecord


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

    def _validate_experiment(
            self,
            job_id: str,
            experiment: ExperimentRecord,
            tree_artifacts: list[TreeArtifactMetadata],
    ):
        job_record = self._load_job_or_raise(job_id)
        prepared_dataset = job_record.prepared_dataset
        if prepared_dataset is None:
            raise ValueError(f"Job '{job_id}' has no prepared dataset")

        return self.validation_coordinator.validate_experiment(
            experiment_id=experiment.experiment_id,
            task_type=job_record.training_request.task_type,
            validation_features_uri=prepared_dataset.validation_features_uri,
            validation_labels_uri=prepared_dataset.validation_labels_uri,
            tree_artifacts=tree_artifacts,
            class_labels=prepared_dataset.class_labels,
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