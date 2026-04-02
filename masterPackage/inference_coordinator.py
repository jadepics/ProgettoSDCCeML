from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(slots=True)
class InferenceResult:
    task_type: str
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


class InferenceCoordinator:
    """
    Responsabilità:
    - caricare il modello finale dal ModelRepository
    - suddividere gli alberi tra i worker vivi
    - inviare richieste di inferenza parziale ai worker tramite WorkerClient
    - aggregare i risultati finali

    Assunzioni correnti:
    - classificazione: ogni worker restituisce una matrice di voti/somme parziali
      di shape (n_samples, n_classes)
    - regressione: ogni worker restituisce la somma parziale delle predizioni
      di shape (n_samples, 1)
    """

    def __init__(
        self,
        leadership_guard,
        worker_registry: WorkerRegistryLike,
        worker_client,
        model_repository,
        max_parallel_requests: int | None = None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.worker_client = worker_client
        self.model_repository = model_repository
        self.max_parallel_requests = max_parallel_requests

    def run_inference(
        self,
        model_id: str,
        features: np.ndarray,
    ) -> InferenceResult:
        self.leadership_guard.require_leader()

        manifest = self.model_repository.load(model_id)
        if manifest is None:
            raise ValueError(f"Model '{model_id}' not found")

        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2D matrix")
        if X.shape[0] == 0:
            raise ValueError("Empty inference batch")

        alive_workers = self.worker_registry.alive_workers()
        if not alive_workers:
            raise RuntimeError("No alive workers available")

        tree_uris = [artifact.artifact_uri for artifact in manifest.tree_artifacts]
        if not tree_uris:
            raise RuntimeError("Model manifest contains no tree artifacts")

        worker_shards = self._assign_tree_uris_to_workers(
            workers=alive_workers,
            tree_uris=tree_uris,
        )
        if not worker_shards:
            raise RuntimeError("No worker shards could be built for inference")

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
                    manifest.model_id,
                    manifest.experiment_id,
                    manifest.model_type,
                    X,
                    uri_shard,
                    manifest.class_labels,
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
                            manifest.model_id,
                            manifest.experiment_id,
                            manifest.model_type,
                            X,
                            uri_shard,
                            manifest.class_labels,
                        )

                if not result.success:
                    raise RuntimeError(
                        f"Inference shard failed on worker {worker.worker_id}: "
                        f"{result.error_message}"
                    )

                responses.append(result)

        if manifest.model_type == "classification":
            aggregated_votes = np.zeros((X.shape[0], len(manifest.class_labels)), dtype=float)

            for response in responses:
                if response.values.shape != aggregated_votes.shape:
                    raise ValueError(
                        "Invalid classification shard response shape: "
                        f"expected {aggregated_votes.shape}, got {response.values.shape}"
                    )
                aggregated_votes += response.values

            predicted_indices = np.argmax(aggregated_votes, axis=1)
            predicted_labels = [manifest.class_labels[index] for index in predicted_indices]

            return InferenceResult(
                task_type="classification",
                predicted_labels=predicted_labels,
                predicted_values=None,
            )

        if manifest.model_type == "regression":
            aggregated_sum = np.zeros((X.shape[0], 1), dtype=float)

            for response in responses:
                if response.values.shape != aggregated_sum.shape:
                    raise ValueError(
                        "Invalid regression shard response shape: "
                        f"expected {aggregated_sum.shape}, got {response.values.shape}"
                    )
                aggregated_sum += response.values

            predicted_values = (aggregated_sum[:, 0] / len(manifest.tree_artifacts)).tolist()

            return InferenceResult(
                task_type="regression",
                predicted_labels=None,
                predicted_values=predicted_values,
            )

        raise ValueError(f"Unsupported model_type '{manifest.model_type}'")

    def _assign_tree_uris_to_workers(
        self,
        workers: list[WorkerLike],
        tree_uris: list[str],
    ) -> list[tuple[WorkerLike, list[str]]]:
        if not workers:
            return []
        if not tree_uris:
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