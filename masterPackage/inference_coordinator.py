from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from common.prediction_io import (
    load_prediction_array,
    path_to_file_uri,
    read_parquet_row_count,
    save_final_prediction_array,
)

@dataclass(slots=True)
class InferenceResult:
    task_type: str
    prediction_uri: str
    n_rows: int
    n_cols: int
    metadata_uri: str | None = None


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


class InferenceCoordinator:
    """
    Inferenza distribuita tree-parallel.

   flusso per inferenza scalabile:
    - il client/master non invia DenseMatrix dentro gRPC;
    - SubmitInference riceve features_uri;
    - i worker leggono le feature da storage condiviso;
    - ogni worker valuta un sottoinsieme di alberi sull'intero batch;
    - ogni worker salva predizioni parziali su EFS;
    - il master legge prediction_uri, aggrega e salva le predizioni finali.

    gRPC viene usato come control plane.
    EFS/storage condiviso viene usato come data plane.
    """

    def __init__(
        self,
        leadership_guard,
        worker_registry: WorkerRegistryLike,
        worker_client,
        model_repository,
        storage_layout,
        max_parallel_requests: int | None = None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.worker_client = worker_client
        self.model_repository = model_repository
        self.storage_layout = storage_layout
        self.max_parallel_requests = max_parallel_requests

    def run_inference(
        self,
        model_id: str,
        features_uri: str,
    ) -> InferenceResult:
        coordinator_start = time.perf_counter()
        self.leadership_guard.require_leader()

        if not features_uri:
            raise ValueError("features_uri must be non-empty")

        manifest = self.model_repository.load(model_id)
        if manifest is None:
            raise ValueError(f"Model '{model_id}' not found")

        n_samples = read_parquet_row_count(features_uri)
        if n_samples <= 0:
            raise ValueError("Empty inference batch")

        alive_workers = self.worker_registry.alive_workers()
        if not alive_workers:
            raise RuntimeError("No alive workers available")

        tree_uris = [artifact.artifact_uri for artifact in manifest.tree_artifacts]
        if not tree_uris:
            raise RuntimeError("Model manifest contains no tree artifacts")

        assignments = self._assign_tree_uris_to_workers(
            workers=alive_workers,
            tree_uris=tree_uris,
        )
        if not assignments:
            raise RuntimeError("No worker shards could be built for inference")

        inference_id = self._new_inference_id()

        prediction_dir = self.storage_layout.inference_prediction_dir(
            model_id=manifest.model_id,
            inference_id=inference_id,
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_output_dir = path_to_file_uri(prediction_dir)

        responses = self._collect_prediction_responses(
            model_id=manifest.model_id,
            experiment_id=manifest.experiment_id,
            task_type=manifest.model_type,
            features_uri=features_uri,
            prediction_output_dir=prediction_output_dir,
            class_labels=manifest.class_labels,
            assignments=assignments,
        )

        if manifest.model_type == "classification":
            final_predictions = self._aggregate_classification_predictions(
                responses=responses,
                n_samples=n_samples,
                class_labels=manifest.class_labels,
            )

        elif manifest.model_type == "regression":
            final_predictions = self._aggregate_regression_predictions(
                responses=responses,
                n_samples=n_samples,
                tree_count=len(manifest.tree_artifacts),
            )

        else:
            raise ValueError(f"Unsupported model_type '{manifest.model_type}'")

        final_prediction_path = self.storage_layout.inference_final_prediction_path(
            model_id=manifest.model_id,
            inference_id=inference_id,
        )

        final_prediction_uri = save_final_prediction_array(
            array=final_predictions,
            output_uri_or_path=final_prediction_path,
        )

        metadata_uri = self._write_inference_metadata(
            model_id=manifest.model_id,
            experiment_id=manifest.experiment_id,
            inference_id=inference_id,
            task_type=manifest.model_type,
            features_uri=features_uri,
            prediction_uri=final_prediction_uri,
            n_rows=int(final_predictions.shape[0]),
            n_cols=int(final_predictions.shape[1]),
            tree_count=len(manifest.tree_artifacts),
            assignments=assignments,
            partial_prediction_count=len(responses),
            elapsed_time_seconds=time.perf_counter() - coordinator_start,
        )

        return InferenceResult(
            task_type=manifest.model_type,
            prediction_uri=final_prediction_uri,
            n_rows=int(final_predictions.shape[0]),
            n_cols=int(final_predictions.shape[1]),
            metadata_uri=metadata_uri,
        )

    def _write_inference_metadata(
        self,
        model_id: str,
        experiment_id: str,
        inference_id: str,
        task_type: str,
        features_uri: str,
        prediction_uri: str,
        n_rows: int,
        n_cols: int,
        tree_count: int,
        assignments: list[tuple[WorkerLike, list[str]]],
        partial_prediction_count: int,
        elapsed_time_seconds: float,
    ) -> str:
        metadata_path = self.storage_layout.inference_metadata_path(
            model_id=model_id,
            inference_id=inference_id,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "inference_id": inference_id,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "task_type": task_type,
            "features_uri": features_uri,
            "prediction_uri": prediction_uri,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "tree_count": int(tree_count),
            "worker_count": int(len(assignments)),
            "partial_prediction_count": int(partial_prediction_count),
            "coordinator_elapsed_time_seconds": float(elapsed_time_seconds),
            "assignments": [
                {
                    "worker_id": worker.worker_id,
                    "host": worker.host,
                    "port": int(worker.port),
                    "tree_count": len(tree_uris),
                }
                for worker, tree_uris in assignments
            ],
            "created_at": time.time(),
        }

        metadata_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        return metadata_path.resolve().as_uri()

    def _collect_prediction_responses(
        self,
        model_id: str,
        experiment_id: str,
        task_type: str,
        features_uri: str,
        prediction_output_dir: str,
        class_labels: list[str],
        assignments: list[tuple[WorkerLike, list[str]]],
    ) -> list:
        responses = []
        max_workers = self.max_parallel_requests or len(assignments)
        max_workers = min(max_workers, len(assignments))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for index, (worker, uri_shard) in enumerate(assignments):
                prediction_id = (
                    f"{experiment_id}_inference_{int(time.time())}_{index:06d}"
                )

                future = pool.submit(
                    self.worker_client.predict_shard_from_uri,
                    worker.host,
                    worker.port,
                    model_id,
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
                            model_id,
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
                        f"Inference shard failed on worker {worker.worker_id}: "
                        f"{result.error_message}"
                    )

                responses.append(result)

        return responses

    def _aggregate_classification_predictions(
        self,
        responses: list,
        n_samples: int,
        class_labels: list[str],
    ) -> np.ndarray:
        if not class_labels:
            raise ValueError("class_labels are required for classification inference")

        n_classes = len(class_labels)
        aggregated_votes = np.zeros((n_samples, n_classes), dtype=float)

        for response in responses:
            values = self._response_values(response)

            if values.shape == aggregated_votes.shape:
                aggregated_votes += values
                continue

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
        predicted_labels = np.asarray(
            [class_labels[index] for index in final_indices],
            dtype=str,
        )

        return predicted_labels.reshape(-1, 1)

    def _aggregate_regression_predictions(
        self,
        responses: list,
        n_samples: int,
        tree_count: int,
    ) -> np.ndarray:
        if tree_count <= 0:
            raise ValueError("tree_count must be > 0")

        aggregated_sum = np.zeros((n_samples, 1), dtype=float)

        for response in responses:
            values = self._response_values(response)

            if values.shape != aggregated_sum.shape:
                raise ValueError(
                    "Invalid regression shard response shape: "
                    f"expected {aggregated_sum.shape}, got {values.shape}"
                )

            aggregated_sum += values

        return aggregated_sum / float(tree_count)

    def _response_values(self, response) -> np.ndarray:
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

        prediction_uri = getattr(response, "prediction_uri", None)

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

    def _new_inference_id(self) -> str:
        return f"inference_{uuid.uuid4().hex}"