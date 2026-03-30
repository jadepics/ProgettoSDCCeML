from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, Sequence

from common.contracts import ForestConfiguration, PreparedDataset, TrainingShard
from common.ids import generate_task_id
from common.storage_layout import StorageLayout


class WorkerLike(Protocol):
    worker_id: str


class ShardPlanner:
    """
    Responsabilità:
    - dividere una foresta candidata in shard assegnabili
    - scegliere una distribuzione deterministica sugli worker vivi
    - costruire TrainingShard coerenti con il layout condiviso

    Nota:
    in questa prima versione il planner:
    - ordina i worker per worker_id
    - divide gli alberi quasi uniformemente
    - assegna un solo shard per worker quando possibile
    """

    def __init__(
        self,
        storage_layout: StorageLayout,
        lease_timeout_seconds: float = 600.0,
        initial_attempt_id: int = 1,
    ) -> None:
        self.storage_layout = storage_layout
        self.lease_timeout_seconds = lease_timeout_seconds
        self.initial_attempt_id = initial_attempt_id

    def plan(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        prepared_dataset: PreparedDataset,
        workers: Sequence[WorkerLike],
    ) -> list[TrainingShard]:
        if forest_config.n_estimators <= 0:
            raise ValueError("forest_config.n_estimators must be > 0")

        ordered_workers = self._normalize_workers(workers)
        if not ordered_workers:
            raise ValueError("At least one worker is required to plan training shards")

        shard_specs = self._split_trees(
            n_trees=forest_config.n_estimators,
            n_workers=len(ordered_workers),
        )

        artifact_output_dir = str(
            self.storage_layout.experiment_dir(job_id, experiment_id)
        )

        lease_expires_at_ts = time.time() + self.lease_timeout_seconds
        shards: list[TrainingShard] = []

        for worker, (tree_start_index, tree_count) in zip(ordered_workers, shard_specs):
            task_id = generate_task_id(
                experiment_id=experiment_id,
                tree_start_index=tree_start_index,
                tree_count=tree_count,
            )

            shard = TrainingShard(
                task_id=task_id,
                attempt_id=self.initial_attempt_id,
                job_id=job_id,
                experiment_id=experiment_id,
                assigned_worker_id=worker.worker_id,
                tree_start_index=tree_start_index,
                tree_count=tree_count,
                forest_config=forest_config,
                train_features_uri=prepared_dataset.train_features_uri,
                train_labels_uri=prepared_dataset.train_labels_uri,
                artifact_output_dir=artifact_output_dir,
                seed_base=forest_config.global_random_seed,
                lease_expires_at_ts=lease_expires_at_ts,
            )
            shards.append(shard)

        return shards

    def replan_missing_ranges(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        prepared_dataset: PreparedDataset,
        workers: Sequence[WorkerLike],
        missing_ranges: Sequence[tuple[int, int]],
        attempt_id: int,
    ) -> list[TrainingShard]:
        """
        Versione utile per la fase successiva di recovery.
        missing_ranges contiene tuple (tree_start_index, tree_count).
        """
        ordered_workers = self._normalize_workers(workers)
        if not ordered_workers:
            raise ValueError("At least one worker is required to replan training shards")

        if attempt_id <= 0:
            raise ValueError("attempt_id must be > 0")

        artifact_output_dir = str(
            self.storage_layout.experiment_dir(job_id, experiment_id)
        )

        lease_expires_at_ts = time.time() + self.lease_timeout_seconds
        shards: list[TrainingShard] = []

        for index, (tree_start_index, tree_count) in enumerate(missing_ranges):
            if tree_count <= 0:
                continue

            worker = ordered_workers[index % len(ordered_workers)]
            task_id = generate_task_id(
                experiment_id=experiment_id,
                tree_start_index=tree_start_index,
                tree_count=tree_count,
            )

            shard = TrainingShard(
                task_id=task_id,
                attempt_id=attempt_id,
                job_id=job_id,
                experiment_id=experiment_id,
                assigned_worker_id=worker.worker_id,
                tree_start_index=tree_start_index,
                tree_count=tree_count,
                forest_config=forest_config,
                train_features_uri=prepared_dataset.train_features_uri,
                train_labels_uri=prepared_dataset.train_labels_uri,
                artifact_output_dir=artifact_output_dir,
                seed_base=forest_config.global_random_seed,
                lease_expires_at_ts=lease_expires_at_ts,
            )
            shards.append(shard)

        return shards

    def _normalize_workers(self, workers: Sequence[WorkerLike]) -> list[WorkerLike]:
        unique_by_id: dict[str, WorkerLike] = {}
        for worker in workers:
            unique_by_id[worker.worker_id] = worker
        return [unique_by_id[worker_id] for worker_id in sorted(unique_by_id.keys())]

    def _split_trees(self, n_trees: int, n_workers: int) -> list[tuple[int, int]]:
        if n_trees <= 0:
            raise ValueError("n_trees must be > 0")
        if n_workers <= 0:
            raise ValueError("n_workers must be > 0")

        chunk_count = min(n_trees, n_workers)
        base = n_trees // chunk_count
        remainder = n_trees % chunk_count

        specs: list[tuple[int, int]] = []
        start = 0

        for index in range(chunk_count):
            tree_count = base + (1 if index < remainder else 0)
            specs.append((start, tree_count))
            start += tree_count

        return specs