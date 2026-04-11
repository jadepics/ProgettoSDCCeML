from __future__ import annotations

import time
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

    Evoluzione restart-safe:
    - plan(...) può pianificare l'intera foresta
    - oppure solo i tree_id mancanti, comprimendoli in range contigui

    Nota:
    - gli worker sono ordinati per worker_id
    - gli shard sono deterministici
    - i task_id restano deterministici perché dipendono da experiment_id,
      tree_start_index e tree_count
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
        missing_tree_ids: Sequence[str] | None = None,
        attempt_id: int | None = None,
    ) -> list[TrainingShard]:
        """
        Se missing_tree_ids è None:
        - pianifica tutta la foresta

        Se missing_tree_ids è valorizzato:
        - pianifica solo gli alberi mancanti
        - comprime i tree_id in range contigui
        """
        if forest_config.n_estimators <= 0:
            raise ValueError("forest_config.n_estimators must be > 0")

        ordered_workers = self._normalize_workers(workers)
        if not ordered_workers:
            raise ValueError("At least one worker is required to plan training shards")

        effective_attempt_id = self.initial_attempt_id if attempt_id is None else attempt_id
        if effective_attempt_id <= 0:
            raise ValueError("attempt_id must be > 0")

        if missing_tree_ids is None:
            shard_specs = self._split_trees(
                n_trees=forest_config.n_estimators,
                n_workers=len(ordered_workers),
            )
        else:
            missing_indices = self._normalize_missing_tree_ids(
                experiment_id=experiment_id,
                missing_tree_ids=missing_tree_ids,
                n_estimators=forest_config.n_estimators,
            )
            if not missing_indices:
                return []

            shard_specs = self._compress_tree_indices_to_ranges(missing_indices)

        return self._build_shards(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            prepared_dataset=prepared_dataset,
            ordered_workers=ordered_workers,
            shard_specs=shard_specs,
            attempt_id=effective_attempt_id,
        )

    def plan_missing_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        prepared_dataset: PreparedDataset,
        workers: Sequence[WorkerLike],
        missing_tree_ids: Sequence[str],
        attempt_id: int,
    ) -> list[TrainingShard]:
        """
        Wrapper semantico comodo per il recovery/restart-safe path.
        """
        return self.plan(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            prepared_dataset=prepared_dataset,
            workers=workers,
            missing_tree_ids=missing_tree_ids,
            attempt_id=attempt_id,
        )

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
        Manteniamo anche questa versione, utile se in futuro il RecoveryPlanner
        produce già direttamente range (tree_start_index, tree_count).
        """
        ordered_workers = self._normalize_workers(workers)
        if not ordered_workers:
            raise ValueError("At least one worker is required to replan training shards")

        if attempt_id <= 0:
            raise ValueError("attempt_id must be > 0")

        normalized_ranges = [
            (tree_start_index, tree_count)
            for tree_start_index, tree_count in missing_ranges
            if tree_count > 0
        ]
        if not normalized_ranges:
            return []

        return self._build_shards(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            prepared_dataset=prepared_dataset,
            ordered_workers=ordered_workers,
            shard_specs=normalized_ranges,
            attempt_id=attempt_id,
        )

    def _build_shards(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        prepared_dataset: PreparedDataset,
        ordered_workers: Sequence[WorkerLike],
        shard_specs: Sequence[tuple[int, int]],
        attempt_id: int,
    ) -> list[TrainingShard]:
        artifact_output_dir = str(
            self.storage_layout.experiment_dir(job_id, experiment_id)
        )

        lease_expires_at_ts = time.time() + self.lease_timeout_seconds
        shards: list[TrainingShard] = []

        for index, (tree_start_index, tree_count) in enumerate(shard_specs):
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

    def _normalize_missing_tree_ids(
        self,
        experiment_id: str,
        missing_tree_ids: Sequence[str],
        n_estimators: int,
    ) -> list[int]:
        """
        Converte i tree_id mancanti in indici interi ordinati, unici e validati.
        """
        normalized_indices: set[int] = set()

        for tree_id in missing_tree_ids:
            tree_index = self._tree_index_from_tree_id(
                experiment_id=experiment_id,
                tree_id=tree_id,
            )

            if tree_index < 0 or tree_index >= n_estimators:
                raise ValueError(
                    f"Tree index {tree_index} out of bounds for "
                    f"{n_estimators} estimators"
                )

            normalized_indices.add(tree_index)

        return sorted(normalized_indices)

    def _compress_tree_indices_to_ranges(
        self,
        tree_indices: Sequence[int],
    ) -> list[tuple[int, int]]:
        """
        Esempio:
        [0, 1, 2, 5, 6, 9] -> [(0, 3), (5, 2), (9, 1)]
        """
        if not tree_indices:
            return []

        ordered = sorted(tree_indices)
        ranges: list[tuple[int, int]] = []

        range_start = ordered[0]
        previous = ordered[0]

        for current in ordered[1:]:
            if current == previous + 1:
                previous = current
                continue

            ranges.append((range_start, previous - range_start + 1))
            range_start = current
            previous = current

        ranges.append((range_start, previous - range_start + 1))
        return ranges

    def _tree_index_from_tree_id(
        self,
        experiment_id: str,
        tree_id: str,
    ) -> int:
        prefix = f"{experiment_id}_tree_"
        if not tree_id.startswith(prefix):
            raise ValueError(
                f"Tree id '{tree_id}' does not match experiment '{experiment_id}'"
            )

        raw_index = tree_id[len(prefix):]
        return int(raw_index)