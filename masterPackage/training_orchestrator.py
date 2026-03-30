from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Protocol

from common.contracts import (
    ExperimentRecord,
    ForestConfiguration,
    TaskRecord,
    TrainingShard,
    TreeArtifactMetadata,
)
from common.enums import ExperimentStatus, JobStatus, TaskStatus
from common.ids import generate_tree_id


class WorkerLike(Protocol):
    worker_id: str
    host: str
    port: int


class WorkerRegistryLike(Protocol):
    def alive_workers(self) -> list[WorkerLike]:
        ...

    def get_retry_candidate(self, exclude_worker_id: str | None = None) -> Optional[WorkerLike]:
        ...


class TrainingOrchestrator:
    """
    Responsabilità:
    - orchestrare il training distribuito di un singolo esperimento
    - pianificare gli shard tramite ShardPlanner
    - inviare gli shard ai worker tramite WorkerClient
    - aggiornare TaskLedger e JobRepository
    - gestire retry semplici lato master

    Nota:
    questa versione è leader-only e copre la fase attuale del progetto.
    RecoveryPlanner e lease manager verranno integrati più avanti.
    """

    def __init__(
        self,
        leadership_guard,
        worker_registry: WorkerRegistryLike,
        task_ledger,
        job_repository,
        shard_planner,
        worker_client,
        lease_timeout_seconds: float = 600.0,
        max_parallel_shards: int | None = None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.task_ledger = task_ledger
        self.job_repository = job_repository
        self.shard_planner = shard_planner
        self.worker_client = worker_client
        self.lease_timeout_seconds = lease_timeout_seconds
        self.max_parallel_shards = max_parallel_shards

    def run_experiment(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> list[TreeArtifactMetadata]:
        self.leadership_guard.require_leader()

        job_record = self.job_repository.load(job_id)
        if job_record is None:
            raise ValueError(f"Job '{job_id}' not found")

        if job_record.prepared_dataset is None:
            raise ValueError(f"Job '{job_id}' has no prepared dataset")

        prepared_dataset = job_record.prepared_dataset
        alive_workers = self.worker_registry.alive_workers()
        if not alive_workers:
            raise RuntimeError("No alive workers available during scheduling")

        shards = self.shard_planner.plan(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            prepared_dataset=prepared_dataset,
            workers=alive_workers,
        )
        if not shards:
            raise RuntimeError("ShardPlanner produced no training shards")

        experiment = self.job_repository.load_experiment(job_id, experiment_id)
        if experiment is None:
            experiment = ExperimentRecord(
                experiment_id=experiment_id,
                forest_config=forest_config,
                status=ExperimentStatus.RUNNING,
                assigned_workers=[shard.assigned_worker_id for shard in shards],
                expected_tree_count=forest_config.n_estimators,
                completed_tree_count=0,
                validation_metrics=None,
            )
        else:
            experiment.status = ExperimentStatus.RUNNING
            experiment.assigned_workers = [shard.assigned_worker_id for shard in shards]

        self.job_repository.save_experiment(job_id, experiment)
        self.job_repository.update_job_status(
            job_id=job_id,
            status=JobStatus.RUNNING,
            message="Training in progress",
        )

        collected_artifacts: list[TreeArtifactMetadata] = []
        max_workers = self.max_parallel_shards or len(shards)
        max_workers = min(max_workers, len(shards))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for shard in shards:
                worker = self._find_worker_or_raise(alive_workers, shard.assigned_worker_id)

                self.task_ledger.save(self._build_task_record(shard, status=TaskStatus.PENDING))
                self.task_ledger.mark_running(shard.task_id)

                future = pool.submit(
                    self.worker_client.train_shard,
                    worker.host,
                    worker.port,
                    shard,
                )
                future_map[future] = (worker, shard)

            for future in as_completed(future_map):
                worker, shard = future_map[future]
                result = future.result()
                effective_shard = shard

                if not result.success:
                    effective_shard, result = self._retry_once(
                        failed_worker=worker,
                        shard=shard,
                    )

                if not result.success:
                    self.task_ledger.mark_failed(
                        effective_shard.task_id,
                        result.error_message or "Unknown training shard failure",
                    )
                    raise RuntimeError(
                        f"Training shard failed permanently for task "
                        f"{effective_shard.task_id}: {result.error_message}"
                    )

                completed_tree_ids = list(result.completed_tree_ids)
                if not completed_tree_ids:
                    completed_tree_ids = [artifact.tree_id for artifact in result.tree_artifacts]

                self.task_ledger.mark_completed(
                    effective_shard.task_id,
                    completed_tree_ids=completed_tree_ids,
                )

                collected_artifacts.extend(result.tree_artifacts)

                completed_count = len(
                    self.task_ledger.completed_tree_ids(job_id, experiment_id)
                )
                experiment.completed_tree_count = completed_count
                self.job_repository.save_experiment(job_id, experiment)

                self.job_repository.update_job_status(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    message=f"Completed {completed_count}/{forest_config.n_estimators} trees",
                )

        collected_artifacts.sort(key=lambda item: item.tree_index)

        if len(collected_artifacts) != forest_config.n_estimators:
            raise RuntimeError(
                f"Expected {forest_config.n_estimators} trees, "
                f"got {len(collected_artifacts)}"
            )

        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_tree_count = len(collected_artifacts)
        self.job_repository.save_experiment(job_id, experiment)

        return collected_artifacts

    def _retry_once(
        self,
        failed_worker: WorkerLike,
        shard: TrainingShard,
    ) -> tuple[TrainingShard, object]:
        retry_worker = self.worker_registry.get_retry_candidate(
            exclude_worker_id=failed_worker.worker_id
        )
        if retry_worker is None:
            raise RuntimeError(
                f"Training shard failed on worker {failed_worker.worker_id} "
                f"and no retry worker is available"
            )

        retry_shard = TrainingShard(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id + 1,
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            assigned_worker_id=retry_worker.worker_id,
            tree_start_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            forest_config=shard.forest_config,
            train_features_uri=shard.train_features_uri,
            train_labels_uri=shard.train_labels_uri,
            artifact_output_dir=shard.artifact_output_dir,
            seed_base=shard.seed_base,
            lease_expires_at_ts=time.time() + self.lease_timeout_seconds,
        )

        self.task_ledger.save(self._build_task_record(retry_shard, status=TaskStatus.PENDING))
        self.task_ledger.mark_running(retry_shard.task_id)

        result = self.worker_client.train_shard(
            retry_worker.host,
            retry_worker.port,
            retry_shard,
        )
        return retry_shard, result

    def _build_task_record(
        self,
        shard: TrainingShard,
        status: TaskStatus,
    ) -> TaskRecord:
        return TaskRecord(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            worker_id=shard.assigned_worker_id,
            status=status,
            tree_ids=[
                generate_tree_id(shard.experiment_id, index)
                for index in range(
                    shard.tree_start_index,
                    shard.tree_start_index + shard.tree_count,
                )
            ],
            completed_tree_ids=[],
            failed_tree_ids=[],
            lease_expires_at_ts=shard.lease_expires_at_ts,
            updated_at=time.time(),
            error_message=None,
        )

    def _find_worker_or_raise(
        self,
        workers: list[WorkerLike],
        worker_id: str,
    ) -> WorkerLike:
        for worker in workers:
            if worker.worker_id == worker_id:
                return worker
        raise RuntimeError(f"Assigned worker '{worker_id}' is not available")