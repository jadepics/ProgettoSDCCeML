from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Protocol

from common.contracts import (
    ExperimentRecord,
    ForestConfiguration,
    ShardTrainingResult,
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

    Questa versione è adattata a un TaskLedger che salva più attempt
    per lo stesso task_id senza sovrascrivere lo storico.
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
            experiment.assigned_workers = list(
                dict.fromkeys(
                    list(experiment.assigned_workers)
                    + [shard.assigned_worker_id for shard in shards]
                )
            )

        self.job_repository.save_experiment(job_id, experiment)
        self.job_repository.update_job_status(
            job_id=job_id,
            status=JobStatus.RUNNING,
            message="Training in progress",
        )

        artifact_by_tree_id: dict[str, TreeArtifactMetadata] = {}

        max_workers = self.max_parallel_shards or len(shards)
        max_workers = min(max_workers, len(shards))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}

            for shard in shards:
                worker = self._find_worker_or_raise(alive_workers, shard.assigned_worker_id)
                future = pool.submit(
                    self._execute_shard_with_retry,
                    worker,
                    shard,
                )
                future_map[future] = shard

            for future in as_completed(future_map):
                original_shard = future_map[future]
                effective_shard, final_result, observed_results = future.result()

                for result in observed_results:
                    self._register_result_artifacts(
                        artifact_by_tree_id=artifact_by_tree_id,
                        result=result,
                    )

                if effective_shard.assigned_worker_id not in experiment.assigned_workers:
                    experiment.assigned_workers.append(effective_shard.assigned_worker_id)

                if not final_result.success:
                    self.task_ledger.mark_partial_failure(
                        task_id=effective_shard.task_id,
                        attempt_id=effective_shard.attempt_id,
                        job_id=effective_shard.job_id,
                        completed_tree_ids=list(final_result.completed_tree_ids),
                        failed_tree_ids=list(final_result.failed_tree_ids),
                        error_message=final_result.error_message or "Unknown training shard failure",
                    )
                    experiment.status = ExperimentStatus.FAILED
                    self.job_repository.save_experiment(job_id, experiment)
                    self.job_repository.update_job_status(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        message=(
                            f"Training shard failed permanently for task "
                            f"{effective_shard.task_id} attempt {effective_shard.attempt_id}: "
                            f"{final_result.error_message or 'unknown error'}"
                        ),
                    )
                    raise RuntimeError(
                        f"Training shard failed permanently for task "
                        f"{effective_shard.task_id} attempt {effective_shard.attempt_id}: "
                        f"{final_result.error_message or 'unknown error'}"
                    )

                completed_tree_ids = list(final_result.completed_tree_ids)
                if not completed_tree_ids:
                    completed_tree_ids = [artifact.tree_id for artifact in final_result.tree_artifacts]

                self.task_ledger.mark_completed(
                    task_id=effective_shard.task_id,
                    attempt_id=effective_shard.attempt_id,
                    job_id=effective_shard.job_id,
                    completed_tree_ids=completed_tree_ids,
                )

                completed_count = len(self.task_ledger.completed_tree_ids(job_id, experiment_id))
                experiment.completed_tree_count = completed_count
                self.job_repository.save_experiment(job_id, experiment)

                self.job_repository.update_job_status(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    message=f"Completed {completed_count}/{forest_config.n_estimators} trees",
                )

        collected_artifacts = sorted(
            artifact_by_tree_id.values(),
            key=lambda item: item.tree_index,
        )

        if len(collected_artifacts) != forest_config.n_estimators:
            raise RuntimeError(
                f"Expected {forest_config.n_estimators} trees, "
                f"got {len(collected_artifacts)}"
            )

        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_tree_count = len(collected_artifacts)
        self.job_repository.save_experiment(job_id, experiment)

        return collected_artifacts

    def _execute_shard_with_retry(
        self,
        worker: WorkerLike,
        shard: TrainingShard,
    ) -> tuple[TrainingShard, ShardTrainingResult, list[ShardTrainingResult]]:
        """
        Esegue un attempt iniziale e, se fallisce, prova un solo retry
        su un worker diverso.

        Ritorna:
        - effective_shard: lo shard dell'attempt finale
        - final_result: il risultato finale da considerare per successo/fallimento
        - observed_results: tutti i risultati osservati, utile per preservare
          eventuali artifact prodotti prima di un retry
        """
        observed_results: list[ShardTrainingResult] = []

        initial_result = self._dispatch_attempt(worker, shard)
        observed_results.append(initial_result)

        if initial_result.success:
            return shard, initial_result, observed_results

        self.task_ledger.mark_partial_failure(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            job_id=shard.job_id,
            completed_tree_ids=list(initial_result.completed_tree_ids),
            failed_tree_ids=list(initial_result.failed_tree_ids),
            error_message=initial_result.error_message or "Unknown training shard failure",
        )


        retry_worker = self.worker_registry.get_retry_candidate(
            exclude_worker_id=worker.worker_id
        )
        if retry_worker is None:
            return shard, initial_result, observed_results

        retry_shard = self._build_retry_shard(
            shard=shard,
            retry_worker=retry_worker,
        )

        retry_result = self._dispatch_attempt(retry_worker, retry_shard)
        observed_results.append(retry_result)

        if not retry_result.success:
            self.task_ledger.mark_partial_failure(
                task_id=retry_shard.task_id,
                attempt_id=retry_shard.attempt_id,
                job_id=retry_shard.job_id,
                completed_tree_ids=list(retry_result.completed_tree_ids),
                failed_tree_ids=list(retry_result.failed_tree_ids),
                error_message=retry_result.error_message or "Unknown retry shard failure",
            )
        return retry_shard, retry_result, observed_results

    def _dispatch_attempt(
        self,
        worker: WorkerLike,
        shard: TrainingShard,
    ) -> ShardTrainingResult:
        self.task_ledger.save(
            self._build_task_record(shard=shard, status=TaskStatus.PENDING)
        )
        self.task_ledger.mark_running(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            job_id=shard.job_id,
        )

        try:
            result = self.worker_client.train_shard(
                worker.host,
                worker.port,
                shard,
            )
        except Exception as exc:
            result = self._build_failed_result(
                shard=shard,
                worker_id=worker.worker_id,
                error_message=str(exc),
            )

        return self._normalize_result_identity(
            shard=shard,
            fallback_worker_id=worker.worker_id,
            result=result,
        )

    def _build_retry_shard(
        self,
        shard: TrainingShard,
        retry_worker: WorkerLike,
    ) -> TrainingShard:
        return TrainingShard(
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

    def _build_failed_result(
        self,
        shard: TrainingShard,
        worker_id: str,
        error_message: str,
    ) -> ShardTrainingResult:
        return ShardTrainingResult(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=worker_id,
            success=False,
            tree_artifacts=[],
            completed_tree_ids=[],
            failed_tree_ids=self._tree_ids_for_shard(shard),
            completed_tree_count=0,
            failed_tree_count=shard.tree_count,
            error_message=error_message,
            elapsed_time_seconds=0.0,
        )

    def _normalize_result_identity(
        self,
        shard: TrainingShard,
        fallback_worker_id: str,
        result: ShardTrainingResult,
    ) -> ShardTrainingResult:
        task_id = result.task_id or shard.task_id
        attempt_id = result.attempt_id or shard.attempt_id
        worker_id = result.worker_id or fallback_worker_id

        if task_id != shard.task_id:
            raise RuntimeError(
                f"Worker returned mismatched task_id: expected {shard.task_id}, got {task_id}"
            )

        if attempt_id != shard.attempt_id:
            raise RuntimeError(
                f"Worker returned mismatched attempt_id: expected {shard.attempt_id}, got {attempt_id}"
            )

        return ShardTrainingResult(
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            success=result.success,
            tree_artifacts=list(result.tree_artifacts),
            completed_tree_ids=list(result.completed_tree_ids),
            failed_tree_ids=list(result.failed_tree_ids),
            completed_tree_count=result.completed_tree_count,
            failed_tree_count=result.failed_tree_count,
            error_message=result.error_message,
            elapsed_time_seconds=result.elapsed_time_seconds,
        )

    def _register_result_artifacts(
        self,
        artifact_by_tree_id: dict[str, TreeArtifactMetadata],
        result: ShardTrainingResult,
    ) -> None:
        for artifact in result.tree_artifacts:
            artifact_by_tree_id[artifact.tree_id] = artifact

    def _tree_ids_for_shard(self, shard: TrainingShard) -> list[str]:
        return [
            generate_tree_id(shard.experiment_id, index)
            for index in range(
                shard.tree_start_index,
                shard.tree_start_index + shard.tree_count,
            )
        ]

    def _find_worker_or_raise(
        self,
        workers: list[WorkerLike],
        worker_id: str,
    ) -> WorkerLike:
        for worker in workers:
            if worker.worker_id == worker_id:
                return worker
        raise RuntimeError(f"Assigned worker '{worker_id}' is not available")