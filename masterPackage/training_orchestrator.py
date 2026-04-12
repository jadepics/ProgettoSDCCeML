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
from common.enums import ExperimentStatus, JobStatus, TaskStatus, TreeStatus
from common.ids import generate_tree_id, tree_seed
from masterPackage.retry_policy import RetryPolicy
from masterPackage.task_lease_manager import TaskLeaseManager


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
            retry_policy: RetryPolicy | None = None,
            task_lease_manager: TaskLeaseManager | None = None,
            worker_heartbeat_monitor=None,
            recovery_planner=None,
    ) -> None:
        self.leadership_guard = leadership_guard
        self.worker_registry = worker_registry
        self.task_ledger = task_ledger
        self.job_repository = job_repository
        self.shard_planner = shard_planner
        self.worker_client = worker_client
        self.lease_timeout_seconds = lease_timeout_seconds
        self.max_parallel_shards = max_parallel_shards
        self.retry_policy = retry_policy or RetryPolicy()
        self.task_lease_manager = task_lease_manager or TaskLeaseManager(
            task_ledger=task_ledger,
            lease_timeout_seconds=lease_timeout_seconds,
        )
        self.worker_heartbeat_monitor = worker_heartbeat_monitor
        self.recovery_planner = recovery_planner

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

        experiment = self._load_or_initialize_experiment(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        completed_tree_ids = set(self._completed_tree_ids(job_id, experiment_id))
        missing_tree_ids = self._missing_tree_ids(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        artifact_by_tree_id = self._collect_persisted_completed_artifacts(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        experiment.completed_tree_count = len(completed_tree_ids)
        self.job_repository.save_experiment(job_id, experiment)
        self.job_repository.update_job_status(
            job_id=job_id,
            status=JobStatus.RUNNING,
            message=(
                f"Recovered {len(completed_tree_ids)}/{forest_config.n_estimators} "
                f"completed trees from persisted state"
            ),
        )

        if not missing_tree_ids:
            final_artifacts = self._collect_final_artifacts(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
                artifact_by_tree_id=artifact_by_tree_id,
            )
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_tree_count = len(final_artifacts)
            self.job_repository.save_experiment(job_id, experiment)
            return final_artifacts

        alive_workers = self._alive_workers_for_scheduling()
        if not alive_workers:
            stale_ids = self._stale_worker_ids()
            if stale_ids:
                raise RuntimeError(
                    f"No alive workers available during scheduling. "
                    f"Stale workers detected: {stale_ids}"
                )
            raise RuntimeError("No alive workers available during scheduling")
        if self.recovery_planner is not None:
            recovery_plan = self.recovery_planner.build_plan(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
                prepared_dataset=prepared_dataset,
                workers=alive_workers,
            )
            missing_tree_ids = recovery_plan.missing_tree_ids
            shards = [
                self._with_next_attempt_id(job_id, shard)
                for shard in recovery_plan.recovery_shards
            ]
        else:
            shards = self._plan_missing_shards(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
                prepared_dataset=prepared_dataset,
                workers=alive_workers,
                missing_tree_ids=missing_tree_ids,
            )
        if not shards:
            raise RuntimeError(
                "Some trees are still missing, but no shards could be planned "
                "for the missing tree set"
            )

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
            message=(
                f"Training missing trees: {len(missing_tree_ids)} remaining out of "
                f"{forest_config.n_estimators}"
            ),
        )

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
                    self.task_lease_manager.release(
                        job_id=effective_shard.job_id,
                        task_id=effective_shard.task_id,
                        attempt_id=effective_shard.attempt_id,
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

                completed_tree_ids_for_attempt = list(final_result.completed_tree_ids)
                if not completed_tree_ids_for_attempt:
                    completed_tree_ids_for_attempt = [
                        artifact.tree_id for artifact in final_result.tree_artifacts
                    ]

                self.task_ledger.mark_completed(
                    task_id=effective_shard.task_id,
                    attempt_id=effective_shard.attempt_id,
                    job_id=effective_shard.job_id,
                    completed_tree_ids=completed_tree_ids_for_attempt,
                )
                self.task_lease_manager.release(
                    job_id=effective_shard.job_id,
                    task_id=effective_shard.task_id,
                    attempt_id=effective_shard.attempt_id,
                )

                completed_count = len(self._completed_tree_ids(job_id, experiment_id))
                experiment.completed_tree_count = completed_count
                self.job_repository.save_experiment(job_id, experiment)

                self.job_repository.update_job_status(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    message=f"Completed {completed_count}/{forest_config.n_estimators} trees",
                )

        final_artifacts = self._collect_final_artifacts(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            artifact_by_tree_id=artifact_by_tree_id,
        )

        if len(final_artifacts) != forest_config.n_estimators:
            raise RuntimeError(
                f"Expected {forest_config.n_estimators} tree artifacts, "
                f"got {len(final_artifacts)}"
            )

        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_tree_count = len(final_artifacts)
        self.job_repository.save_experiment(job_id, experiment)

        return final_artifacts

    def _execute_shard_with_retry(
        self,
        worker: WorkerLike,
        shard: TrainingShard,
    ) -> tuple[TrainingShard, ShardTrainingResult, list[ShardTrainingResult]]:
        observed_results: list[ShardTrainingResult] = []

        current_worker = worker
        current_shard = shard

        while True:
            current_shard, result = self._dispatch_attempt(current_worker, current_shard)
            observed_results.append(result)

            if result.success:
                return current_shard, result, observed_results

            self.task_ledger.mark_partial_failure(
                task_id=current_shard.task_id,
                attempt_id=current_shard.attempt_id,
                job_id=current_shard.job_id,
                completed_tree_ids=list(result.completed_tree_ids),
                failed_tree_ids=list(result.failed_tree_ids),
                error_message=result.error_message or "Unknown training shard failure",
            )
            self.task_lease_manager.release(
                job_id=current_shard.job_id,
                task_id=current_shard.task_id,
                attempt_id=current_shard.attempt_id,
            )

            should_retry = self.retry_policy.should_retry(
                attempt_id=current_shard.attempt_id,
                error_message=result.error_message,
            )
            if not should_retry:
                return current_shard, result, observed_results

            retry_worker = self.worker_registry.get_retry_candidate(
                exclude_worker_id=current_worker.worker_id
            )
            if retry_worker is None:
                return current_shard, result, observed_results

            backoff_seconds = self.retry_policy.backoff_seconds_for(
                attempt_id=current_shard.attempt_id
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds)

            current_shard = self._build_retry_shard(
                shard=current_shard,
                retry_worker=retry_worker,
            )
            current_worker = retry_worker

    def _dispatch_attempt(
        self,
        worker: WorkerLike,
        shard: TrainingShard,
    ) -> tuple[TrainingShard, ShardTrainingResult]:
        leased_shard = self.task_lease_manager.acquire(shard)

        self.task_ledger.save(
            self._build_task_record(shard=leased_shard, status=TaskStatus.PENDING)
        )
        self.task_ledger.mark_running(
            task_id=leased_shard.task_id,
            attempt_id=leased_shard.attempt_id,
            job_id=leased_shard.job_id,
        )

        try:
            result = self.worker_client.train_shard(
                worker.host,
                worker.port,
                leased_shard,
            )
        except Exception as exc:
            result = self._build_failed_result(
                shard=leased_shard,
                worker_id=worker.worker_id,
                error_message=str(exc),
            )

        normalized = self._normalize_result_identity(
            shard=leased_shard,
            fallback_worker_id=worker.worker_id,
            result=result,
        )
        return leased_shard, normalized

    def _alive_workers_for_scheduling(self) -> list[WorkerLike]:
        if self.worker_heartbeat_monitor is None:
            return self.worker_registry.alive_workers()

        snapshots = self.worker_heartbeat_monitor.alive_workers()
        worker_by_id = {
            worker.worker_id: worker
            for worker in self.worker_registry.list_workers()
        }

        result: list[WorkerLike] = []
        for snapshot in snapshots:
            worker = worker_by_id.get(snapshot.worker_id)
            if worker is not None:
                result.append(worker)

        return result

    def _stale_worker_ids(self) -> list[str]:
        if self.worker_heartbeat_monitor is None:
            return []

        return self.worker_heartbeat_monitor.stale_worker_ids()