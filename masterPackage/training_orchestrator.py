from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
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
from common.ids import generate_tree_id
from masterPackage.Metrics.Scalability_Metrics_Collector import (
    ScalabilityMetricsCollector,
)
from masterPackage.retry_policy import RetryPolicy
from masterPackage.task_lease_manager import TaskLeaseManager


class WorkerLike(Protocol):
    worker_id: str
    host: str
    port: int


class WorkerRegistryLike(Protocol):
    def alive_workers(self) -> list[WorkerLike]:
        ...

    def list_workers(self) -> list[WorkerLike]:
        ...

    def get_retry_candidate(
        self,
        exclude_worker_id: str | None = None,
    ) -> Optional[WorkerLike]:
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
        if max_parallel_shards is not None and max_parallel_shards <= 0:
            raise ValueError("max_parallel_shards must be > 0 when provided")

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
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> list[TreeArtifactMetadata]:
        self.leadership_guard.require_leader()

        experiment_started_at = time.time()

        self._metrics_event(
            metrics_collector,
            "orchestrator_experiment_started",
            job_id=job_id,
            experiment_id=experiment_id,
            n_estimators=forest_config.n_estimators,
            max_parallel_shards=self.max_parallel_shards,
            lease_timeout_seconds=self.lease_timeout_seconds,
        )

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

        max_recovery_rounds = int(
            os.getenv("TRAINING_MAX_RECOVERY_ROUNDS", "20")
        )
        max_idle_rounds = int(
            os.getenv("TRAINING_MAX_IDLE_RECOVERY_ROUNDS", "3")
        )
        deferred_sleep_seconds = float(
            os.getenv("TRAINING_RECOVERY_DEFERRED_SLEEP_SECONDS", "5")
        )

        artifact_by_tree_id: dict[str, TreeArtifactMetadata] = {}
        idle_rounds = 0
        last_completed_count = -1

        for recovery_round in range(1, max_recovery_rounds + 1):
            round_started_at = time.time()

            self.leadership_guard.require_leader()

            self._metrics_event(
                metrics_collector,
                "recovery_round_started",
                job_id=job_id,
                experiment_id=experiment_id,
                recovery_round=recovery_round,
                max_recovery_rounds=max_recovery_rounds,
            )

            artifact_by_tree_id.update(
                self._collect_persisted_completed_artifacts(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                )
            )

            completed_tree_ids = set(
                self._durable_completed_tree_ids(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                )
            )

            missing_tree_ids = self._missing_tree_ids(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
            )

            completed_count = len(completed_tree_ids)

            self._metrics_event(
                metrics_collector,
                "recovery_state_observed",
                job_id=job_id,
                experiment_id=experiment_id,
                recovery_round=recovery_round,
                completed_tree_count=completed_count,
                missing_tree_count=len(missing_tree_ids),
                expected_tree_count=forest_config.n_estimators,
            )

            print(
                "[TrainingOrchestrator] resume/recovery state:",
                f"job_id={job_id}",
                f"experiment_id={experiment_id}",
                f"completed={completed_count}/{forest_config.n_estimators}",
                f"missing={len(missing_tree_ids)}",
                flush=True,
            )

            experiment.completed_tree_count = completed_count
            experiment.status = ExperimentStatus.RUNNING
            self.job_repository.save_experiment(job_id, experiment)

            self.job_repository.update_job_status(
                job_id=job_id,
                status=JobStatus.RUNNING,
                message=(
                    f"Recovery round {recovery_round}: "
                    f"{completed_count}/{forest_config.n_estimators} trees completed, "
                    f"{len(missing_tree_ids)} missing"
                ),
            )

            print(
                "[TrainingOrchestrator] recovery round:",
                f"round={recovery_round}",
                f"completed={completed_count}",
                f"missing={len(missing_tree_ids)}",
                flush=True,
            )

            if not missing_tree_ids:
                final_artifacts = self._collect_final_artifacts(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                    artifact_by_tree_id=artifact_by_tree_id,
                )

                if len(final_artifacts) != forest_config.n_estimators:
                    raise RuntimeError(
                        f"Experiment {experiment_id} has no missing trees, "
                        f"but final artifacts are incomplete: "
                        f"{len(final_artifacts)}/{forest_config.n_estimators}"
                    )

                experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_tree_count = len(final_artifacts)
                self.job_repository.save_experiment(job_id, experiment)

                self.job_repository.update_job_status(
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    message=(
                        f"Training experiment completed: "
                        f"{len(final_artifacts)}/{forest_config.n_estimators} trees"
                    ),
                )

                self._metrics_event(
                    metrics_collector,
                    "orchestrator_experiment_completed",
                    job_id=job_id,
                    experiment_id=experiment_id,
                    completed_tree_count=len(final_artifacts),
                    expected_tree_count=forest_config.n_estimators,
                    elapsed_time_seconds=time.time() - experiment_started_at,
                )

                return final_artifacts

            alive_workers = self._alive_workers_for_scheduling()
            self._log_alive_workers(alive_workers)

            self._metrics_event(
                metrics_collector,
                "alive_workers_observed",
                job_id=job_id,
                experiment_id=experiment_id,
                recovery_round=recovery_round,
                worker_count=len(alive_workers),
                workers=[
                    self._format_worker(worker)
                    for worker in alive_workers
                ],
            )

            if not alive_workers:
                stale_ids = self._stale_worker_ids()

                self._metrics_event(
                    metrics_collector,
                    "no_alive_workers",
                    job_id=job_id,
                    experiment_id=experiment_id,
                    recovery_round=recovery_round,
                    stale_worker_ids=stale_ids,
                )

                if stale_ids:
                    raise RuntimeError(
                        f"No alive workers available during scheduling. "
                        f"Stale workers detected: {stale_ids}"
                    )

                raise RuntimeError("No alive workers available during scheduling")

            attempts = self.task_ledger.list_attempts_by_experiment(
                job_id=job_id,
                experiment_id=experiment_id,
            )

            expected_tree_ids = self._expected_tree_ids(
                experiment_id=experiment_id,
                forest_config=forest_config,
            )

            is_initial_full_training_plan = self._is_initial_full_training_plan(
                attempts=attempts,
                missing_tree_ids=missing_tree_ids,
                expected_tree_ids=expected_tree_ids,
            )

            if self.recovery_planner is not None and not is_initial_full_training_plan:
                recovery_plan = self.recovery_planner.build_plan(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                    prepared_dataset=prepared_dataset,
                    workers=alive_workers,
                )

                force_after_rounds = int(
                    os.getenv(
                        "TRAINING_FORCE_RECLAIM_DEFERRED_AFTER_ROUNDS",
                        "3",
                    )
                )

                self._metrics_event(
                    metrics_collector,
                    "recovery_plan_created",
                    job_id=job_id,
                    experiment_id=experiment_id,
                    recovery_round=recovery_round,
                    recover_now_tree_count=len(recovery_plan.recover_now_tree_ids),
                    deferred_tree_count=len(recovery_plan.deferred_tree_ids),
                    recovery_shard_count=len(recovery_plan.recovery_shards),
                )

                if (
                    recovery_plan.deferred_tree_ids
                    and not recovery_plan.recover_now_tree_ids
                    and recovery_round < force_after_rounds
                ):
                    print(
                        "[TrainingOrchestrator] recovery deferred:",
                        f"deferred={len(recovery_plan.deferred_tree_ids)}",
                        f"round={recovery_round}/{force_after_rounds}",
                        f"sleep={deferred_sleep_seconds}",
                        flush=True,
                    )

                    self._metrics_event(
                        metrics_collector,
                        "recovery_deferred",
                        job_id=job_id,
                        experiment_id=experiment_id,
                        recovery_round=recovery_round,
                        deferred_tree_count=len(recovery_plan.deferred_tree_ids),
                        sleep_seconds=deferred_sleep_seconds,
                    )

                    time.sleep(deferred_sleep_seconds)
                    continue

                if (
                    recovery_plan.deferred_tree_ids
                    and not recovery_plan.recover_now_tree_ids
                    and recovery_round >= force_after_rounds
                ):
                    print(
                        "[TrainingOrchestrator] forcing recovery of missing trees "
                        "after repeated deferred rounds:",
                        f"deferred={len(recovery_plan.deferred_tree_ids)}",
                        f"missing={len(missing_tree_ids)}",
                        flush=True,
                    )

                    shards = self._plan_missing_shards(
                        job_id=job_id,
                        experiment_id=experiment_id,
                        forest_config=forest_config,
                        prepared_dataset=prepared_dataset,
                        workers=alive_workers,
                        missing_tree_ids=missing_tree_ids,
                    )

                    print(
                        "[TrainingOrchestrator] using forced missing shard planner:",
                        f"missing={len(missing_tree_ids)}",
                        f"planned_shards={len(shards)}",
                        flush=True,
                    )

                    self._metrics_event(
                        metrics_collector,
                        "forced_missing_shard_plan_created",
                        job_id=job_id,
                        experiment_id=experiment_id,
                        recovery_round=recovery_round,
                        missing_tree_count=len(missing_tree_ids),
                        planned_shard_count=len(shards),
                    )

                else:
                    missing_tree_ids = list(recovery_plan.recover_now_tree_ids)
                    shards = list(recovery_plan.recovery_shards)

                    print(
                        "[TrainingOrchestrator] using recovery planner:",
                        f"recover_now={len(recovery_plan.recover_now_tree_ids)}",
                        f"deferred={len(recovery_plan.deferred_tree_ids)}",
                        flush=True,
                    )

                    self._metrics_event(
                        metrics_collector,
                        "recovery_shards_selected",
                        job_id=job_id,
                        experiment_id=experiment_id,
                        recovery_round=recovery_round,
                        recover_now_tree_count=len(recovery_plan.recover_now_tree_ids),
                        deferred_tree_count=len(recovery_plan.deferred_tree_ids),
                        planned_shard_count=len(shards),
                    )

            else:
                shards = self._plan_missing_shards(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                    prepared_dataset=prepared_dataset,
                    workers=alive_workers,
                    missing_tree_ids=missing_tree_ids,
                )

                print(
                    "[TrainingOrchestrator] using initial/missing shard planner:",
                    f"is_initial_full_training_plan={is_initial_full_training_plan}",
                    flush=True,
                )

                self._metrics_event(
                    metrics_collector,
                    "initial_or_missing_shard_plan_created",
                    job_id=job_id,
                    experiment_id=experiment_id,
                    recovery_round=recovery_round,
                    is_initial_full_training_plan=is_initial_full_training_plan,
                    missing_tree_count=len(missing_tree_ids),
                    planned_shard_count=len(shards),
                )

            if not shards:
                idle_rounds += 1

                print(
                    "[TrainingOrchestrator] no shards planned:",
                    f"idle_rounds={idle_rounds}/{max_idle_rounds}",
                    flush=True,
                )

                self._metrics_event(
                    metrics_collector,
                    "no_shards_planned",
                    job_id=job_id,
                    experiment_id=experiment_id,
                    recovery_round=recovery_round,
                    idle_rounds=idle_rounds,
                    max_idle_rounds=max_idle_rounds,
                    missing_tree_count=len(missing_tree_ids),
                )

                if idle_rounds >= max_idle_rounds:
                    raise RuntimeError(
                        "Some trees are still missing, but no shards could be "
                        "planned for too many consecutive recovery rounds"
                    )

                time.sleep(deferred_sleep_seconds)
                continue

            self._log_planned_shards(shards)

            max_workers = self._effective_max_parallel_shards(
                shard_count=len(shards),
            )

            self._record_training_plan(
                metrics_collector=metrics_collector,
                worker_count=len(alive_workers),
                shard_count=len(shards),
                n_estimators_total=forest_config.n_estimators,
            )

            self._metrics_event(
                metrics_collector,
                "training_round_scheduled",
                job_id=job_id,
                experiment_id=experiment_id,
                recovery_round=recovery_round,
                worker_count=len(alive_workers),
                shard_count=len(shards),
                missing_tree_count=len(missing_tree_ids),
                expected_tree_count=forest_config.n_estimators,
                max_parallel_shards=max_workers,
                total_shard_tree_count=sum(shard.tree_count for shard in shards),
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
                    f"Training scheduled: {len(shards)} shards for "
                    f"{len(missing_tree_ids)}/{forest_config.n_estimators} "
                    f"missing trees; parallelism={max_workers}"
                ),
            )

            had_failure = False

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_map = {}

                for shard in shards:
                    worker = self._find_worker_or_raise(
                        workers=alive_workers,
                        worker_id=shard.assigned_worker_id,
                    )

                    print(
                        "[TrainingOrchestrator] dispatching shard:",
                        f"task_id={shard.task_id}",
                        f"attempt_id={shard.attempt_id}",
                        f"worker={worker.worker_id}@{worker.host}:{worker.port}",
                        f"tree_start={shard.tree_start_index}",
                        f"tree_count={shard.tree_count}",
                        flush=True,
                    )

                    self._metrics_event(
                        metrics_collector,
                        "shard_dispatched",
                        job_id=job_id,
                        experiment_id=experiment_id,
                        recovery_round=recovery_round,
                        task_id=shard.task_id,
                        attempt_id=shard.attempt_id,
                        worker_id=worker.worker_id,
                        worker_host=worker.host,
                        worker_port=worker.port,
                        tree_start_index=shard.tree_start_index,
                        tree_count=shard.tree_count,
                    )

                    future = pool.submit(
                        self._execute_shard_with_retry,
                        worker,
                        shard,
                        metrics_collector,
                    )
                    future_map[future] = shard

                for future in as_completed(future_map):
                    initial_shard = future_map[future]

                    try:
                        (
                            effective_shard,
                            final_result,
                            observed_results,
                        ) = future.result()

                    except Exception as exc:
                        had_failure = True

                        error_message = (
                            f"Unexpected executor failure for task "
                            f"{initial_shard.task_id} attempt "
                            f"{initial_shard.attempt_id}: {exc}"
                        )

                        self.task_ledger.mark_partial_failure(
                            task_id=initial_shard.task_id,
                            attempt_id=initial_shard.attempt_id,
                            job_id=initial_shard.job_id,
                            completed_tree_ids=[],
                            failed_tree_ids=self._tree_ids_for_shard(
                                initial_shard
                            ),
                            error_message=error_message,
                        )

                        self.task_lease_manager.release(
                            job_id=initial_shard.job_id,
                            task_id=initial_shard.task_id,
                            attempt_id=initial_shard.attempt_id,
                        )

                        self._metrics_event(
                            metrics_collector,
                            "shard_executor_failure",
                            job_id=job_id,
                            experiment_id=experiment_id,
                            task_id=initial_shard.task_id,
                            attempt_id=initial_shard.attempt_id,
                            worker_id=initial_shard.assigned_worker_id,
                            tree_count=initial_shard.tree_count,
                            error_message=error_message,
                        )

                        print(
                            "[TrainingOrchestrator] shard executor failure, "
                            "will retry through recovery loop:",
                            error_message,
                            flush=True,
                        )

                        continue

                    for result in observed_results:
                        self._register_result_artifacts(
                            artifact_by_tree_id=artifact_by_tree_id,
                            result=result,
                        )

                    if (
                        effective_shard.assigned_worker_id
                        not in experiment.assigned_workers
                    ):
                        experiment.assigned_workers.append(
                            effective_shard.assigned_worker_id
                        )

                    if not final_result.success:
                        had_failure = True

                        self.task_ledger.mark_partial_failure(
                            task_id=effective_shard.task_id,
                            attempt_id=effective_shard.attempt_id,
                            job_id=effective_shard.job_id,
                            completed_tree_ids=list(
                                final_result.completed_tree_ids
                            ),
                            failed_tree_ids=list(
                                final_result.failed_tree_ids
                            ),
                            error_message=(
                                final_result.error_message
                                or "Unknown training shard failure"
                            ),
                        )

                        self.task_lease_manager.release(
                            job_id=effective_shard.job_id,
                            task_id=effective_shard.task_id,
                            attempt_id=effective_shard.attempt_id,
                        )

                        self._record_shard_result(
                            metrics_collector=metrics_collector,
                            worker_id=final_result.worker_id,
                            task_id=effective_shard.task_id,
                            completed_tree_count=final_result.completed_tree_count,
                            failed_tree_count=final_result.failed_tree_count,
                            elapsed_time_seconds=final_result.elapsed_time_seconds,
                        )

                        self._metrics_event(
                            metrics_collector,
                            "shard_failed_permanently",
                            job_id=job_id,
                            experiment_id=experiment_id,
                            task_id=effective_shard.task_id,
                            attempt_id=effective_shard.attempt_id,
                            worker_id=effective_shard.assigned_worker_id,
                            completed_tree_count=final_result.completed_tree_count,
                            failed_tree_count=final_result.failed_tree_count,
                            error_message=final_result.error_message,
                        )

                        print(
                            "[TrainingOrchestrator] shard failed permanently, "
                            "will retry missing trees through recovery loop:",
                            f"task_id={effective_shard.task_id}",
                            f"attempt_id={effective_shard.attempt_id}",
                            f"error={final_result.error_message}",
                            flush=True,
                        )

                        continue

                    completed_tree_ids_for_attempt = list(
                        final_result.completed_tree_ids
                    )

                    if not completed_tree_ids_for_attempt:
                        completed_tree_ids_for_attempt = [
                            artifact.tree_id
                            for artifact in final_result.tree_artifacts
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

                    artifact_by_tree_id.update(
                        self._collect_persisted_completed_artifacts(
                            job_id=job_id,
                            experiment_id=experiment_id,
                            forest_config=forest_config,
                        )
                    )

                    new_completed_count = len(
                        self._durable_completed_tree_ids(
                            job_id=job_id,
                            experiment_id=experiment_id,
                            forest_config=forest_config,
                        )
                    )

                    experiment.completed_tree_count = new_completed_count
                    self.job_repository.save_experiment(job_id, experiment)

                    self._record_shard_result(
                        metrics_collector=metrics_collector,
                        worker_id=final_result.worker_id,
                        task_id=effective_shard.task_id,
                        completed_tree_count=final_result.completed_tree_count,
                        failed_tree_count=final_result.failed_tree_count,
                        elapsed_time_seconds=final_result.elapsed_time_seconds,
                    )

                    self._metrics_event(
                        metrics_collector,
                        "shard_completed",
                        job_id=job_id,
                        experiment_id=experiment_id,
                        task_id=effective_shard.task_id,
                        attempt_id=effective_shard.attempt_id,
                        worker_id=effective_shard.assigned_worker_id,
                        completed_tree_ids=completed_tree_ids_for_attempt,
                        completed_tree_count=final_result.completed_tree_count,
                        failed_tree_count=final_result.failed_tree_count,
                        shard_elapsed_time_seconds=final_result.elapsed_time_seconds,
                        durable_completed_tree_count=new_completed_count,
                        expected_tree_count=forest_config.n_estimators,
                    )

                    print(
                        "[TrainingOrchestrator] shard completed:",
                        f"task_id={effective_shard.task_id}",
                        f"attempt_id={effective_shard.attempt_id}",
                        f"worker={effective_shard.assigned_worker_id}",
                        f"completed_tree_ids={completed_tree_ids_for_attempt}",
                        f"progress={new_completed_count}/"
                        f"{forest_config.n_estimators}",
                        flush=True,
                    )

                    self.job_repository.update_job_status(
                        job_id=job_id,
                        status=JobStatus.RUNNING,
                        message=(
                            f"Completed {new_completed_count}/"
                            f"{forest_config.n_estimators} trees"
                        ),
                    )

            artifact_by_tree_id.update(
                self._collect_persisted_completed_artifacts(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                )
            )

            durable_completed_count = len(
                self._durable_completed_tree_ids(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                )
            )

            if durable_completed_count <= last_completed_count:
                idle_rounds += 1
            else:
                idle_rounds = 0
                last_completed_count = durable_completed_count

            self._metrics_event(
                metrics_collector,
                "recovery_round_completed",
                job_id=job_id,
                experiment_id=experiment_id,
                recovery_round=recovery_round,
                durable_completed_tree_count=durable_completed_count,
                expected_tree_count=forest_config.n_estimators,
                idle_rounds=idle_rounds,
                had_failure=had_failure,
                elapsed_time_seconds=time.time() - round_started_at,
            )

            if had_failure and idle_rounds >= max_idle_rounds:
                raise RuntimeError(
                    f"Training recovery made no progress for "
                    f"{idle_rounds} consecutive rounds. "
                    f"Completed {durable_completed_count}/"
                    f"{forest_config.n_estimators} trees."
                )

        final_artifacts = self._collect_final_artifacts(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            artifact_by_tree_id=artifact_by_tree_id,
        )

        self._metrics_event(
            metrics_collector,
            "orchestrator_experiment_failed",
            job_id=job_id,
            experiment_id=experiment_id,
            reason="max_recovery_rounds_exceeded",
            final_artifact_count=len(final_artifacts),
            expected_tree_count=forest_config.n_estimators,
            elapsed_time_seconds=time.time() - experiment_started_at,
        )

        raise RuntimeError(
            f"Training recovery exceeded max rounds "
            f"({max_recovery_rounds}). "
            f"Final artifacts: {len(final_artifacts)}/"
            f"{forest_config.n_estimators}"
        )

    def _is_current_attempt(
        self,
        shard: TrainingShard,
        result: ShardTrainingResult,
    ) -> bool:
        latest_attempt_id = self.task_ledger.latest_attempt_id(
            job_id=shard.job_id,
            task_id=shard.task_id,
        )

        if latest_attempt_id is None:
            return True

        return int(latest_attempt_id) == int(result.attempt_id)

    def _execute_shard_with_retry(
        self,
        worker: WorkerLike,
        shard: TrainingShard,
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> tuple[TrainingShard, ShardTrainingResult, list[ShardTrainingResult]]:
        observed_results: list[ShardTrainingResult] = []

        current_worker = worker
        current_shard = shard

        while True:
            self._metrics_event(
                metrics_collector,
                "shard_attempt_started",
                job_id=current_shard.job_id,
                experiment_id=current_shard.experiment_id,
                task_id=current_shard.task_id,
                attempt_id=current_shard.attempt_id,
                worker_id=current_worker.worker_id,
                worker_host=current_worker.host,
                worker_port=current_worker.port,
                tree_start_index=current_shard.tree_start_index,
                tree_count=current_shard.tree_count,
            )

            print(
                "[TrainingOrchestrator] starting attempt:",
                f"task_id={current_shard.task_id}",
                f"attempt_id={current_shard.attempt_id}",
                f"worker={current_worker.worker_id}@{current_worker.host}:{current_worker.port}",
                f"tree_start={current_shard.tree_start_index}",
                f"tree_count={current_shard.tree_count}",
                flush=True,
            )

            current_shard, result = self._dispatch_attempt(
                worker=current_worker,
                shard=current_shard,
                metrics_collector=metrics_collector,
            )

            if not self._is_current_attempt(current_shard, result):
                print(
                    "[TrainingOrchestrator] stale attempt ignored:",
                    f"task_id={current_shard.task_id}",
                    f"attempt_id={result.attempt_id}",
                    flush=True,
                )

                self._metrics_event(
                    metrics_collector,
                    "stale_attempt_ignored",
                    job_id=current_shard.job_id,
                    experiment_id=current_shard.experiment_id,
                    task_id=current_shard.task_id,
                    shard_attempt_id=current_shard.attempt_id,
                    result_attempt_id=result.attempt_id,
                    worker_id=result.worker_id,
                )

                observed_results.append(result)

                return current_shard, result, observed_results

            observed_results.append(result)

            if result.success:
                print(
                    "[TrainingOrchestrator] attempt succeeded:",
                    f"task_id={current_shard.task_id}",
                    f"attempt_id={current_shard.attempt_id}",
                    f"worker={current_worker.worker_id}",
                    f"completed={result.completed_tree_count}",
                    f"failed={result.failed_tree_count}",
                    flush=True,
                )

                self._metrics_event(
                    metrics_collector,
                    "shard_attempt_succeeded",
                    job_id=current_shard.job_id,
                    experiment_id=current_shard.experiment_id,
                    task_id=current_shard.task_id,
                    attempt_id=current_shard.attempt_id,
                    worker_id=current_worker.worker_id,
                    completed_tree_count=result.completed_tree_count,
                    failed_tree_count=result.failed_tree_count,
                    elapsed_time_seconds=result.elapsed_time_seconds,
                )

                return current_shard, result, observed_results

            print(
                "[TrainingOrchestrator] attempt failed:",
                f"task_id={current_shard.task_id}",
                f"attempt_id={current_shard.attempt_id}",
                f"worker={current_worker.worker_id}",
                f"completed={result.completed_tree_count}",
                f"failed={result.failed_tree_count}",
                f"error={result.error_message}",
                flush=True,
            )

            self._metrics_event(
                metrics_collector,
                "shard_attempt_failed",
                job_id=current_shard.job_id,
                experiment_id=current_shard.experiment_id,
                task_id=current_shard.task_id,
                attempt_id=current_shard.attempt_id,
                worker_id=current_worker.worker_id,
                completed_tree_count=result.completed_tree_count,
                failed_tree_count=result.failed_tree_count,
                elapsed_time_seconds=result.elapsed_time_seconds,
                error_message=result.error_message,
            )

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
                self._metrics_event(
                    metrics_collector,
                    "shard_retry_not_allowed",
                    job_id=current_shard.job_id,
                    experiment_id=current_shard.experiment_id,
                    task_id=current_shard.task_id,
                    attempt_id=current_shard.attempt_id,
                    worker_id=current_worker.worker_id,
                    error_message=result.error_message,
                )

                return current_shard, result, observed_results

            retry_worker = self.worker_registry.get_retry_candidate(
                exclude_worker_id=current_worker.worker_id,
            )

            if retry_worker is None:
                print(
                    "[TrainingOrchestrator] retry skipped: no retry candidate",
                    f"task_id={current_shard.task_id}",
                    f"attempt_id={current_shard.attempt_id}",
                    flush=True,
                )

                self._metrics_event(
                    metrics_collector,
                    "shard_retry_skipped_no_candidate",
                    job_id=current_shard.job_id,
                    experiment_id=current_shard.experiment_id,
                    task_id=current_shard.task_id,
                    attempt_id=current_shard.attempt_id,
                    previous_worker_id=current_worker.worker_id,
                )

                return current_shard, result, observed_results

            backoff_seconds = self.retry_policy.backoff_seconds_for(
                attempt_id=current_shard.attempt_id,
            )

            self._record_retry(
                metrics_collector=metrics_collector,
                task_id=current_shard.task_id,
                worker_id=current_worker.worker_id,
                reason=result.error_message or "training shard failure",
            )

            self._metrics_event(
                metrics_collector,
                "shard_retry_scheduled",
                job_id=current_shard.job_id,
                experiment_id=current_shard.experiment_id,
                task_id=current_shard.task_id,
                failed_attempt_id=current_shard.attempt_id,
                previous_worker_id=current_worker.worker_id,
                retry_worker_id=retry_worker.worker_id,
                backoff_seconds=backoff_seconds,
            )

            if backoff_seconds > 0:
                print(
                    "[TrainingOrchestrator] retry backoff:",
                    f"task_id={current_shard.task_id}",
                    f"seconds={backoff_seconds}",
                    flush=True,
                )

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
        metrics_collector: Optional[ScalabilityMetricsCollector] = None,
    ) -> tuple[TrainingShard, ShardTrainingResult]:
        leased_shard = self.task_lease_manager.acquire(shard)

        self._metrics_event(
            metrics_collector,
            "task_lease_acquired",
            job_id=leased_shard.job_id,
            experiment_id=leased_shard.experiment_id,
            task_id=leased_shard.task_id,
            attempt_id=leased_shard.attempt_id,
            worker_id=worker.worker_id,
            lease_expires_at_ts=leased_shard.lease_expires_at_ts,
        )

        self.task_ledger.save(
            self._build_task_record(
                shard=leased_shard,
                status=TaskStatus.PENDING,
            )
        )

        self.task_ledger.mark_running(
            task_id=leased_shard.task_id,
            attempt_id=leased_shard.attempt_id,
            job_id=leased_shard.job_id,
        )

        self._metrics_event(
            metrics_collector,
            "task_marked_running",
            job_id=leased_shard.job_id,
            experiment_id=leased_shard.experiment_id,
            task_id=leased_shard.task_id,
            attempt_id=leased_shard.attempt_id,
            worker_id=worker.worker_id,
        )

        rpc_started_at = time.time()

        try:
            result = self.worker_client.train_shard(
                worker.host,
                worker.port,
                leased_shard,
            )
            rpc_success = True
            rpc_error_message = None

        except Exception as exc:
            rpc_success = False
            rpc_error_message = str(exc)

            result = self._build_failed_result(
                shard=leased_shard,
                worker_id=worker.worker_id,
                error_message=str(exc),
            )

        rpc_latency_seconds = time.time() - rpc_started_at

        normalized = self._normalize_result_identity(
            shard=leased_shard,
            fallback_worker_id=worker.worker_id,
            result=result,
        )

        self._record_train_rpc(
            metrics_collector=metrics_collector,
            worker_id=worker.worker_id,
            task_id=leased_shard.task_id,
            tree_count=leased_shard.tree_count,
            latency_seconds=rpc_latency_seconds,
            request_bytes=None,
            response_bytes=None,
            success=rpc_success and normalized.success,
        )

        self._metrics_event(
            metrics_collector,
            "train_rpc_completed",
            job_id=leased_shard.job_id,
            experiment_id=leased_shard.experiment_id,
            task_id=leased_shard.task_id,
            attempt_id=leased_shard.attempt_id,
            worker_id=worker.worker_id,
            worker_host=worker.host,
            worker_port=worker.port,
            tree_count=leased_shard.tree_count,
            latency_seconds=rpc_latency_seconds,
            rpc_success=rpc_success,
            result_success=normalized.success,
            completed_tree_count=normalized.completed_tree_count,
            failed_tree_count=normalized.failed_tree_count,
            error_message=rpc_error_message or normalized.error_message,
        )

        return leased_shard, normalized

    # --------------------------------------------------------
    # scalability metrics helpers
    # --------------------------------------------------------

    def _metrics_event(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        event: str,
        **payload,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_event(event, **payload)
        except Exception as exc:
            print(
                "[TrainingOrchestrator] Failed to record scalability event "
                f"'{event}': {exc}",
                flush=True,
            )

    def _metrics_timer(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        name: str,
        **payload,
    ):
        if metrics_collector is None:
            return nullcontext()

        try:
            return metrics_collector.timer(name, **payload)
        except Exception:
            return nullcontext()

    def _record_training_plan(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        worker_count: int,
        shard_count: int,
        n_estimators_total: int,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_training_plan(
                worker_count=worker_count,
                shard_count=shard_count,
                n_estimators_total=n_estimators_total,
            )
        except AttributeError:
            self._metrics_event(
                metrics_collector,
                "training_plan_created",
                worker_count=worker_count,
                shard_count=shard_count,
                n_estimators_total=n_estimators_total,
            )
        except Exception as exc:
            print(
                "[TrainingOrchestrator] Failed to record training plan metrics: "
                f"{exc}",
                flush=True,
            )

    def _record_train_rpc(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        worker_id: str,
        task_id: str,
        tree_count: int,
        latency_seconds: float,
        request_bytes: Optional[int],
        response_bytes: Optional[int],
        success: bool,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_train_rpc(
                worker_id=worker_id,
                task_id=task_id,
                tree_count=tree_count,
                latency_seconds=latency_seconds,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                success=success,
            )
        except AttributeError:
            self._metrics_event(
                metrics_collector,
                "train_rpc_completed",
                worker_id=worker_id,
                task_id=task_id,
                tree_count=tree_count,
                latency_seconds=latency_seconds,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                success=success,
            )
        except Exception as exc:
            print(
                "[TrainingOrchestrator] Failed to record train RPC metrics: "
                f"{exc}",
                flush=True,
            )

    def _record_shard_result(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        worker_id: str,
        task_id: str,
        completed_tree_count: int,
        failed_tree_count: int,
        elapsed_time_seconds: float,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_shard_result(
                worker_id=worker_id,
                task_id=task_id,
                completed_tree_count=completed_tree_count,
                failed_tree_count=failed_tree_count,
                elapsed_time_seconds=elapsed_time_seconds,
            )
        except AttributeError:
            self._metrics_event(
                metrics_collector,
                "shard_result_recorded",
                worker_id=worker_id,
                task_id=task_id,
                completed_tree_count=completed_tree_count,
                failed_tree_count=failed_tree_count,
                elapsed_time_seconds=elapsed_time_seconds,
            )
        except Exception as exc:
            print(
                "[TrainingOrchestrator] Failed to record shard result metrics: "
                f"{exc}",
                flush=True,
            )

    def _record_retry(
        self,
        metrics_collector: Optional[ScalabilityMetricsCollector],
        task_id: str,
        worker_id: str,
        reason: str,
    ) -> None:
        if metrics_collector is None:
            return

        try:
            metrics_collector.record_retry(
                task_id=task_id,
                worker_id=worker_id,
                reason=reason,
            )
        except AttributeError:
            self._metrics_event(
                metrics_collector,
                "shard_retry",
                task_id=task_id,
                worker_id=worker_id,
                reason=reason,
            )
        except Exception as exc:
            print(
                "[TrainingOrchestrator] Failed to record retry metrics: "
                f"{exc}",
                flush=True,
            )

    # --------------------------------------------------------
    # existing orchestration helpers
    # --------------------------------------------------------

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

    def _load_or_initialize_experiment(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> ExperimentRecord:
        experiment = self.job_repository.load_experiment(
            job_id,
            experiment_id,
        )

        if experiment is not None:
            return experiment

        experiment = ExperimentRecord(
            experiment_id=experiment_id,
            forest_config=forest_config,
            status=ExperimentStatus.PENDING,
            assigned_workers=[],
            expected_tree_count=forest_config.n_estimators,
            completed_tree_count=0,
            validation_metrics=None,
        )

        self.job_repository.save_experiment(job_id, experiment)

        return experiment

    def _completed_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
    ) -> list[str]:
        return self.task_ledger.completed_tree_ids(
            job_id=job_id,
            experiment_id=experiment_id,
        )

    def _durable_completed_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> list[str]:
        expected_tree_ids = self._expected_tree_ids(
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        expected_set = set(expected_tree_ids)

        completed = {
            tree_id
            for tree_id in self.task_ledger.completed_tree_ids(
                job_id=job_id,
                experiment_id=experiment_id,
            )
            if tree_id in expected_set
        }

        for tree_id in expected_tree_ids:
            metadata = self._load_tree_metadata(
                job_id=job_id,
                experiment_id=experiment_id,
                tree_id=tree_id,
            )

            if metadata is None:
                continue

            if metadata.status != TreeStatus.COMPLETED:
                continue

            if not self._tree_artifact_exists(
                job_id=job_id,
                experiment_id=experiment_id,
                tree_index=metadata.tree_index,
            ):
                continue

            completed.add(tree_id)

        return [
            tree_id
            for tree_id in expected_tree_ids
            if tree_id in completed
        ]

    def _tree_artifact_exists(
        self,
        job_id: str,
        experiment_id: str,
        tree_index: int,
    ) -> bool:
        artifact_path = (
            self.task_ledger.artifact_store.layout.root
            / "jobs"
            / job_id
            / "experiments"
            / experiment_id
            / "trees"
            / f"tree_{tree_index}.joblib"
        )

        return self.task_ledger.artifact_store.exists(artifact_path)

    def _missing_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> list[str]:
        expected_tree_ids = self._expected_tree_ids(
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        completed_tree_ids = set(
            self._durable_completed_tree_ids(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
            )
        )

        return [
            tree_id
            for tree_id in expected_tree_ids
            if tree_id not in completed_tree_ids
        ]

    def _plan_missing_shards(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        prepared_dataset,
        workers: list[WorkerLike],
        missing_tree_ids: list[str],
    ) -> list[TrainingShard]:
        attempts = self.task_ledger.list_attempts_by_experiment(
            job_id=job_id,
            experiment_id=experiment_id,
        )

        attempt_id = (
            1
            if not attempts
            else max(record.attempt_id for record in attempts) + 1
        )

        expected_tree_ids = self._expected_tree_ids(
            experiment_id=experiment_id,
            forest_config=forest_config,
        )

        is_initial_full_training_plan = self._is_initial_full_training_plan(
            attempts=attempts,
            missing_tree_ids=missing_tree_ids,
            expected_tree_ids=expected_tree_ids,
        )

        if is_initial_full_training_plan:
            return self.shard_planner.plan(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
                prepared_dataset=prepared_dataset,
                workers=workers,
                missing_tree_ids=None,
                attempt_id=attempt_id,
            )

        return self.shard_planner.plan_missing_tree_ids(
            job_id=job_id,
            experiment_id=experiment_id,
            forest_config=forest_config,
            prepared_dataset=prepared_dataset,
            workers=workers,
            missing_tree_ids=missing_tree_ids,
            attempt_id=attempt_id,
        )

    def _find_worker_or_raise(
        self,
        workers: list[WorkerLike],
        worker_id: str,
    ) -> WorkerLike:
        for worker in workers:
            if worker.worker_id == worker_id:
                return worker

        raise RuntimeError(f"Worker '{worker_id}' is not alive or not registered")

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
            tree_ids=self._tree_ids_for_shard(shard),
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
        failed_tree_ids = self._tree_ids_for_shard(shard)

        return ShardTrainingResult(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=worker_id,
            success=False,
            tree_artifacts=[],
            completed_tree_ids=[],
            failed_tree_ids=failed_tree_ids,
            completed_tree_count=0,
            failed_tree_count=len(failed_tree_ids),
            error_message=error_message,
            elapsed_time_seconds=0.0,
        )

    def _normalize_result_identity(
        self,
        shard: TrainingShard,
        fallback_worker_id: str,
        result: ShardTrainingResult,
    ) -> ShardTrainingResult:
        completed_tree_ids = list(dict.fromkeys(result.completed_tree_ids))

        if not completed_tree_ids and result.tree_artifacts:
            completed_tree_ids = [
                artifact.tree_id
                for artifact in result.tree_artifacts
            ]

        all_tree_ids = self._tree_ids_for_shard(shard)

        failed_tree_ids = list(dict.fromkeys(result.failed_tree_ids))

        if not result.success and not failed_tree_ids:
            completed_set = set(completed_tree_ids)
            failed_tree_ids = [
                tree_id
                for tree_id in all_tree_ids
                if tree_id not in completed_set
            ]

        success = result.success and len(failed_tree_ids) == 0

        return ShardTrainingResult(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            worker_id=result.worker_id or fallback_worker_id,
            success=success,
            tree_artifacts=list(result.tree_artifacts),
            completed_tree_ids=completed_tree_ids,
            failed_tree_ids=failed_tree_ids,
            completed_tree_count=len(completed_tree_ids),
            failed_tree_count=len(failed_tree_ids),
            error_message=result.error_message,
            elapsed_time_seconds=result.elapsed_time_seconds,
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
            lease_expires_at_ts=None,
        )

    def _register_result_artifacts(
        self,
        artifact_by_tree_id: dict[str, TreeArtifactMetadata],
        result: ShardTrainingResult,
    ) -> None:
        completed = set(result.completed_tree_ids)

        for artifact in result.tree_artifacts:
            if artifact.tree_id in completed or artifact.status == TreeStatus.COMPLETED:
                artifact_by_tree_id[artifact.tree_id] = artifact

    def _collect_persisted_completed_artifacts(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> dict[str, TreeArtifactMetadata]:
        result: dict[str, TreeArtifactMetadata] = {}

        for tree_id in self._expected_tree_ids(
            experiment_id=experiment_id,
            forest_config=forest_config,
        ):
            metadata = self._load_tree_metadata(
                job_id=job_id,
                experiment_id=experiment_id,
                tree_id=tree_id,
            )

            if metadata is not None and metadata.status == TreeStatus.COMPLETED:
                result[tree_id] = metadata

        return result

    def _collect_final_artifacts(
        self,
        job_id: str,
        experiment_id: str,
        forest_config: ForestConfiguration,
        artifact_by_tree_id: dict[str, TreeArtifactMetadata],
    ) -> list[TreeArtifactMetadata]:
        final_artifacts: list[TreeArtifactMetadata] = []

        for tree_index in range(forest_config.n_estimators):
            tree_id = generate_tree_id(experiment_id, tree_index)

            artifact = artifact_by_tree_id.get(tree_id)

            if artifact is None:
                artifact = self._load_tree_metadata(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    tree_id=tree_id,
                )

            if artifact is not None and artifact.status == TreeStatus.COMPLETED:
                final_artifacts.append(artifact)

        final_artifacts.sort(key=lambda item: item.tree_index)

        return final_artifacts

    def _load_tree_metadata(
        self,
        job_id: str,
        experiment_id: str,
        tree_id: str,
    ) -> TreeArtifactMetadata | None:
        tree_index = self._tree_index_from_tree_id(
            experiment_id=experiment_id,
            tree_id=tree_id,
        )

        metadata_path = (
            self.task_ledger.artifact_store.layout.root
            / "jobs"
            / job_id
            / "experiments"
            / experiment_id
            / "trees"
            / f"tree_{tree_index}.json"
        )

        if not self.task_ledger.artifact_store.exists(metadata_path):
            return None

        payload = self.task_ledger.artifact_store.read_json(metadata_path)

        return TreeArtifactMetadata.from_dict(payload)

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

        return int(tree_id[len(prefix):])

    def _expected_tree_ids(
        self,
        experiment_id: str,
        forest_config: ForestConfiguration,
    ) -> list[str]:
        return [
            generate_tree_id(experiment_id, tree_index)
            for tree_index in range(forest_config.n_estimators)
        ]

    def _tree_ids_for_shard(
        self,
        shard: TrainingShard,
    ) -> list[str]:
        return [
            generate_tree_id(shard.experiment_id, tree_index)
            for tree_index in range(
                shard.tree_start_index,
                shard.tree_start_index + shard.tree_count,
            )
        ]

    def _is_initial_full_training_plan(
        self,
        attempts,
        missing_tree_ids: list[str],
        expected_tree_ids: list[str],
    ) -> bool:
        return (
            not attempts
            and set(missing_tree_ids) == set(expected_tree_ids)
        )

    def _effective_max_parallel_shards(
        self,
        shard_count: int,
    ) -> int:
        if shard_count <= 0:
            return 0

        max_workers = self.max_parallel_shards or shard_count
        max_workers = min(max_workers, shard_count)

        return max_workers

    def _log_alive_workers(
        self,
        alive_workers: list[WorkerLike],
    ) -> None:
        print(
            "[TrainingOrchestrator] alive_workers:",
            [self._format_worker(worker) for worker in alive_workers],
            flush=True,
        )

    def _log_planned_shards(
        self,
        shards: list[TrainingShard],
    ) -> None:
        print(
            "[TrainingOrchestrator] planned_shards:",
            [
                {
                    "task_id": shard.task_id,
                    "attempt_id": shard.attempt_id,
                    "worker_id": shard.assigned_worker_id,
                    "tree_start_index": shard.tree_start_index,
                    "tree_count": shard.tree_count,
                }
                for shard in shards
            ],
            flush=True,
        )

    def _format_worker(
        self,
        worker: WorkerLike,
    ) -> str:
        return f"{worker.worker_id}@{worker.host}:{worker.port}"