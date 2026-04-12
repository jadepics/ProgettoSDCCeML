from __future__ import annotations

from dataclasses import dataclass

from common.contracts import TaskRecord, TrainingShard
from common.ids import generate_tree_id


@dataclass(frozen=True)
class RecoveryPlan:
    job_id: str
    experiment_id: str
    expected_tree_ids: list[str]
    completed_tree_ids: list[str]
    missing_tree_ids: list[str]
    expired_running_tasks: list[TaskRecord]
    stale_worker_ids: list[str]
    recovery_shards: list[TrainingShard]


class RecoveryPlanner:
    """
    Responsabilità:
    - osservare lo stato persistito di un esperimento
    - capire quali tree sono già completati
    - identificare tree mancanti
    - considerare task RUNNING con lease scaduta
    - considerare worker stale
    - produrre shard di recovery solo per il lavoro mancante

    Nota:
    questa prima versione NON muta stato e NON dispatcha nulla.
    Produce solo un RecoveryPlan.
    """

    def __init__(
        self,
        task_ledger,
        shard_planner,
        worker_heartbeat_monitor,
    ) -> None:
        self.task_ledger = task_ledger
        self.shard_planner = shard_planner
        self.worker_heartbeat_monitor = worker_heartbeat_monitor

    def build_plan(
        self,
        job_id: str,
        experiment_id: str,
        forest_config,
        prepared_dataset,
        workers,
    ) -> RecoveryPlan:
        expected_tree_ids = self._expected_tree_ids(
            experiment_id=experiment_id,
            n_estimators=forest_config.n_estimators,
        )

        completed_tree_ids = self.task_ledger.completed_tree_ids(
            job_id=job_id,
            experiment_id=experiment_id,
        )

        completed_set = set(completed_tree_ids)
        missing_tree_ids = [
            tree_id for tree_id in expected_tree_ids
            if tree_id not in completed_set
        ]

        expired_running_tasks = self.task_ledger.list_expired_running_tasks(
            job_id=job_id,
            experiment_id=experiment_id,
        )

        stale_worker_ids = self.worker_heartbeat_monitor.stale_worker_ids()

        if not missing_tree_ids:
            recovery_shards: list[TrainingShard] = []
        else:
            recovery_shards = self.shard_planner.plan_missing_tree_ids(
                job_id=job_id,
                experiment_id=experiment_id,
                forest_config=forest_config,
                prepared_dataset=prepared_dataset,
                workers=workers,
                missing_tree_ids=missing_tree_ids,
                attempt_id=1,
            )

        return RecoveryPlan(
            job_id=job_id,
            experiment_id=experiment_id,
            expected_tree_ids=expected_tree_ids,
            completed_tree_ids=list(completed_tree_ids),
            missing_tree_ids=missing_tree_ids,
            expired_running_tasks=list(expired_running_tasks),
            stale_worker_ids=list(stale_worker_ids),
            recovery_shards=list(recovery_shards),
        )

    def _expected_tree_ids(
        self,
        experiment_id: str,
        n_estimators: int,
    ) -> list[str]:
        return [
            generate_tree_id(experiment_id, index)
            for index in range(n_estimators)
        ]