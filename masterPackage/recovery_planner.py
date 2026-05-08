from __future__ import annotations

import time
from dataclasses import dataclass

from common.contracts import TaskRecord, TrainingShard
from common.enums import TaskStatus
from common.ids import generate_tree_id


@dataclass(frozen=True)
class RecoveryDecision:
    tree_id: str
    action: str
    reason: str
    owner_task_id: str | None
    owner_worker_id: str | None


@dataclass(frozen=True)
class RecoveryPlan:
    job_id: str
    experiment_id: str
    expected_tree_ids: list[str]
    completed_tree_ids: list[str]
    all_missing_tree_ids: list[str]
    recover_now_tree_ids: list[str]
    deferred_tree_ids: list[str]
    expired_running_tasks: list[TaskRecord]
    stale_worker_ids: list[str]
    decisions: list[RecoveryDecision]
    recovery_shards: list[TrainingShard]


class RecoveryPlanner:
    """
    Responsabilità:
    - osservare lo stato persistito di un esperimento
    - capire quali tree sono già completati
    - identificare tree mancanti
    - distinguere tra task probabilmente solo lenti e task abbandonati
    - produrre shard di recovery solo per i tree da recuperare subito

    Politica base:
    - tree già completed -> nessuna azione
    - tree missing senza owner RUNNING -> recupera subito
    - tree missing con owner RUNNING e lease non scaduta -> rimanda
    - tree missing con owner RUNNING, lease scaduta, worker stale -> recupera subito
    - tree missing con owner RUNNING, lease scaduta, worker vivo:
        - se scaduta da poco -> rimanda
        - se oltre grace period -> recupera subito
    """

    def __init__(
        self,
        task_ledger,
        shard_planner,
        worker_heartbeat_monitor,
        alive_worker_expired_lease_grace_seconds: float = 30.0,
    ) -> None:
        if alive_worker_expired_lease_grace_seconds < 0:
            raise ValueError("alive_worker_expired_lease_grace_seconds must be >= 0")

        self.task_ledger = task_ledger
        self.shard_planner = shard_planner
        self.worker_heartbeat_monitor = worker_heartbeat_monitor
        self.alive_worker_expired_lease_grace_seconds = (
            alive_worker_expired_lease_grace_seconds
        )

    def build_plan(
        self,
        job_id: str,
        experiment_id: str,
        forest_config,
        prepared_dataset,
        workers,
        now_ts: float | None = None,
    ) -> RecoveryPlan:
        effective_now = time.time() if now_ts is None else now_ts

        expected_tree_ids = self._expected_tree_ids(
            experiment_id=experiment_id,
            n_estimators=forest_config.n_estimators,
        )

        completed_tree_ids = self.task_ledger.completed_tree_ids(
            job_id=job_id,
            experiment_id=experiment_id,
        )
        completed_set = set(completed_tree_ids)

        all_missing_tree_ids = [
            tree_id
            for tree_id in expected_tree_ids
            if tree_id not in completed_set
        ]

        expired_running_tasks = self.task_ledger.list_expired_running_tasks(
            job_id=job_id,
            experiment_id=experiment_id,
            now_ts=effective_now,
        )

        stale_worker_ids = self._stale_worker_ids(now_ts=effective_now)
        stale_worker_ids_set = set(stale_worker_ids)

        running_owner_by_tree = self._running_owner_by_tree(
            job_id=job_id,
            experiment_id=experiment_id,
        )

        decisions: list[RecoveryDecision] = []
        recover_now_tree_ids: list[str] = []
        deferred_tree_ids: list[str] = []

        for tree_id in all_missing_tree_ids:
            owner = running_owner_by_tree.get(tree_id)

            decision = self._decide_missing_tree(
                tree_id=tree_id,
                owner=owner,
                stale_worker_ids=stale_worker_ids_set,
                now_ts=effective_now,
            )
            decisions.append(decision)

            if decision.action == "recover_now":
                recover_now_tree_ids.append(tree_id)
            else:
                deferred_tree_ids.append(tree_id)

        recovery_shards: list[TrainingShard] = []

        if recover_now_tree_ids and workers:
            recovery_attempt_id = self._next_recovery_attempt_id(
                job_id=job_id,
                experiment_id=experiment_id,
            )

            recovery_shards = list(
                self.shard_planner.plan_missing_tree_ids(
                    job_id=job_id,
                    experiment_id=experiment_id,
                    forest_config=forest_config,
                    prepared_dataset=prepared_dataset,
                    workers=workers,
                    missing_tree_ids=recover_now_tree_ids,
                    attempt_id=recovery_attempt_id,
                )
            )

        return RecoveryPlan(
            job_id=job_id,
            experiment_id=experiment_id,
            expected_tree_ids=expected_tree_ids,
            completed_tree_ids=list(completed_tree_ids),
            all_missing_tree_ids=all_missing_tree_ids,
            recover_now_tree_ids=recover_now_tree_ids,
            deferred_tree_ids=deferred_tree_ids,
            expired_running_tasks=list(expired_running_tasks),
            stale_worker_ids=stale_worker_ids,
            decisions=decisions,
            recovery_shards=recovery_shards,
        )

    def _expected_tree_ids(
        self,
        experiment_id: str,
        n_estimators: int,
    ) -> list[str]:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")

        return [
            generate_tree_id(experiment_id, index)
            for index in range(n_estimators)
        ]

    def _running_owner_by_tree(
        self,
        job_id: str,
        experiment_id: str,
    ) -> dict[str, TaskRecord]:
        attempts = self.task_ledger.list_all_attempts(job_id)

        running_attempts = [
            record
            for record in attempts
            if record.experiment_id == experiment_id
            and record.status == TaskStatus.RUNNING
        ]

        running_attempts.sort(
            key=lambda record: (record.updated_at, record.attempt_id),
            reverse=True,
        )

        owner_by_tree: dict[str, TaskRecord] = {}

        for record in running_attempts:
            for tree_id in self._record_tree_ids(record):
                if tree_id not in owner_by_tree:
                    owner_by_tree[tree_id] = record

        return owner_by_tree

    def _decide_missing_tree(
        self,
        tree_id: str,
        owner: TaskRecord | None,
        stale_worker_ids: set[str],
        now_ts: float,
    ) -> RecoveryDecision:
        if owner is None:
            return RecoveryDecision(
                tree_id=tree_id,
                action="recover_now",
                reason="missing tree has no RUNNING owner in persisted ledger",
                owner_task_id=None,
                owner_worker_id=None,
            )

        owner_worker_id = self._record_worker_id(owner)
        lease_expires_at_ts = owner.lease_expires_at_ts
        worker_is_stale = owner_worker_id in stale_worker_ids

        if lease_expires_at_ts is None:
            if worker_is_stale:
                return RecoveryDecision(
                    tree_id=tree_id,
                    action="recover_now",
                    reason="owner worker is stale and RUNNING task has no active lease",
                    owner_task_id=owner.task_id,
                    owner_worker_id=owner_worker_id,
                )

            return RecoveryDecision(
                tree_id=tree_id,
                action="defer",
                reason="owner task is RUNNING with no lease expiry but worker is still alive",
                owner_task_id=owner.task_id,
                owner_worker_id=owner_worker_id,
            )

        if lease_expires_at_ts > now_ts:
            return RecoveryDecision(
                tree_id=tree_id,
                action="defer",
                reason="owner task is RUNNING and lease is still active",
                owner_task_id=owner.task_id,
                owner_worker_id=owner_worker_id,
            )

        overdue_seconds = now_ts - lease_expires_at_ts

        if worker_is_stale:
            return RecoveryDecision(
                tree_id=tree_id,
                action="recover_now",
                reason="owner task lease expired and owner worker is stale",
                owner_task_id=owner.task_id,
                owner_worker_id=owner_worker_id,
            )

        if overdue_seconds <= self.alive_worker_expired_lease_grace_seconds:
            return RecoveryDecision(
                tree_id=tree_id,
                action="defer",
                reason=(
                    "owner task lease expired but worker is still alive and within "
                    "grace period"
                ),
                owner_task_id=owner.task_id,
                owner_worker_id=owner_worker_id,
            )

        return RecoveryDecision(
            tree_id=tree_id,
            action="recover_now",
            reason=(
                "owner task lease expired, worker is alive, but grace period "
                "was exceeded"
            ),
            owner_task_id=owner.task_id,
            owner_worker_id=owner_worker_id,
        )

    def _next_recovery_attempt_id(
        self,
        job_id: str,
        experiment_id: str,
    ) -> int:
        attempts = self.task_ledger.list_attempts_by_experiment(
            job_id=job_id,
            experiment_id=experiment_id,
        )

        if not attempts:
            return 1

        return max(record.attempt_id for record in attempts) + 1

    def _stale_worker_ids(self, now_ts: float) -> list[str]:
        """
        Supporta sia monitor che espongono stale_workers(now_ts),
        sia monitor che espongono stale_worker_ids().
        """
        if hasattr(self.worker_heartbeat_monitor, "stale_workers"):
            stale_snapshots = self.worker_heartbeat_monitor.stale_workers(
                now_ts=now_ts
            )
            return [
                snapshot.worker_id
                for snapshot in stale_snapshots
            ]

        if hasattr(self.worker_heartbeat_monitor, "stale_worker_ids"):
            return list(self.worker_heartbeat_monitor.stale_worker_ids())

        return []

    def _record_worker_id(self, record: TaskRecord) -> str | None:
        worker_id = getattr(record, "worker_id", None)
        if worker_id is not None:
            return worker_id

        return getattr(record, "assigned_worker_id", None)

    def _record_tree_ids(self, record: TaskRecord) -> list[str]:
        tree_ids = getattr(record, "tree_ids", None)
        if tree_ids is None:
            return []
        return list(tree_ids)