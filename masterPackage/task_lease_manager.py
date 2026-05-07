from __future__ import annotations

import time

from common.contracts import TaskRecord, TrainingShard


class TaskLeaseManager:
    """
    Gestisce lease temporanee dei task lato master.

    Obiettivi:
    - assegnare una lease expiration agli shard prima del dispatch
    - rinnovare la lease di un attempt già persistito
    - rilasciare la lease quando il task non è più RUNNING
    - individuare attempt RUNNING con lease scaduta

    Nota importante:
    - acquire(...) NON persiste direttamente nel TaskLedger
    - acquire(...) restituisce uno shard con lease_expires_at_ts valorizzato
    - spetta al chiamante persistire subito il relativo TaskRecord
      prima del dispatch al worker
    """

    def __init__(
        self,
        task_ledger,
        lease_timeout_seconds: float = 600.0,
    ) -> None:
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be > 0")

        self.task_ledger = task_ledger
        self.lease_timeout_seconds = lease_timeout_seconds

    def acquire(
        self,
        shard: TrainingShard,
    ) -> TrainingShard:
        """
        Returns a copy of the shard with a fresh lease expiration timestamp.
        Persistence in TaskLedger is intentionally left to the caller.
        """
        expires_at = self._expires_at()

        return TrainingShard(
            task_id=shard.task_id,
            attempt_id=shard.attempt_id,
            job_id=shard.job_id,
            experiment_id=shard.experiment_id,
            assigned_worker_id=shard.assigned_worker_id,
            tree_start_index=shard.tree_start_index,
            tree_count=shard.tree_count,
            forest_config=shard.forest_config,
            train_features_uri=shard.train_features_uri,
            train_labels_uri=shard.train_labels_uri,
            artifact_output_dir=shard.artifact_output_dir,
            seed_base=shard.seed_base,
            lease_expires_at_ts=expires_at,
        )

    def renew(
        self,
        job_id: str,
        task_id: str,
        attempt_id: int,
    ) -> float:
        expires_at = self._expires_at()

        self.task_ledger.update_lease(
            task_id=task_id,
            attempt_id=attempt_id,
            job_id=job_id,
            lease_expires_at_ts=expires_at,
        )

        return expires_at

    def release(
        self,
        job_id: str,
        task_id: str,
        attempt_id: int,
    ) -> None:
        self.task_ledger.clear_lease(
            task_id=task_id,
            attempt_id=attempt_id,
            job_id=job_id,
        )

    def expired_running_tasks(
        self,
        job_id: str,
        experiment_id: str | None = None,
        now_ts: float | None = None,
    ) -> list[TaskRecord]:
        return self.task_ledger.list_expired_running_tasks(
            job_id=job_id,
            experiment_id=experiment_id,
            now_ts=now_ts,
        )

    def _expires_at(self) -> float:
        return time.time() + self.lease_timeout_seconds