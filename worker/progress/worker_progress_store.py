from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from common.contracts import WorkerProgressSnapshot


class WorkerProgressStore:

    def __init__(self, store, paths):
        self.store = store
        self.paths = paths

    def save_snapshot(self, snapshot: WorkerProgressSnapshot):
        key = self.paths.worker_snapshot_path(
            snapshot.experiment_id,
            snapshot.worker_id,
            snapshot.task_id,
        )

        self.store.save_json(key, asdict(snapshot))

    def load_snapshot(
        self,
        experiment_id: str,
        worker_id: str,
        task_id: str,
    ) -> Optional[WorkerProgressSnapshot]:

        key = self.paths.worker_snapshot_path(
            experiment_id,
            worker_id,
            task_id,
        )

        if not self.store.exists(key):
            return None

        data = self.store.load_json(key)
        return WorkerProgressSnapshot(**data)

    def snapshot_exists(
        self,
        experiment_id: str,
        worker_id: str,
        task_id: str,
    ) -> bool:

        key = self.paths.worker_snapshot_path(
            experiment_id,
            worker_id,
            task_id,
        )

        return self.store.exists(key)