from pathlib import Path
from typing import Dict, Any, Optional
import json

from worker.utils.io_utils import atomic_json_write
from worker.utils.time_utils import current_time_seconds
from worker.storage.paths import worker_snapshot_path


class WorkerProgressStore:
    """
    Persistent store for tracking worker progress per job/experiment.

    Now:
    - one snapshot per (job_id, experiment_id, worker_id)
    - idempotent start_task
    """

    def __init__(self, base_storage_dir: Path, worker_id: str):
        self.base_storage_dir = Path(base_storage_dir)
        self.worker_id = worker_id

        # in-memory state keyed by (job_id, experiment_id)
        self.state: Dict[str, Any] = {
            "tasks": {}
        }

    # ------------------------
    # Internal helpers
    # ------------------------

    def _get_snapshot_path(self, job_id: str, experiment_id: str) -> Path:
        relative_path = worker_snapshot_path(job_id, experiment_id, self.worker_id)
        return self.base_storage_dir / relative_path

    def _load_snapshot(self, job_id: str, experiment_id: str) -> None:
        path = self._get_snapshot_path(job_id, experiment_id)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {"tasks": {}}

    def _persist_snapshot(self, job_id: str, experiment_id: str) -> None:
        path = self._get_snapshot_path(job_id, experiment_id)
        atomic_json_write(path, self.state)

    # ------------------------
    # Lifecycle per job
    # ------------------------

    def load(self, job_id: str, experiment_id: str) -> None:
        self._load_snapshot(job_id, experiment_id)

    def save(self, job_id: str, experiment_id: str) -> None:
        self._persist_snapshot(job_id, experiment_id)

    # ------------------------
    # Task management (IDEMPOTENT)
    # ------------------------

    def start_task(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Idempotent start.

        Returns:
            - "STARTED"
            - "RETRY"
            - "ALREADY_COMPLETED"
        """
        self._load_snapshot(job_id, experiment_id)

        existing = self.state["tasks"].get(task_id)
        attempt_id = metadata.get("attempt_id")

        if existing:
            # 🔴 Task già completato → idempotenza forte
            if existing.get("status") == "COMPLETED":
                return "ALREADY_COMPLETED"

            # 🔴 Stesso attempt → retry
            if existing.get("attempt_id") == attempt_id:
                return "RETRY"

            # 🔴 Nuovo attempt → overwrite controllato
            # (puoi in futuro loggare o gestire diversamente)

        # Nuovo task
        self.state["tasks"][task_id] = {
            "status": "RUNNING",
            "attempt_id": attempt_id,
            "metadata": metadata,
            "progress": 0.0,
            "shards_completed": [],
            "completed_tree_ids": [],
            "failed_tree_ids": [],
            "last_update": current_time_seconds()
        }

        self._persist_snapshot(job_id, experiment_id)
        return "STARTED"

    def update_progress(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str,
        shard_id: int,
        progress: float
    ) -> None:
        self._load_snapshot(job_id, experiment_id)

        task = self._get_task(task_id)

        if shard_id not in task["shards_completed"]:
            task["shards_completed"].append(shard_id)

        task["progress"] = progress
        task["last_update"] = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    def complete_task(self, job_id: str, experiment_id: str, task_id: str) -> None:
        self._load_snapshot(job_id, experiment_id)

        task = self._get_task(task_id)
        task["status"] = "COMPLETED"
        task["progress"] = 1.0
        task["last_update"] = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    def fail_task(self, job_id: str, experiment_id: str, task_id: str, error: str) -> None:
        self._load_snapshot(job_id, experiment_id)

        task = self._get_task(task_id)
        task["status"] = "FAILED"
        task["error"] = error
        task["last_update"] = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    # ------------------------
    # Queries
    # ------------------------

    def get_task(self, job_id: str, experiment_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        self._load_snapshot(job_id, experiment_id)
        return self.state["tasks"].get(task_id)

    def _get_task(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.state["tasks"]:
            raise KeyError(f"Task {task_id} not found")
        return self.state["tasks"][task_id]

    def get_running_tasks(self, job_id: str, experiment_id: str):
        self._load_snapshot(job_id, experiment_id)

        return {
            tid: t for tid, t in self.state["tasks"].items()
            if t["status"] == "RUNNING"
        }

    def get_completed_shards(self, job_id: str, experiment_id: str, task_id: str):
        self._load_snapshot(job_id, experiment_id)
        task = self._get_task(task_id)
        return set(task.get("shards_completed", []))