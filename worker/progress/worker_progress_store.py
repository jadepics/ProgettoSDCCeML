from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

from worker.storage.artifact_store import ArtifactStore
from worker.utils.time_utils import current_time_seconds
from worker.storage.paths import worker_snapshot_path

from common.contracts import WorkerProgressSnapshot


# ======================================================
# SNAPSHOT CONTRACTS (allineati e tipizzati)
# ======================================================

@dataclass(slots=True)
class TaskProgressSnapshot:
    task_id: str
    attempt_id: int
    status: str  # RUNNING | COMPLETED | FAILED
    progress: float
    completed_tree_ids: List[str] = field(default_factory=list)
    failed_tree_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    last_update: float = 0.0

    @staticmethod
    def from_dict(data: dict) -> "TaskProgressSnapshot":
        return TaskProgressSnapshot(
            task_id=data["task_id"],
            attempt_id=data["attempt_id"],
            status=data["status"],
            progress=data.get("progress", 0.0),
            completed_tree_ids=data.get("completed_tree_ids", []),
            failed_tree_ids=data.get("failed_tree_ids", []),
            error_message=data.get("error_message"),
            last_update=data.get("last_update", 0.0),
        )


@dataclass(slots=True)
class WorkerProgressState:
    worker_id: str
    job_id: str
    experiment_id: str
    tasks: Dict[str, TaskProgressSnapshot] = field(default_factory=dict)
    last_update: float = 0.0

    @staticmethod
    def from_dict(data: dict) -> "WorkerProgressState":
        tasks = {
            tid: TaskProgressSnapshot.from_dict(tdata)
            for tid, tdata in data.get("tasks", {}).items()
        }
        return WorkerProgressState(
            worker_id=data["worker_id"],
            job_id=data["job_id"],
            experiment_id=data["experiment_id"],
            tasks=tasks,
            last_update=data.get("last_update", 0.0),
        )


# ======================================================
# STORE
# ======================================================

class WorkerProgressStore:

    def __init__(self, artifact_store: ArtifactStore, worker_id: str):
        self.store: ArtifactStore = artifact_store
        self.worker_id: str = worker_id
        self.state: Optional[WorkerProgressState] = None

    # ------------------------
    # Internal helpers
    # ------------------------
    def _to_worker_snapshot(
            self,
            job_id: str,
            experiment_id: str,
            task_id: str,
            task: TaskProgressSnapshot
    ) -> WorkerProgressSnapshot:
        return WorkerProgressSnapshot(
            worker_id=self.worker_id,
            task_id=task_id,
            experiment_id=experiment_id,
            completed_tree_ids=list(task.completed_tree_ids),
            running_tree_ids=[],  # opzionale (puoi estenderlo dopo)
            failed_tree_ids=list(task.failed_tree_ids),
        )

    def _get_snapshot_key(self, job_id: str, experiment_id: str) -> str:
        return worker_snapshot_path(job_id, experiment_id, self.worker_id)


    def load(self, job_id: str, experiment_id: str) -> None:
        self._load_snapshot(job_id, experiment_id)


    def save(self, job_id: str, experiment_id: str) -> None:
        self._persist_snapshot(job_id, experiment_id)


    # ------------------------
    # Queries (brevi e sicure)
    # ------------------------

    def get_task(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str
    ) -> Optional[TaskProgressSnapshot]:

        self._load_snapshot(job_id, experiment_id)

        if self.state is None:
            return None

        return self.state.tasks.get(task_id)


    def get_running_tasks(
        self,
        job_id: str,
        experiment_id: str
    ) -> Dict[str, TaskProgressSnapshot]:

        self._load_snapshot(job_id, experiment_id)

        if self.state is None:
            return {}

        return {
            tid: t for tid, t in self.state.tasks.items()
            if t.status == "RUNNING"
        }


    def get_completed_tree_ids(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str
    ) -> set[str]:

        task = self.get_task(job_id, experiment_id, task_id)

        if task is None:
            return set()

        return set(task.completed_tree_ids)

    # ------------------------
    # Internal helpers
    # ------------------------

    def _load_snapshot(self, job_id: str, experiment_id: str) -> None:
        key = self._get_snapshot_key(job_id, experiment_id)

        if self.store.exists(key):
            data = self.store.load_json(key)
            self.state = WorkerProgressState.from_dict(data)
            return

        self.state = WorkerProgressState(
            worker_id=self.worker_id,
            job_id=job_id,
            experiment_id=experiment_id,
            tasks={},
            last_update=current_time_seconds(),
        )

    def _persist_snapshot(self, job_id: str, experiment_id: str) -> None:
        key = self._get_snapshot_key(job_id, experiment_id)

        assert self.state is not None

        payload = {
            "worker_id": self.state.worker_id,
            "job_id": self.state.job_id,
            "experiment_id": self.state.experiment_id,
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "attempt_id": task.attempt_id,
                    "status": task.status,
                    "progress": task.progress,
                    "completed_tree_ids": list(task.completed_tree_ids),
                    "failed_tree_ids": list(task.failed_tree_ids),
                    "error_message": task.error_message,
                    "last_update": task.last_update,
                }
                for task_id, task in self.state.tasks.items()
            },
            "last_update": current_time_seconds(),
        }

        self.store.save_json_atomic(key, payload)

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

        assert self.state is not None

        existing = self.state.tasks.get(task_id)
        attempt_id = metadata.get("attempt_id")

        if existing is not None:
            # 🔴 Task già completato → idempotenza forte
            if existing.status == "COMPLETED":
                return "ALREADY_COMPLETED"

            # 🔴 Stesso attempt → retry
            if existing.attempt_id == attempt_id:
                return "RETRY"

            # 🔴 Nuovo attempt → overwrite controllato
            # (puoi in futuro loggare o gestire diversamente)

        # Nuovo task
        self.state.tasks[task_id] = TaskProgressSnapshot(
            task_id=task_id,
            attempt_id=attempt_id,
            status="RUNNING",
            progress=0.0,
            completed_tree_ids=[],
            failed_tree_ids=[],
            error_message=None,
            last_update=current_time_seconds(),
        )

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

        assert self.state is not None
        task = self.state.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        task.progress = progress
        task.last_update = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    def update_task(
        self,
        job_id: str,
        experiment_id: str,
        task_id: str,
        completed_tree_ids: List[str],
        failed_tree_ids: List[str],
    ) -> None:
        self._load_snapshot(job_id, experiment_id)

        assert self.state is not None
        task = self.state.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        task.completed_tree_ids = list(completed_tree_ids)
        task.failed_tree_ids = list(failed_tree_ids)
        task.last_update = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    def complete_task(self, job_id: str, experiment_id: str, task_id: str) -> None:
        self._load_snapshot(job_id, experiment_id)

        assert self.state is not None
        task = self.state.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        task.status = "COMPLETED"
        task.progress = 1.0
        task.last_update = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)

    def fail_task(self, job_id: str, experiment_id: str, task_id: str, error: str) -> None:
        self._load_snapshot(job_id, experiment_id)

        assert self.state is not None
        task = self.state.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")

        task.status = "FAILED"
        task.error_message = error
        task.last_update = current_time_seconds()

        self._persist_snapshot(job_id, experiment_id)