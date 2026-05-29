from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol


class WorkerInfoLike(Protocol):
    worker_id: str
    host: str
    port: int
    last_heartbeat: float
    running_tasks: int


class WorkerRegistryLike(Protocol):
    def list_workers(self) -> list[WorkerInfoLike]:
        ...


@dataclass(frozen=True)
class WorkerHeartbeatSnapshot:
    worker_id: str
    host: str
    port: int
    last_heartbeat: float
    age_seconds: float
    running_tasks: int
    is_stale: bool
    last_progress_age_seconds: float = 0.0
    is_zombie: bool = False


class WorkerHeartbeatMonitor:
    """
    Responsabilità:
    - osservare i heartbeat dei worker registrati
    - distinguere worker vivi e worker stale/dead
    - produrre snapshot utili a orchestrator e recovery planner

    Nota:
    questa prima versione è on-demand:
    - nessun thread interno
    - nessuna mutazione automatica del registry
    - il timeout è deciso lato master
    """

    def __init__(
        self,
        worker_registry: WorkerRegistryLike,
        heartbeat_timeout_seconds: float = 30.0,
        task_progress_timeout_seconds: float = 120.0,

    ) -> None:
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be > 0")
        if task_progress_timeout_seconds <= 0:
            raise ValueError("task_progress_timeout_seconds must be > 0")

        self.worker_registry = worker_registry
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.task_progress_timeout_seconds = task_progress_timeout_seconds

    def snapshot(
        self,
        now_ts: Optional[float] = None,
    ) -> list[WorkerHeartbeatSnapshot]:
        effective_now = time.time() if now_ts is None else now_ts
        result: list[WorkerHeartbeatSnapshot] = []

        for worker in self.worker_registry.list_workers():
            age_seconds = max(0.0, effective_now - float(worker.last_heartbeat))
            is_stale = age_seconds > self.heartbeat_timeout_seconds

            active_task_last_progress_ts = getattr(
                worker,
                "active_task_last_progress_ts",
                {},
            ) or {}

            if active_task_last_progress_ts:
                last_progress_ts = max(active_task_last_progress_ts.values())
                last_progress_age_seconds = max(0.0, effective_now - float(last_progress_ts))
            else:
                last_progress_age_seconds = age_seconds

            is_zombie = (
                not is_stale
                and int(worker.running_tasks) > 0
                and last_progress_age_seconds > self.task_progress_timeout_seconds
            )

            result.append(
                WorkerHeartbeatSnapshot(
                    worker_id=worker.worker_id,
                    host=worker.host,
                    port=worker.port,
                    last_heartbeat=float(worker.last_heartbeat),
                    age_seconds=age_seconds,
                    running_tasks=int(worker.running_tasks),
                    is_stale=is_stale,
                    last_progress_age_seconds=last_progress_age_seconds,
                    is_zombie=is_zombie,
                )
            )

        result.sort(key=lambda item: (item.is_stale, item.age_seconds, item.worker_id))
        return result

    def alive_workers(
        self,
        now_ts: Optional[float] = None,
    ) -> list[WorkerHeartbeatSnapshot]:
        return [
            item
            for item in self.snapshot(now_ts=now_ts)
            if not item.is_stale
        ]

    def stale_workers(
        self,
        now_ts: Optional[float] = None,
    ) -> list[WorkerHeartbeatSnapshot]:
        return [
            item
            for item in self.snapshot(now_ts=now_ts)
            if item.is_stale
        ]

    def zombie_workers(
            self,
            now_ts: Optional[float] = None
    ) -> list[WorkerHeartbeatSnapshot]:
        return [
            item
            for item in self.snapshot(now_ts=now_ts)
            if item.is_zombie]

    def stale_worker_ids(
        self,
        now_ts: Optional[float] = None,
    ) -> list[str]:
        return [
            item.worker_id
            for item in self.stale_workers(now_ts=now_ts)]

    def zombie_worker_ids(
            self,
            now_ts: Optional[float] = None
    ) -> list[str]:
        return [
            item.worker_id
            for item in self.zombie_workers(now_ts=now_ts)]

    def is_worker_stale(
        self,
        worker: WorkerInfoLike,
        now_ts: Optional[float] = None,
    ) -> bool:
        effective_now = time.time() if now_ts is None else now_ts
        age_seconds = max(0.0, effective_now - float(worker.last_heartbeat))
        return age_seconds > self.heartbeat_timeout_seconds