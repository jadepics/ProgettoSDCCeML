from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


class ScalabilityMetricsCollector:
    """
    Raccoglie metriche di scalabilità e comunicazione per un job.

    Responsabilità:
    - registrare eventi JSONL
    - misurare durate tramite timer
    - produrre summary finale
    - calcolare throughput, speedup ed efficienza quando possibile
    """

    def __init__(
        self,
        job_id: str,
        metrics_dir: Path,
        baseline_time_seconds: Optional[float] = None,
    ) -> None:
        self.job_id = job_id
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.metrics_dir / "scalability_events.jsonl"
        self.summary_path = self.metrics_dir / "scalability_summary.json"

        self.baseline_time_seconds = baseline_time_seconds
        self.started_at = time.time()

        self.counters: dict[str, int] = {
            "completed_tree_count": 0,
            "failed_tree_count": 0,
            "retry_count": 0,
            "train_rpc_count": 0,
            "predict_rpc_count": 0,
        }

        self.timings: dict[str, float] = {}

    def record_event(
        self,
        event: str,
        **payload: Any,
    ) -> None:
        row = {
            "timestamp": time.time(),
            "job_id": self.job_id,
            "event": event,
            **payload,
        }

        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, default=str) + "\n")

    @contextmanager
    def timer(
        self,
        name: str,
        **payload: Any,
    ):
        start = time.time()
        self.record_event(f"{name}_started", **payload)

        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[name] = self.timings.get(name, 0.0) + elapsed

            self.record_event(
                f"{name}_completed",
                elapsed_time_seconds=elapsed,
                **payload,
            )

    def record_training_plan(
        self,
        worker_count: int,
        shard_count: int,
        n_estimators_total: int,
    ) -> None:
        self.record_event(
            "training_plan_created",
            worker_count=worker_count,
            shard_count=shard_count,
            n_estimators_total=n_estimators_total,
        )

    def record_train_rpc(
        self,
        worker_id: str,
        task_id: str,
        tree_count: int,
        latency_seconds: float,
        request_bytes: Optional[int] = None,
        response_bytes: Optional[int] = None,
        success: bool = True,
    ) -> None:
        self.counters["train_rpc_count"] += 1

        self.record_event(
            "train_rpc_completed",
            worker_id=worker_id,
            task_id=task_id,
            tree_count=tree_count,
            latency_seconds=latency_seconds,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            success=success,
        )

    def record_shard_result(
        self,
        worker_id: str,
        task_id: str,
        completed_tree_count: int,
        failed_tree_count: int,
        elapsed_time_seconds: float,
    ) -> None:
        self.counters["completed_tree_count"] += completed_tree_count
        self.counters["failed_tree_count"] += failed_tree_count

        self.record_event(
            "shard_completed",
            worker_id=worker_id,
            task_id=task_id,
            completed_tree_count=completed_tree_count,
            failed_tree_count=failed_tree_count,
            elapsed_time_seconds=elapsed_time_seconds,
        )

    def record_retry(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
    ) -> None:
        self.counters["retry_count"] += 1

        self.record_event(
            "shard_retry",
            task_id=task_id,
            worker_id=worker_id,
            reason=reason,
        )

    def write_summary(
        self,
        worker_count: int,
        n_estimators_total: int,
    ) -> dict[str, Any]:
        total_time = time.time() - self.started_at
        training_time = self.timings.get("training", 0.0)

        completed_tree_count = self.counters["completed_tree_count"]

        throughput = None
        if training_time > 0:
            throughput = completed_tree_count / training_time

        speedup = None
        efficiency = None

        if self.baseline_time_seconds is not None and total_time > 0:
            speedup = self.baseline_time_seconds / total_time

            if worker_count > 0:
                efficiency = speedup / worker_count

        summary = {
            "job_id": self.job_id,
            "worker_count": worker_count,
            "n_estimators_total": n_estimators_total,
            "total_time_seconds": total_time,
            "training_time_seconds": training_time,
            "timings": self.timings,
            "counters": self.counters,
            "throughput_trees_per_second": throughput,
            "baseline_time_seconds": self.baseline_time_seconds,
            "speedup": speedup,
            "efficiency": efficiency,
        }

        with self.summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, default=str)

        return summary