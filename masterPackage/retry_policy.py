from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """
    Politica di retry lato master per gli shard di training.

    max_attempts_per_task:
    - numero totale massimo di attempt per lo stesso task logico
    - include anche il primo attempt
    - esempio: 2 = attempt iniziale + 1 retry

    base_backoff_seconds:
    - backoff lineare semplice
    - attempt 1 -> 1 * base_backoff_seconds
    - attempt 2 -> 2 * base_backoff_seconds
    """

    max_attempts_per_task: int = 2
    base_backoff_seconds: float = 0.0
    retry_on_timeout: bool = True
    retry_on_worker_failure: bool = True
    retry_on_unknown_error: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts_per_task <= 0:
            raise ValueError("max_attempts_per_task must be > 0")
        if self.base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be >= 0")

    def should_retry(
        self,
        attempt_id: int,
        error_message: str | None,
    ) -> bool:
        if attempt_id >= self.max_attempts_per_task:
            return False

        category = self._classify_error(error_message)

        if category == "timeout":
            return self.retry_on_timeout

        if category == "worker_failure":
            return self.retry_on_worker_failure

        return self.retry_on_unknown_error

    def backoff_seconds_for(
        self,
        attempt_id: int,
    ) -> float:
        if self.base_backoff_seconds <= 0:
            return 0.0
        return float(attempt_id) * self.base_backoff_seconds

    def _classify_error(
        self,
        error_message: str | None,
    ) -> str:
        if not error_message:
            return "unknown"

        message = error_message.strip().lower()

        timeout_markers = [
            "timeout",
            "deadline exceeded",
            "timed out",
        ]
        if any(marker in message for marker in timeout_markers):
            return "timeout"

        worker_failure_markers = [
            "unavailable",
            "connection refused",
            "connection reset",
            "broken pipe",
            "transport closed",
            "worker",
            "rpc",
        ]
        if any(marker in message for marker in worker_failure_markers):
            return "worker_failure"

        return "unknown"