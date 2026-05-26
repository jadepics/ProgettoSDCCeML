from __future__ import annotations

import os
import signal
import time


def maybe_fail(point: str) -> None:
    target = os.getenv("CHAOS_POINT", "").strip()
    if target != point:
        return

    action = os.getenv("CHAOS_ACTION", "crash").strip().lower()
    delay = float(os.getenv("CHAOS_DELAY_SECONDS", "0") or 0)

    if delay > 0:
        time.sleep(delay)

    if action == "crash":
        os._exit(137)

    if action == "freeze":
        os.kill(os.getpid(), signal.SIGSTOP)

    if action == "sleep":
        time.sleep(float(os.getenv("CHAOS_SLEEP_SECONDS", "999999")))
        return

    raise ValueError(f"Unknown CHAOS_ACTION={action}")