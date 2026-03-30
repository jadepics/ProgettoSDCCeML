import time
from contextlib import contextmanager


def current_time_seconds() -> float:
    """
    Returns current time in seconds (monotonic-safe for measurements if needed).
    """
    return time.time()

@contextmanager
def timer():
    """
    Context manager to measure execution time.

    Usage:
        with timer() as t:
            # do work
        print(t.duration)
    """
    start = time.time()

    class TimerResult:
        duration = None

    result = TimerResult()

    yield result

    end = time.time()
    result.duration = end - start