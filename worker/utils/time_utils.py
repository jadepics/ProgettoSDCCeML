import time
from contextlib import contextmanager


def current_time_seconds() -> float:
    """
restituisce il time corrente in secondi
    """
    return time.time()

def now_ts() -> float:
    return time.time()

@contextmanager
def timer():
    """
    Context manager per misurare il tempo di esecuzione
    """
    start = time.time()

    class TimerResult:
        duration = None

    result = TimerResult()

    yield result

    end = time.time()
    result.duration = end - start