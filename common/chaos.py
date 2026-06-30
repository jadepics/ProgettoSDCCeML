from __future__ import annotations

import os
import signal
import time

"""
file comune che inietta un fault artificiale se il punto di esecuzione richiesto
coincide con quello configurato via environment.

Variabili d'ambiente usate:
- CHAOS_POINT: nome del punto in cui attivare il fault
- CHAOS_ACTION: tipo di fault da simulare ("crash", "freeze", "sleep")
- CHAOS_DELAY_SECONDS: ritardo opzionale prima di attivare il fault
- CHAOS_SLEEP_SECONDS: durata del fault se l'azione è "sleep"

Uso tipico:
il chiamante inserisce maybe_fail("nome_punto") in punti strategici
del workflow per simulare crash o blocchi controllati durante i test
di fault tolerance.
"""
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