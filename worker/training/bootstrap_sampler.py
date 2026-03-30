from __future__ import annotations

import numpy as np


class BootstrapSampler:
    """
    Responsabile della generazione del bootstrap sample.

    Garantisce:
    - comportamento deterministico (seed-based)
    - compatibilità con retry (idempotenza)
    """

    def __init__(self, bootstrap: bool):
        self.bootstrap = bootstrap

    def sample_indices(self, n_samples: int, seed: int) -> np.ndarray:
        """
        Restituisce gli indici da usare per il training.

        Se bootstrap = True:
            sampling con replacement

        Se bootstrap = False:
            restituisce indici sequenziali
        """
        if not self.bootstrap:
            return np.arange(n_samples)

        rng = np.random.default_rng(seed)
        return rng.integers(0, n_samples, size=n_samples)