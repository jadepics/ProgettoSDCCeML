from __future__ import annotations

import numpy as np


class BootstrapSampler:
    """
    Responsabile della generazione del bootstrap sample.

    Garantisce:
    - comportamento deterministico (seed-based)
    - compatibilità con retry (idempotenza)
    """

    class BootstrapSampler:
        def sample_indices(self, n_samples: int, seed: int, bootstrap: bool) -> np.ndarray:
            if not bootstrap:
                return np.arange(n_samples)

            rng = np.random.default_rng(seed)
            return rng.integers(0, n_samples, size=n_samples)