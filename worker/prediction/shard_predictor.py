from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from worker.storage.artifact_store import ArtifactStore


@dataclass
class ShardPredictionResult:
    values: np.ndarray
    n_rows: int
    n_cols: int


class ShardPredictor:

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def predict(self, artifact_uris: List[str], X: np.ndarray) -> ShardPredictionResult:

        if not artifact_uris:
            raise ValueError("No artifact URIs provided")

        predictions = []

        for uri in artifact_uris:
            if not self.artifact_store.tree_artifact_exists(uri):
                raise FileNotFoundError(uri)

            model = self.artifact_store.load_tree_artifact(uri)
            predictions.append(model.predict(X))

        final = self._aggregate(predictions)

        return ShardPredictionResult(
            values=final,
            n_rows=final.shape[0],
            n_cols=1
        )

    def _aggregate(self, preds: List[np.ndarray]) -> np.ndarray:
        stacked = np.vstack(preds)

        n_samples = stacked.shape[1]
        out = np.empty(n_samples)

        for i in range(n_samples):
            col = stacked[:, i]
            values, counts = np.unique(col, return_counts=True)
            out[i] = values[np.argmax(counts)]

        return out