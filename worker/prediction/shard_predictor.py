from __future__ import annotations

from typing import List

import numpy as np

from worker.storage.artifact_store import ArtifactStore


class ShardPredictor:
    """
    Handles prediction for a shard of trees.

    Responsibilities:
    - Load trained trees from artifact store
    - Run predictions
    - Aggregate results
    """

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def predict(
        self,
        experiment_id: str,
        tree_ids: List[str],
        X: np.ndarray,
    ) -> List[float]:

        predictions_per_tree = []

        for tree_id in tree_ids:
            artifact_key = self._build_artifact_key(experiment_id, tree_id)

            # Load model
            model = self.artifact_store.load_tree_artifact(artifact_key)

            # Predict
            preds = model.predict(X)
            predictions_per_tree.append(preds)

        # Aggregate (majority voting or mean depending on task)
        return self._aggregate(predictions_per_tree)

    def _build_artifact_key(self, experiment_id: str, tree_id: str) -> str:
        """
        Reconstructs the artifact key used during training.
        Must match TreeArtifactWriter / paths.py logic.
        """
        return f"jobs/{experiment_id}/trees/{tree_id}.joblib"

    def _aggregate(self, predictions_per_tree: List[np.ndarray]) -> List[float]:
        """
        Aggregation strategy:
        - classification → majority vote
        - regression → mean
        """

        if not predictions_per_tree:
            return []

        # Stack predictions: shape (n_trees, n_samples)
        stacked = np.vstack(predictions_per_tree)

        # Majority vote (classification-like)
        # You may adapt this depending on your task type
        aggregated = []

        for i in range(stacked.shape[1]):
            column = stacked[:, i]
            values, counts = np.unique(column, return_counts=True)
            aggregated.append(values[np.argmax(counts)])

        return aggregated