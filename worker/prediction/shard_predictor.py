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
        artifact_uris: List[str],
        X: np.ndarray,
    ) -> np.ndarray:

        if not artifact_uris:
            raise ValueError("No artifact URIs provided for prediction")

        predictions_per_tree = []

        # --------------------------------------------------
        # Load + predict per tree
        # --------------------------------------------------
        for uri in artifact_uris:

            if not self.artifact_store.tree_artifact_exists(uri):
                raise FileNotFoundError(f"Artifact not found: {uri}")

            model = self.artifact_store.load_tree_artifact(uri)

            preds = model.predict(X)
            predictions_per_tree.append(preds)

        # --------------------------------------------------
        # Aggregate predictions
        # --------------------------------------------------
        return self._aggregate(predictions_per_tree)

    def _aggregate(self, predictions_per_tree: List[np.ndarray]) -> np.ndarray:
        """
        Majority voting aggregation for classification.

        Shape:
            predictions_per_tree: (n_trees, n_samples)
        Returns:
            (n_samples,)
        """

        stacked = np.vstack(predictions_per_tree)  # (n_trees, n_samples)

        n_samples = stacked.shape[1]
        final_predictions = []

        for i in range(n_samples):
            column = stacked[:, i]

            values, counts = np.unique(column, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)