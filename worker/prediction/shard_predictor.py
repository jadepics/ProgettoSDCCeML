from __future__ import annotations

from typing import List
import numpy as np

from worker.storage.artifact_store import ArtifactStore


class ShardPredictionResult:
    def __init__(self, values: np.ndarray):
        """
        values: np.ndarray shape (n_samples, n_trees)
        """
        self.values = values
        self.n_rows = values.shape[0]
        self.n_cols = values.shape[1]


class ShardPredictor:
    """
    Prediction per shard (insieme di alberi).
    NON aggrega: restituisce predizioni per albero.
    """

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def predict(
        self,
        tree_artifact_uris: List[str],
        X: np.ndarray,
    ) -> ShardPredictionResult:
        """
        tree_artifact_uris: List[str]  # URIs forniti dal master
        X: np.ndarray                  # shape (n_samples, n_features)

        return: ShardPredictionResult
        """

        if not tree_artifact_uris:
            raise ValueError("No tree artifacts provided")

        n_samples: int = X.shape[0]
        n_trees: int = len(tree_artifact_uris)

        # matrice output: (samples x trees)
        predictions = np.zeros((n_samples, n_trees))

        # ----------------------------------------
        # Loop sugli alberi assegnati
        # ----------------------------------------
        for j, uri in enumerate(tree_artifact_uris):
            """
            uri: str
            """

            if not self.artifact_store.tree_artifact_exists(uri):
                raise FileNotFoundError(uri)

            # ----------------------------------------
            # LOAD MODEL (NO ASSUNZIONI)
            # ----------------------------------------
            tree = self.artifact_store.load_tree_artifact(uri)

            # ----------------------------------------
            # PREDICT
            # ----------------------------------------
            pred = tree.predict(X)

            # sicurezza shape
            pred = np.asarray(pred).reshape(-1)

            if pred.shape[0] != n_samples:
                raise ValueError(
                    f"Tree prediction size mismatch: expected {n_samples}, got {pred.shape[0]}"
                )

            predictions[:, j] = pred

        return ShardPredictionResult(predictions)