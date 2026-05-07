import numpy as np
from typing import List

from worker.storage.artifact_store import ArtifactStore


class ShardPredictor:

    def __init__(self, artifact_store : ArtifactStore):
        self.artifact_store = artifact_store

    def predict(
            self,
            tree_artifact_uris: List[str],  # list[str]
            X: np.ndarray,  # np.ndarray
            task_type: str,  # str ("classification" | "regression")
            class_labels: List[str],  # list[str]
    ):

        if not tree_artifact_uris:
            raise ValueError("No tree artifacts provided")

        trees = [
            self.artifact_store.load_tree_artifact(uri)
            for uri in tree_artifact_uris
        ]

        n_samples = X.shape[0]

        # ----------------------------------------
        # CLASSIFICATION
        # ----------------------------------------
        if task_type == "classification":
            n_classes = len(class_labels)

            # mapping label → index
            label_to_index = {
                label: i for i, label in enumerate(class_labels)
            }

            votes = np.zeros((n_samples, n_classes), dtype=np.float64)

            for tree in trees:
                preds = tree.predict(X)  # shape: (n_samples,)

                for i, pred in enumerate(preds):
                    class_idx = label_to_index[pred]
                    votes[i, class_idx] += 1.0

            return votes

        # ----------------------------------------
        # REGRESSION
        # ----------------------------------------
        elif task_type == "regression":
            sums = np.zeros((n_samples, 1), dtype=np.float64)

            for tree in trees:
                preds = tree.predict(X)  # shape: (n_samples,)
                sums[:, 0] += preds

            return sums

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")