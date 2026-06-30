import numpy as np
from typing import List

from worker.storage.artifact_store import ArtifactStore

"""
Modulo responsabile della predizione locale eseguita dal worker.

Il predictor carica gli alberi addestrati a partire dagli artifact salvati
su storage condiviso e calcola il contributo dello shard alla predizione
complessiva del modello distribuito.
"""

class ShardPredictor:
    """
    Esegue la predizione usando un insieme di alberi assegnati allo shard.

    La classe non conosce i dettagli fisici dello storage: usa ArtifactStore
    per caricare gli alberi e si limita a calcolare i risultati prodotti
    dagli alberi disponibili.
    """

    def __init__(self, artifact_store : ArtifactStore):
        self.artifact_store = artifact_store

    def predict(
            self,
            tree_artifact_uris: List[str],  # list[str]
            X: np.ndarray,  # np.ndarray
            task_type: str,  # str ("classification" | "regression")
            class_labels: List[str],  # list[str]
    ):
        """
        Carica gli alberi indicati dagli URI e calcola la predizione dello shard.

        Per la classificazione restituisce una matrice di voti con dimensione
        (n_samples, n_classes), dove ogni albero incrementa il conteggio della
        classe predetta.

        Per la regressione restituisce una matrice con dimensione
        (n_samples, 1), contenente la somma delle predizioni numeriche prodotte
        dagli alberi dello shard.

        L'aggregazione finale tra shard viene gestita dal master.
        """

        if not tree_artifact_uris:
            raise ValueError("No tree artifacts provided")

        # Ogni URI identifica un albero addestrato salvato come artifact.
        # Gli alberi vengono caricati prima di eseguire la predizione sullo shard.
        trees = [
            self.artifact_store.load_tree_artifact(uri)
            for uri in tree_artifact_uris
        ]

        n_samples = X.shape[0]

        # ----------------------------------------
        # CLASSIFICAZIONE
        # ----------------------------------------
        if task_type == "classification":
            n_classes = len(class_labels)

            if n_classes == 0:
                raise ValueError("class_labels must be non-empty for classification")

            label_to_index = {
                str(label): i
                for i, label in enumerate(class_labels)
            }

            # La matrice dei voti contiene, per ogni campione, il numero di alberi
            # che hanno predetto ciascuna classe.
            votes = np.zeros((n_samples, n_classes), dtype=np.float64)

            # Ogni albero vota una classe per ciascun campione.
            for tree in trees:
                preds = tree.predict(X)

                for i, pred in enumerate(preds):
                    pred_key = str(pred)

                    if pred_key not in label_to_index:
                        raise ValueError(
                            f"Predicted label '{pred}' not found in class_labels "
                            f"{class_labels}"
                        )

                    class_idx = label_to_index[pred_key]
                    votes[i, class_idx] += 1.0

            return votes

        # ----------------------------------------
        # REGRESSIONE
        # ----------------------------------------
        elif task_type == "regression":

            # Per la regressione si accumula la somma delle predizioni numeriche
            # prodotte dagli alberi dello shard.
            sums = np.zeros((n_samples, 1), dtype=np.float64)

            for tree in trees:
                preds = tree.predict(X)  # Dimensione attesa: (n_samples,)
                sums[:, 0] += preds

            return sums

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")