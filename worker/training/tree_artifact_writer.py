from __future__ import annotations


from common.ids import generate_tree_id
from common.contracts import TreeArtifactMetadata
from common.enums import TreeStatus
from worker.storage.artifact_store import ArtifactStore
from worker.storage.paths import (
    tree_artifact_path,
    tree_metadata_path,
)


class TreeArtifactWriter:
    """
    Responsabile del salvataggio degli alberi (artifact) e dei relativi metadata.

    Garantisce:
    - path deterministici
    - idempotenza storage
    - consistenza tra artifact e metadata
    """

    def __init__(self, artifact_store: ArtifactStore, worker_id: str):
        self.store: ArtifactStore = artifact_store
        self.worker_id: str = worker_id

    def write_tree(
        self,
        model: object,
        job_id: str,
        experiment_id: str,
        task_id: str,
        tree_index: int,
        seed: int,
        training_time_seconds: float,
    ) -> TreeArtifactMetadata:

        # --------------------------------------------------
        # Path deterministici
        # --------------------------------------------------
        artifact_key = tree_artifact_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )
        metadata_key = tree_metadata_path(
            job_id=job_id,
            experiment_id=experiment_id,
            tree_index=tree_index,
        )

        # pulizia hardening: eventuali file temporanei rimasti da crash precedenti
        self._cleanup_tmp_paths(artifact_key, metadata_key)

        tree_id = generate_tree_id(experiment_id, tree_index)

        # --------------------------------------------------
        # FAST PATH → metadata già esiste
        # (più veloce + evita load modello inutile)
        # --------------------------------------------------
        if self.store.exists(metadata_key):
            data = self.store.load_json(metadata_key)
            metadata = TreeArtifactMetadata.from_dict(data)
            if self.store.exists(metadata.artifact_uri):
                return metadata

            # metadata senza artifact → stato incoerente, lo rimuovo e ricostruisco
            self.store.delete(metadata_key)

        # --------------------------------------------------
        #  Scrittura artifact (idempotente)
        # --------------------------------------------------
        self.store.save_tree_artifact_if_not_exists(artifact_key, model)

        # --------------------------------------------------
        #  Feature importances dal modello addestrato
        # --------------------------------------------------
        feature_importances = self._extract_feature_importances(model)

        # --------------------------------------------------
        #  COSTRUZIONE METADATA (sempre)
        # --------------------------------------------------
        metadata = TreeArtifactMetadata(
            tree_id=tree_id,
            job_id=job_id,
            experiment_id=experiment_id,
            task_id=task_id,
            tree_index=tree_index,
            worker_id=self.worker_id,
            seed=seed,
            artifact_uri=artifact_key,
            status=TreeStatus.COMPLETED,
            training_time_seconds=training_time_seconds,
            feature_importances=feature_importances,
        )

        # --------------------------------------------------
        #  Scrittura metadata ATOMICA (sempre)
        # --------------------------------------------------
        self.store.save_json_atomic(metadata_key, metadata.to_dict())
        return metadata

    def _cleanup_tmp_paths(self, artifact_key: str, metadata_key: str) -> None:
        for key in (artifact_key + ".tmp", metadata_key + ".tmp"):
            if self.store.exists(key):
                self.store.delete(key)

    @staticmethod
    def _extract_feature_importances(model: object) -> list[float]:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []

        if hasattr(importances, "tolist"):
            importances = importances.tolist()

        return [float(value) for value in importances]