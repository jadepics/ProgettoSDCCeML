from __future__ import annotations

from common.contracts import (
    ForestConfiguration,
    ModelManifest,
    PreparedDataset,
    TreeArtifactMetadata,
    ValidationMetrics,
)
from common.enums import ModelStatus, TreeStatus


class ModelManifestBuilder:
    """
    Responsabilità:
    - costruire il manifest finale del modello selezionato
    - verificare che gli artifact appartengano tutti allo stesso esperimento
    - includere solo alberi COMPLETED
    - ordinare deterministicamente gli artifact per tree_index

    Nota:
    questa classe costruisce il manifest, ma non lo salva.
    Il salvataggio resta responsabilità del ModelRepository.
    """

    def build(
        self,
        model_id: str,
        job_id: str,
        experiment_id: str,
        model_type: str,
        forest_config: ForestConfiguration,
        prepared_dataset: PreparedDataset,
        tree_artifacts: list[TreeArtifactMetadata],
        validation_metrics: ValidationMetrics,
        test_metrics: dict | None = None,
        status: ModelStatus = ModelStatus.READY,
    ) -> ModelManifest:
        if model_type not in {"classification", "regression"}:
            raise ValueError("model_type must be 'classification' or 'regression'")

        if forest_config.experiment_id != experiment_id:
            raise ValueError(
                "forest_config.experiment_id does not match manifest experiment_id"
            )

        completed_artifacts = [
            artifact
            for artifact in tree_artifacts
            if artifact.status == TreeStatus.COMPLETED
        ]

        if not completed_artifacts:
            raise ValueError("Cannot build ModelManifest without completed tree artifacts")

        for artifact in completed_artifacts:
            if artifact.job_id != job_id:
                raise ValueError(
                    f"Tree artifact '{artifact.tree_id}' does not belong to job '{job_id}'"
                )
            if artifact.experiment_id != experiment_id:
                raise ValueError(
                    f"Tree artifact '{artifact.tree_id}' does not belong to experiment '{experiment_id}'"
                )

        completed_artifacts = sorted(
            completed_artifacts,
            key=lambda item: item.tree_index,
        )

        class_labels = prepared_dataset.class_labels or []

        return ModelManifest(
            model_id=model_id,
            job_id=job_id,
            experiment_id=experiment_id,
            model_type=model_type,
            forest_config=forest_config,
            class_labels=class_labels,
            feature_names=prepared_dataset.schema.feature_names,
            target_column=prepared_dataset.schema.target_column,
            train_features_uri=prepared_dataset.train_features_uri,
            train_labels_uri=prepared_dataset.train_labels_uri,
            validation_features_uri=prepared_dataset.validation_features_uri,
            validation_labels_uri=prepared_dataset.validation_labels_uri,
            test_features_uri=prepared_dataset.test_features_uri,
            test_labels_uri=prepared_dataset.test_labels_uri,
            tree_artifacts=completed_artifacts,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            preparation_metadata=prepared_dataset.preparation_metadata,
            status=status,
        )