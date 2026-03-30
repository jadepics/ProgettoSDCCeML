from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.contracts import PreparedDataset
from common.repositories import SharedArtifactStore
from masterPackage.data.dataset_loader import DatasetLoader
from masterPackage.data.dataset_validator import DatasetValidator
from masterPackage.data.split_manager import SplitManager


class DataPreparationService:
    """
    Responsabilità:
    - caricare il dataset sorgente
    - validarlo
    - creare gli split train/validation/test
    - persistere schema e split su storage condiviso
    - restituire il PreparedDataset finale

    Nota:
    in questa milestone non applica ancora una pipeline di preprocessing.
    Quando introdurrai PreprocessingPipelineBuilder, il punto giusto in cui
    inserirlo sarà tra validate(...) e split(...).
    """

    def __init__(
        self,
        dataset_loader: DatasetLoader,
        dataset_validator: DatasetValidator,
        split_manager: SplitManager,
        artifact_store: SharedArtifactStore,
    ) -> None:
        self.dataset_loader = dataset_loader
        self.dataset_validator = dataset_validator
        self.split_manager = split_manager
        self.artifact_store = artifact_store

    def prepare(
        self,
        job_id: str,
        dataset_uri: str,
        target_column: str,
        task_type: str,
        validation_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> PreparedDataset:
        """
        Esegue l'intera pipeline dati del master e restituisce il PreparedDataset.

        Flusso:
        1. load del dataset sorgente
        2. validazione e costruzione DatasetSchema
        3. split deterministico
        4. persistenza schema + split
        5. costruzione PreparedDataset
        """
        df = self.dataset_loader.load(dataset_uri)

        schema = self.dataset_validator.validate(
            df=df,
            dataset_uri=dataset_uri,
            target_column=target_column,
            task_type=task_type,
        )

        splits = self.split_manager.split(
            df=df,
            target_column=target_column,
            task_type=schema.task_type,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        self._persist_schema(job_id, schema.to_dict())
        uris = self._persist_splits(job_id, target_column, splits)

        prepared_dataset = PreparedDataset(
            dataset_id=f"{job_id}_prepared_dataset",
            schema=schema,
            train_features_uri=uris["train_features_uri"],
            train_labels_uri=uris["train_labels_uri"],
            validation_features_uri=uris["validation_features_uri"],
            validation_labels_uri=uris["validation_labels_uri"],
            test_features_uri=uris["test_features_uri"],
            test_labels_uri=uris["test_labels_uri"],
            class_labels=splits.class_labels,
            n_features=len(schema.feature_names),
            n_train=len(splits.train_features),
            n_validation=len(splits.validation_features),
            n_test=len(splits.test_features),
        )

        return prepared_dataset

    def _persist_schema(self, job_id: str, schema_payload: dict) -> None:
        path = self.artifact_store.layout.dataset_schema_path(job_id)
        self.artifact_store.write_json(path, schema_payload)

    def _persist_splits(
        self,
        job_id: str,
        target_column: str,
        splits,
    ) -> dict[str, str]:
        layout = self.artifact_store.layout

        train_features_path = layout.train_features_path(job_id)
        train_labels_path = layout.train_labels_path(job_id)
        validation_features_path = layout.validation_features_path(job_id)
        validation_labels_path = layout.validation_labels_path(job_id)
        test_features_path = layout.test_features_path(job_id)
        test_labels_path = layout.test_labels_path(job_id)

        self._write_dataframe_parquet_atomic(train_features_path, splits.train_features)
        self._write_series_parquet_atomic(train_labels_path, splits.train_labels, target_column)

        self._write_dataframe_parquet_atomic(
            validation_features_path,
            splits.validation_features,
        )
        self._write_series_parquet_atomic(
            validation_labels_path,
            splits.validation_labels,
            target_column,
        )

        self._write_dataframe_parquet_atomic(test_features_path, splits.test_features)
        self._write_series_parquet_atomic(test_labels_path, splits.test_labels, target_column)

        return {
            "train_features_uri": self._to_file_uri(train_features_path),
            "train_labels_uri": self._to_file_uri(train_labels_path),
            "validation_features_uri": self._to_file_uri(validation_features_path),
            "validation_labels_uri": self._to_file_uri(validation_labels_path),
            "test_features_uri": self._to_file_uri(test_features_path),
            "test_labels_uri": self._to_file_uri(test_labels_path),
        }

    def _write_dataframe_parquet_atomic(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)

    def _write_series_parquet_atomic(
        self,
        path: Path,
        series: pd.Series,
        column_name: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        series.to_frame(name=column_name).to_parquet(temp_path, index=False)
        temp_path.replace(path)

    def _to_file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()