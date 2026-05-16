from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from common.contracts import PreparedDataset, DatasetPreparationMetadata
from common.repositories import SharedArtifactStore
from masterPackage.data.dataset_loader import DatasetLoader
from masterPackage.data.dataset_validator import DatasetValidator
from masterPackage.data.split_manager import SplitManager


class DataPreparationService:
    """
    Responsabilità:
    - caricare il dataset sorgente
    - applicare eventuali scenari controllati sul dataset
    - validarlo
    - creare gli split train/validation/test
    - persistere schema, report scenario e split su storage condiviso
    - restituire il PreparedDataset finale

    Nota:
    la rimozione delle feature sospette di leakage deve avvenire prima
    dell'encoding categorico, altrimenti colonne come diabetes_stage vengono
    trasformate in dummy column e diventano più difficili da tracciare.
    """

    DEFAULT_LEAKAGE_COLUMNS_BY_TARGET: dict[str, list[str]] = {
        "diagnosed_diabetes": [
            "diabetes_stage",
        ],
    }

    SUPPORTED_DATASET_SCENARIOS = {
        "baseline_original",
        "baseline_no_leakage",
    }

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
        dataset_scenario: str = "baseline_original",
        leakage_columns: list[str] | None = None,
    ) -> PreparedDataset:
        """
        Esegue l'intera pipeline dati del master e restituisce il PreparedDataset.

        Flusso:
        1. load del dataset sorgente
        2. applicazione scenario dataset
        3. encoding delle feature categoriche
        4. validazione e costruzione DatasetSchema
        5. split deterministico
        6. persistenza schema + scenario report + split
        7. costruzione PreparedDataset
        """
        df = self.dataset_loader.load(dataset_uri)

        df, scenario_report = self._apply_dataset_scenario(
            df=df,
            dataset_uri=dataset_uri,
            target_column=target_column,
            dataset_scenario=dataset_scenario,
            leakage_columns=leakage_columns,
        )

        df = self._encode_categorical_features(df, target_column)

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
        self._persist_dataset_scenario_report(job_id, scenario_report)

        scenario_report_uri = self._persist_dataset_scenario_report(
            job_id,
            scenario_report,
        )

        preparation_metadata = DatasetPreparationMetadata(
            dataset_scenario=scenario_report["dataset_scenario"],
            dropped_columns=scenario_report["dropped_columns"],
            requested_leakage_columns=scenario_report["requested_leakage_columns"],
            missing_requested_leakage_columns=scenario_report["missing_requested_leakage_columns"],
            original_column_count=scenario_report["original_column_count"],
            final_column_count=scenario_report["final_column_count"],
            original_row_count=scenario_report["original_row_count"],
            final_row_count=scenario_report["final_row_count"],
            scenario_report_uri=scenario_report_uri,
        )
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
            preparation_metadata=preparation_metadata,
        )

        return prepared_dataset

    def _apply_dataset_scenario(
        self,
        df: pd.DataFrame,
        dataset_uri: str,
        target_column: str,
        dataset_scenario: str,
        leakage_columns: list[str] | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        scenario = self._normalize_dataset_scenario(dataset_scenario)

        original_columns = list(df.columns)
        original_row_count = len(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        dropped_columns: list[str] = []

        if scenario == "baseline_no_leakage":
            candidate_columns = self._resolve_leakage_columns(
                target_column=target_column,
                leakage_columns=leakage_columns,
            )

            dropped_columns = [
                column
                for column in candidate_columns
                if column in df.columns and column != target_column
            ]

            if dropped_columns:
                df = df.drop(columns=dropped_columns)

        report = {
            "job_dataset_uri": dataset_uri,
            "dataset_scenario": scenario,
            "target_column": target_column,
            "original_row_count": original_row_count,
            "final_row_count": len(df),
            "original_column_count": len(original_columns),
            "final_column_count": len(df.columns),
            "original_columns": original_columns,
            "final_columns": list(df.columns),
            "requested_leakage_columns": leakage_columns,
            "dropped_columns": dropped_columns,
            "missing_requested_leakage_columns": self._missing_requested_columns(
                df_columns_before=original_columns,
                requested_columns=leakage_columns,
            ),
        }

        return df, report

    def _normalize_dataset_scenario(self, dataset_scenario: str) -> str:
        if dataset_scenario is None or not dataset_scenario.strip():
            return "baseline_original"

        scenario = dataset_scenario.strip().lower()

        if scenario not in self.SUPPORTED_DATASET_SCENARIOS:
            raise ValueError(
                f"Unsupported dataset_scenario '{dataset_scenario}'. "
                f"Supported values: {sorted(self.SUPPORTED_DATASET_SCENARIOS)}"
            )

        return scenario

    def _resolve_leakage_columns(
        self,
        target_column: str,
        leakage_columns: list[str] | None,
    ) -> list[str]:
        if leakage_columns is not None:
            return list(dict.fromkeys(leakage_columns))

        return self.DEFAULT_LEAKAGE_COLUMNS_BY_TARGET.get(target_column, [])

    def _missing_requested_columns(
        self,
        df_columns_before: list[str],
        requested_columns: list[str] | None,
    ) -> list[str]:
        if requested_columns is None:
            return []

        existing = set(df_columns_before)
        return [
            column
            for column in requested_columns
            if column not in existing
        ]

    def _persist_schema(self, job_id: str, schema_payload: dict) -> None:
        path = self.artifact_store.layout.dataset_schema_path(job_id)
        self.artifact_store.write_json(path, schema_payload)

    def _persist_dataset_scenario_report(
        self,
        job_id: str,
        scenario_report: dict[str, Any],
    ) -> None:
        schema_path = self.artifact_store.layout.dataset_schema_path(job_id)
        report_path = schema_path.parent / "dataset_scenario_report.json"
        self.artifact_store.write_json(report_path, scenario_report)
        return self._to_file_uri(report_path)

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

    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> pd.DataFrame:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        features = df.drop(columns=[target_column])
        target = df[target_column]

        encoded_features = pd.get_dummies(
            features,
            dummy_na=True,
            dtype=float,
        )

        result = encoded_features.copy()
        result[target_column] = target.to_numpy()
        return result