from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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
    gli scenari vengono applicati prima dell'encoding categorico. All interno di contracts
    abbiamo definito il keep ed il drop per le feature, gli scenari sono le diverse feature
    che abbiamo scelto di togliere per avere diverse visioni del modello come lavora e vedere
    come le metriche variano al variare della feature selection.
    """

    DEFAULT_LEAKAGE_COLUMNS_BY_TARGET: dict[str, list[str]] = {
        "diagnosed_diabetes": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "diabetes_stage": [
            "diagnosed_diabetes",
            "diabetes_risk_score",
        ],
    }

    DIAGNOSTIC_MARKER_COLUMNS: list[str] = [
        "hba1c",
        "glucose_fasting",
        "glucose_postprandial",
        "insulin_level",
    ]

    DIAGNOSTIC_NOISE_STD_FRACTION_BY_SCENARIO: dict[str, float] = {
        "diagnostic_noise_10pct": 0.10,
        "diagnostic_noise_25pct": 0.25,
        "diagnostic_noise_50pct": 0.50,
    }

    BINARY_IMBALANCE_BY_SCENARIO: dict[str, tuple[str, float]] = {
        "imbalance_positive_80": ("1", 0.80),
        "imbalance_positive_90": ("1", 0.90),
        "imbalance_negative_80": ("0", 0.80),
    }

    DROP_COLUMNS_BY_SCENARIO: dict[str, list[str]] = {
        "baseline_no_leakage": [
            "diabetes_stage",
            "diabetes_risk_score",
        ],
        "no_diagnostic_features": [
            "diabetes_stage",
            "diabetes_risk_score",
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
        ],
        "no_diagnostic_extended": [
            "diabetes_stage",
            "diabetes_risk_score",
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
            "insulin_level",
        ],
        "stage_multiclass_no_leakage": [
            "diagnosed_diabetes",
            "diabetes_risk_score",
        ],
    }

    KEEP_COLUMNS_BY_SCENARIO: dict[str, list[str]] = {
        "clinical_only": [
            "family_history_diabetes",
            "hypertension_history",
            "cardiovascular_history",
            "bmi",
            "waist_to_hip_ratio",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "cholesterol_total",
            "hdl_cholesterol",
            "ldl_cholesterol",
            "triglycerides",
            "insulin_level",
        ],
        "glucose_only": [
            "hba1c",
            "glucose_fasting",
            "glucose_postprandial",
        ],
    }

    SUPPORTED_DATASET_SCENARIOS = {
        "baseline_original",
        *DROP_COLUMNS_BY_SCENARIO.keys(),
        *KEEP_COLUMNS_BY_SCENARIO.keys(),
        *DIAGNOSTIC_NOISE_STD_FRACTION_BY_SCENARIO.keys(),
        *BINARY_IMBALANCE_BY_SCENARIO.keys(),
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
        df = self.dataset_loader.load(dataset_uri)

        df, scenario_report = self._apply_dataset_scenario(
            df=df,
            dataset_uri=dataset_uri,
            target_column=target_column,
            dataset_scenario=dataset_scenario,
            leakage_columns=leakage_columns,
            random_seed=random_seed,
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

        scenario_report_uri = self._persist_dataset_scenario_report(
            job_id=job_id,
            scenario_report=scenario_report,
        )

        preparation_metadata = self._build_preparation_metadata(
            scenario_report=scenario_report,
            scenario_report_uri=scenario_report_uri,
        )

        uris = self._persist_splits(job_id, target_column, splits)

        return PreparedDataset(
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

    def _apply_dataset_scenario(
        self,
        df: pd.DataFrame,
        dataset_uri: str,
        target_column: str,
        dataset_scenario: str,
        leakage_columns: list[str] | None,
        random_seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        scenario = self._normalize_dataset_scenario(dataset_scenario)

        original_columns = list(df.columns)
        original_row_count = len(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        scenario_type = "none"
        requested_drop_columns: list[str] | None = None
        dropped_columns: list[str] = []
        missing_requested_drop_columns: list[str] = []

        requested_keep_columns: list[str] | None = None
        kept_columns: list[str] = []
        missing_requested_keep_columns: list[str] = []

        requested_leakage_columns = leakage_columns
        scenario_parameters: dict[str, Any] = {}

        if scenario == "baseline_original":
            scenario_type = "none"

        elif scenario == "baseline_no_leakage":
            scenario_type = "drop_columns"

            requested_drop_columns = self._resolve_leakage_columns(
                target_column=target_column,
                leakage_columns=leakage_columns,
            )

            df, dropped_columns, missing_requested_drop_columns = self._drop_columns(
                df=df,
                target_column=target_column,
                requested_columns=requested_drop_columns,
            )

        elif scenario in self.DROP_COLUMNS_BY_SCENARIO:
            scenario_type = "drop_columns"

            requested_drop_columns = self.DROP_COLUMNS_BY_SCENARIO[scenario]

            df, dropped_columns, missing_requested_drop_columns = self._drop_columns(
                df=df,
                target_column=target_column,
                requested_columns=requested_drop_columns,
            )

        elif scenario in self.DIAGNOSTIC_NOISE_STD_FRACTION_BY_SCENARIO:
            scenario_type = "diagnostic_noise"

            requested_drop_columns = self._resolve_leakage_columns(
                target_column=target_column,
                leakage_columns=leakage_columns,
            )

            df, dropped_columns, missing_requested_drop_columns = self._drop_columns(
                df=df,
                target_column=target_column,
                requested_columns=requested_drop_columns,
            )

            noise_fraction = self.DIAGNOSTIC_NOISE_STD_FRACTION_BY_SCENARIO[scenario]
            df, noise_report = self._add_diagnostic_marker_noise(
                df=df,
                target_column=target_column,
                std_fraction=noise_fraction,
                random_seed=random_seed,
            )
            scenario_parameters.update(noise_report)

        elif scenario in self.BINARY_IMBALANCE_BY_SCENARIO:
            scenario_type = "binary_class_imbalance"

            requested_drop_columns = self._resolve_leakage_columns(
                target_column=target_column,
                leakage_columns=leakage_columns,
            )

            df, dropped_columns, missing_requested_drop_columns = self._drop_columns(
                df=df,
                target_column=target_column,
                requested_columns=requested_drop_columns,
            )

            target_label, target_ratio = self.BINARY_IMBALANCE_BY_SCENARIO[scenario]
            df, imbalance_report = self._apply_binary_class_imbalance(
                df=df,
                target_column=target_column,
                target_label=target_label,
                target_ratio=target_ratio,
                random_seed=random_seed,
            )
            scenario_parameters.update(imbalance_report)

        elif scenario in self.KEEP_COLUMNS_BY_SCENARIO:
            scenario_type = "keep_columns"

            requested_keep_columns = self.KEEP_COLUMNS_BY_SCENARIO[scenario]

            df, kept_columns, missing_requested_keep_columns = self._keep_columns(
                df=df,
                target_column=target_column,
                requested_columns=requested_keep_columns,
            )

        else:
            raise ValueError(f"Unsupported dataset_scenario '{scenario}'")

        final_columns = list(df.columns)

        report = {
            "job_dataset_uri": dataset_uri,
            "dataset_scenario": scenario,
            "scenario_type": scenario_type,
            "target_column": target_column,

            "original_row_count": original_row_count,
            "final_row_count": len(df),
            "original_column_count": len(original_columns),
            "final_column_count": len(final_columns),

            "original_columns": original_columns,
            "final_columns": final_columns,

            "requested_drop_columns": requested_drop_columns,
            "dropped_columns": dropped_columns,
            "missing_requested_drop_columns": missing_requested_drop_columns,

            "requested_keep_columns": requested_keep_columns,
            "kept_columns": kept_columns,
            "missing_requested_keep_columns": missing_requested_keep_columns,

            "requested_leakage_columns": requested_leakage_columns,
            "missing_requested_leakage_columns": self._missing_requested_columns(
                df_columns_before=original_columns,
                requested_columns=requested_leakage_columns,
            ),

            "scenario_parameters": scenario_parameters,
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
        if leakage_columns:
            return list(dict.fromkeys(leakage_columns))

        return self.DEFAULT_LEAKAGE_COLUMNS_BY_TARGET.get(target_column, [])

    def _drop_columns(
        self,
        df: pd.DataFrame,
        target_column: str,
        requested_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        requested_columns = list(dict.fromkeys(requested_columns))

        dropped_columns = [
            column
            for column in requested_columns
            if column in df.columns and column != target_column
        ]

        missing_columns = self._missing_requested_columns(
            df_columns_before=list(df.columns),
            requested_columns=requested_columns,
        )

        if dropped_columns:
            df = df.drop(columns=dropped_columns)

        return df, dropped_columns, missing_columns

    def _keep_columns(
        self,
        df: pd.DataFrame,
        target_column: str,
        requested_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        requested_columns = list(dict.fromkeys(requested_columns))

        kept_columns = [
            column
            for column in requested_columns
            if column in df.columns and column != target_column
        ]

        missing_columns = self._missing_requested_columns(
            df_columns_before=list(df.columns),
            requested_columns=requested_columns,
        )

        if not kept_columns:
            raise ValueError(
                "Dataset scenario requested keep_columns, but none of the requested "
                "feature columns exist in the dataset"
            )

        final_columns = kept_columns + [target_column]
        df = df[final_columns].copy()

        return df, kept_columns, missing_columns

    def _add_diagnostic_marker_noise(
        self,
        df: pd.DataFrame,
        target_column: str,
        std_fraction: float,
        random_seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rng = np.random.default_rng(random_seed)
        df = df.copy()

        noisy_columns = [
            column
            for column in self.DIAGNOSTIC_MARKER_COLUMNS
            if column in df.columns and column != target_column
        ]

        column_reports: dict[str, dict[str, float]] = {}
        for column in noisy_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            std = float(values.std(ddof=0))
            noise_std = std * std_fraction

            if not np.isfinite(noise_std) or noise_std <= 0.0:
                continue

            min_value = float(values.min())
            max_value = float(values.max())
            noise = rng.normal(loc=0.0, scale=noise_std, size=len(df))
            perturbed = values.to_numpy(dtype=float) + noise
            perturbed = np.clip(perturbed, min_value, max_value)
            df[column] = perturbed

            column_reports[column] = {
                "original_std": std,
                "noise_std": noise_std,
                "min_value": min_value,
                "max_value": max_value,
            }

        return df, {
            "noise_std_fraction": std_fraction,
            "noisy_columns": noisy_columns,
            "noise_column_reports": column_reports,
            "random_seed": random_seed,
        }

    def _apply_binary_class_imbalance(
        self,
        df: pd.DataFrame,
        target_column: str,
        target_label: str,
        target_ratio: float,
        random_seed: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio must be between 0 and 1")

        target_as_str = df[target_column].astype(str)
        labels = sorted(target_as_str.unique().tolist())
        if len(labels) != 2:
            raise ValueError(
                "Binary class imbalance scenarios require exactly 2 target classes; "
                f"found {labels}"
            )

        if target_label not in labels:
            raise ValueError(
                f"Target imbalance label '{target_label}' not found in classes {labels}"
            )

        other_label = next(label for label in labels if label != target_label)
        target_df = df[target_as_str == target_label]
        other_df = df[target_as_str == other_label]

        target_count = len(target_df)
        other_count = len(other_df)
        desired_other_count = int(round(target_count * (1.0 - target_ratio) / target_ratio))

        if desired_other_count <= other_count:
            sampled_target_df = target_df
            sampled_other_df = other_df.sample(
                n=max(1, desired_other_count),
                random_state=random_seed,
            )
        else:
            desired_target_count = int(round(other_count * target_ratio / (1.0 - target_ratio)))
            sampled_target_df = target_df.sample(
                n=min(target_count, max(1, desired_target_count)),
                random_state=random_seed,
            )
            sampled_other_df = other_df

        result = pd.concat([sampled_target_df, sampled_other_df], axis=0)
        result = result.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

        final_counts = result[target_column].astype(str).value_counts().to_dict()

        return result, {
            "target_label": target_label,
            "other_label": other_label,
            "requested_target_ratio": target_ratio,
            "original_class_counts": {
                target_label: target_count,
                other_label: other_count,
            },
            "final_class_counts": final_counts,
            "random_seed": random_seed,
        }

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

    def _build_preparation_metadata(
        self,
        scenario_report: dict[str, Any],
        scenario_report_uri: str,
    ) -> DatasetPreparationMetadata:
        return DatasetPreparationMetadata(
            dataset_scenario=scenario_report["dataset_scenario"],
            scenario_type=scenario_report["scenario_type"],

            requested_drop_columns=scenario_report["requested_drop_columns"],
            dropped_columns=scenario_report["dropped_columns"],
            missing_requested_drop_columns=scenario_report["missing_requested_drop_columns"],

            requested_keep_columns=scenario_report["requested_keep_columns"],
            kept_columns=scenario_report["kept_columns"],
            missing_requested_keep_columns=scenario_report["missing_requested_keep_columns"],

            requested_leakage_columns=scenario_report["requested_leakage_columns"],
            missing_requested_leakage_columns=scenario_report["missing_requested_leakage_columns"],

            original_column_count=scenario_report["original_column_count"],
            final_column_count=scenario_report["final_column_count"],
            original_row_count=scenario_report["original_row_count"],
            final_row_count=scenario_report["final_row_count"],

            scenario_report_uri=scenario_report_uri,
            scenario_parameters=scenario_report.get("scenario_parameters", {}),
        )

    def _persist_schema(self, job_id: str, schema_payload: dict) -> None:
        path = self.artifact_store.layout.dataset_schema_path(job_id)
        self.artifact_store.write_json(path, schema_payload)

    def _persist_dataset_scenario_report(
        self,
        job_id: str,
        scenario_report: dict[str, Any],
    ) -> str:
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