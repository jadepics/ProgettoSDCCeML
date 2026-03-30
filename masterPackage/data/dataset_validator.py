from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from common.contracts import DatasetSchema


class DatasetValidator:
    """
    Responsabilità:
    - verificare che il dataset sia adatto al training
    - costruire il DatasetSchema condiviso

    Cosa controlla:
    - dataset non vuoto
    - target_column presente
    - esistenza di almeno una feature
    - task_type valido
    - regressione: target numerico
    - classificazione: almeno 2 classi

    Non esegue lo split e non salva nulla su disco.
    """

    SUPPORTED_TASK_TYPES = {"classification", "regression"}

    def validate(
        self,
        df: pd.DataFrame,
        dataset_uri: str,
        target_column: str,
        task_type: str,
    ) -> DatasetSchema:
        """
        Valida dataset e costruisce il relativo schema.

        Returns
        -------
        DatasetSchema
            Schema condiviso usato nelle fasi successive.
        """
        self._validate_not_empty(df)
        self._validate_target_exists(df, target_column)
        normalized_task_type = self._normalize_task_type(task_type)
        feature_names = self._extract_feature_names(df, target_column)
        self._validate_has_features(feature_names)

        label_mapping = None

        if normalized_task_type == "classification":
            label_mapping = self._validate_classification_target(df, target_column)
        elif normalized_task_type == "regression":
            self._validate_regression_target(df, target_column)

        return DatasetSchema(
            dataset_uri=dataset_uri,
            target_column=target_column,
            feature_names=feature_names,
            task_type=normalized_task_type,
            label_mapping=label_mapping,
            preprocessing_uri=None,
        )

    def _normalize_task_type(self, task_type: str) -> str:
        task_type = task_type.strip().lower()
        if task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                f"Supported values: {sorted(self.SUPPORTED_TASK_TYPES)}"
            )
        return task_type

    def _validate_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Dataset is empty")

    def _validate_target_exists(self, df: pd.DataFrame, target_column: str) -> None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

    def _extract_feature_names(self, df: pd.DataFrame, target_column: str) -> list[str]:
        return [column for column in df.columns if column != target_column]

    def _validate_has_features(self, feature_names: list[str]) -> None:
        if not feature_names:
            raise ValueError("Dataset must contain at least one feature column")

    def _validate_classification_target(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, int]:
        """
        Verifica che il target di classificazione abbia almeno 2 classi.

        Restituisce anche una label_mapping deterministica:
        label stringa -> indice intero
        """
        target_as_str = df[target_column].astype(str)
        class_labels = sorted(target_as_str.unique().tolist())

        if len(class_labels) < 2:
            raise ValueError("Classification dataset must contain at least 2 classes")

        return {label: index for index, label in enumerate(class_labels)}

    def _validate_regression_target(self, df: pd.DataFrame, target_column: str) -> None:
        if not is_numeric_dtype(df[target_column]):
            raise ValueError("Regression target column must be numeric")