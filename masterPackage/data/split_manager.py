from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DatasetSplits:
    """
    Contenitore intermedio per gli split del dataset.

    Questa struttura NON è il PreparedDataset finale:
    è solo il risultato dello split in memoria,
    prima della persistenza su storage condiviso.
    """
    train_features: pd.DataFrame
    train_labels: pd.Series
    validation_features: pd.DataFrame
    validation_labels: pd.Series
    test_features: pd.DataFrame
    test_labels: pd.Series
    class_labels: list[str] | None


class SplitManager:
    """
    Responsabilità:
    - dividere il dataset in train / validation / test
    - usare uno split riproducibile con seed
    - usare stratify in classificazione quando possibile

    Strategia:
    1. si separa prima il test set
    2. poi dal restante train+validation si estrae il validation set
    """

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        validation_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> DatasetSplits:
        self._validate_ratios(validation_ratio, test_ratio)

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        class_labels = None
        if task_type == "classification":
            class_labels = sorted(y.astype(str).unique().tolist())

        # 1) Split train+validation vs test
        if test_ratio > 0.0:
            stratify_target = self._build_stratify_target(y, task_type, test_ratio)
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X,
                y,
                test_size=test_ratio,
                random_state=random_seed,
                stratify=stratify_target,
            )
        else:
            X_train_val = X
            y_train_val = y
            X_test = self._empty_features_like(X)
            y_test = self._empty_labels_like(y)

        # 2) Split train vs validation
        if validation_ratio > 0.0:
            remaining_ratio = 1.0 - test_ratio
            validation_ratio_on_remaining = validation_ratio / remaining_ratio

            stratify_target = self._build_stratify_target(
                y_train_val,
                task_type,
                validation_ratio_on_remaining,
            )

            X_train, X_validation, y_train, y_validation = train_test_split(
                X_train_val,
                y_train_val,
                test_size=validation_ratio_on_remaining,
                random_state=random_seed,
                stratify=stratify_target,
            )
        else:
            X_train = X_train_val
            y_train = y_train_val
            X_validation = self._empty_features_like(X)
            y_validation = self._empty_labels_like(y)

        return DatasetSplits(
            train_features=X_train.reset_index(drop=True),
            train_labels=y_train.reset_index(drop=True),
            validation_features=X_validation.reset_index(drop=True),
            validation_labels=y_validation.reset_index(drop=True),
            test_features=X_test.reset_index(drop=True),
            test_labels=y_test.reset_index(drop=True),
            class_labels=class_labels,
        )

    def _validate_ratios(self, validation_ratio: float, test_ratio: float) -> None:
        if validation_ratio < 0.0 or test_ratio < 0.0:
            raise ValueError("validation_ratio and test_ratio must be >= 0")

        if validation_ratio + test_ratio >= 1.0:
            raise ValueError("validation_ratio + test_ratio must be < 1.0")

    def _build_stratify_target(
        self,
        y: pd.Series,
        task_type: str,
        split_ratio: float,
    ) -> pd.Series | None:
        if task_type != "classification":
            return None

        if y.empty:
            return None

        counts = y.astype(str).value_counts()
        n_classes = len(counts)

        if n_classes < 2:
            return None

        if counts.min() < 2:
            return None

        split_size = int(round(len(y) * split_ratio))
        remaining_size = len(y) - split_size
        if split_size < n_classes or remaining_size < n_classes:
            return None

        return y

    def _empty_features_like(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.iloc[0:0].copy()

    def _empty_labels_like(self, y: pd.Series) -> pd.Series:
        return y.iloc[0:0].copy()