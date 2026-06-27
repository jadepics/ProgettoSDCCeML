from __future__ import annotations

from typing import Optional, Union
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common.contracts import ForestConfiguration


class DecisionTreeFactory:
    """
    Costruisce alberi decisionali configurati.

    Isola completamente:
    - scelta del tipo di modello
    - parsing dei parametri
    """

    def __init__(self):
        self.task_type = None

    def _parse_max_features(self, value):
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return value

        value = str(value).strip().lower()

        if value in {"", "none"}:
            return None

        if value in {"sqrt", "log2"}:
            return value

        try:
            if value.isdigit():
                return int(value)

            return float(value)
        except ValueError:
            raise ValueError(f"Invalid max_features value: {value}")

    def _normalize_criterion(self, task_type: str, criterion: str) -> str:
        criterion = str(criterion or "").strip().lower()

        if task_type == "classification":
            if not criterion:
                return "gini"

            allowed = {"gini", "entropy", "log_loss"}
            if criterion not in allowed:
                raise ValueError(
                    f"Invalid classification criterion: {criterion}. "
                    f"Allowed values: {sorted(allowed)}"
                )

            return criterion

        if not criterion:
            return "squared_error"

        allowed = {"squared_error", "friedman_mse", "absolute_error", "poisson"}
        if criterion not in allowed:
            raise ValueError(
                f"Invalid regression criterion: {criterion}. "
                f"Allowed values: {sorted(allowed)}"
            )

        return criterion

    def create(
        self,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: Union[str, int, float, None],
        criterion: str,
        seed: int,
        task_type: str
    ):
        """
        Costruisce un DecisionTree pronto per il fit.
        """
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.task_type = task_type
        parsed_max_features = self._parse_max_features(max_features)
        parsed_criterion = self._normalize_criterion(task_type, criterion)

        if self.task_type == "classification":
            return DecisionTreeClassifier(
                criterion=parsed_criterion,
                max_depth=max_depth,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features=parsed_max_features,
                random_state=seed,
            )

        else:
            return DecisionTreeRegressor(
                criterion=parsed_criterion,
                max_depth=max_depth,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features=parsed_max_features,
                random_state=seed,
            )