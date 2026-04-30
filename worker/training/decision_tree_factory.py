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
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid max_features value: {value}")

    def create(
        self,
        max_depth: Optional[int],
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: Union[str, float, None],
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

        if self.task_type == "classification":
            return DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features=parsed_max_features,
                random_state=seed,
            )

        else:
            return DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=max(2, min_samples_split),
                min_samples_leaf=max(1, min_samples_leaf),
                max_features=parsed_max_features,
                random_state=seed,
            )