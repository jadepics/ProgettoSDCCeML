from __future__ import annotations

from itertools import product

from common.contracts import (
    ExperimentRecord,
    ForestConfiguration,
    TrainingRequest,
)
from common.enums import ExperimentStatus
from common.ids import generate_experiment_id


class ExperimentPlanner:
    """
    Responsabilità:
    - trasformare una TrainingRequest in una lista di esperimenti candidati
    - costruire ForestConfiguration coerenti
    - assegnare experiment_id deterministici rispetto al job

    Nota:
    in questa prima versione:
    - n_estimators della singola foresta viene preso da training_request.n_estimators_total
    - HyperparameterSpace definisce le combinazioni candidate degli altri parametri
    - ogni combinazione genera un ExperimentRecord
    """

    def plan(self, request: TrainingRequest) -> list[ExperimentRecord]:
        task_type = request.task_type.strip().lower()
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")

        space = request.hyperparameter_space

        max_depth_candidates = self._non_empty(space.max_depth_candidates, [None])
        max_features_candidates = self._non_empty(space.max_features_candidates, [None])
        min_samples_split_candidates = self._non_empty(space.min_samples_split_candidates, [2])
        min_samples_leaf_candidates = self._non_empty(space.min_samples_leaf_candidates, [1])
        criterion_candidates = self._non_empty(
            space.criterion_candidates,
            ["gini"] if task_type == "classification" else ["squared_error"],
        )

        experiments: list[ExperimentRecord] = []

        combinations = product(
            max_depth_candidates,
            max_features_candidates,
            min_samples_split_candidates,
            min_samples_leaf_candidates,
            criterion_candidates,
        )

        for index, (
            max_depth,
            max_features,
            min_samples_split,
            min_samples_leaf,
            criterion,
        ) in enumerate(combinations):
            experiment_id = generate_experiment_id(request.job_id, index)

            forest_config = ForestConfiguration(
                experiment_id=experiment_id,
                task_type=task_type,
                n_estimators=request.n_estimators_total,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                bootstrap=request.bootstrap,
                global_random_seed=request.global_random_seed,
            )

            experiment = ExperimentRecord(
                experiment_id=experiment_id,
                forest_config=forest_config,
                status=ExperimentStatus.PENDING,
                assigned_workers=[],
                expected_tree_count=request.n_estimators_total,
                completed_tree_count=0,
                validation_metrics=None,
            )
            experiments.append(experiment)

        if not experiments:
            raise ValueError("ExperimentPlanner produced no experiments")

        return experiments

    def select_initial_experiment(self, request: TrainingRequest) -> ExperimentRecord:
        """
        Helper comodo per la fase attuale:
        restituisce il primo esperimento pianificato.
        """
        experiments = self.plan(request)
        return experiments[0]

    def _non_empty(self, values, default):
        values = list(values)
        return values if values else list(default)