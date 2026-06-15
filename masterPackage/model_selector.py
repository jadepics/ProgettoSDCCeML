from __future__ import annotations

from typing import Optional

from common.contracts import ExperimentRecord


class ModelSelector:
    """
    Responsabilità:
    - scegliere il miglior esperimento tra quelli completati
    - supportare sia classification che regression

    selection_metric:
    - "auto"              -> accuracy per classification, r2 per regression
    - "accuracy"          -> forza accuracy
    - "balanced_accuracy" -> forza balanced accuracy
    - "macro_f1"          -> forza macro F1
    - "weighted_f1"       -> forza weighted F1
    - "r2"                -> forza r2
    """

    def __init__(self, selection_metric: str = "auto") -> None:
        allowed = {
            "auto",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "r2",
        }
        if selection_metric not in allowed:
            raise ValueError(
                f"Unsupported selection_metric '{selection_metric}'. "
                f"Allowed values: {sorted(allowed)}"
            )
        self.selection_metric = selection_metric

    def select_best(
        self,
        experiments: list[ExperimentRecord],
    ) -> ExperimentRecord:
        candidates = [
            experiment
            for experiment in experiments
            if experiment.validation_metrics is not None
        ]

        if not candidates:
            raise ValueError("No experiments with validation metrics available")

        best_experiment: Optional[ExperimentRecord] = None
        best_score: Optional[float] = None

        for experiment in candidates:
            score = self._extract_score(experiment)

            if best_score is None or score > best_score:
                best_score = score
                best_experiment = experiment

        if best_experiment is None:
            raise ValueError("Unable to select best experiment")

        return best_experiment

    def _extract_score(self, experiment: ExperimentRecord) -> float:
        metrics = experiment.validation_metrics
        if metrics is None:
            raise ValueError(
                f"Experiment '{experiment.experiment_id}' has no validation metrics"
            )

        metric_name = self._resolve_metric_name(experiment)

        if metric_name == "accuracy":
            if metrics.accuracy is None:
                raise ValueError(
                    f"Experiment '{experiment.experiment_id}' has no accuracy "
                    "inside validation_metrics"
                )
            return float(metrics.accuracy)

        if metric_name == "balanced_accuracy":
            if metrics.balanced_accuracy is None:
                raise ValueError(
                    f"Experiment '{experiment.experiment_id}' has no balanced_accuracy "
                    "inside validation_metrics"
                )
            return float(metrics.balanced_accuracy)

        if metric_name == "macro_f1":
            if metrics.macro_f1 is None:
                raise ValueError(
                    f"Experiment '{experiment.experiment_id}' has no macro_f1 "
                    "inside validation_metrics"
                )
            return float(metrics.macro_f1)

        if metric_name == "weighted_f1":
            if metrics.weighted_f1 is None:
                raise ValueError(
                    f"Experiment '{experiment.experiment_id}' has no weighted_f1 "
                    "inside validation_metrics"
                )
            return float(metrics.weighted_f1)

        if metric_name == "r2":
            if metrics.r2 is not None:
                return float(metrics.r2)

            # Compatibilità con vecchi job dove r2 era salvato male dentro classification_report
            report = metrics.classification_report or {}
            if isinstance(report, dict) and "r2" in report:
                return float(report["r2"])

            raise ValueError(
                f"Experiment '{experiment.experiment_id}' has no r2 "
                "inside validation_metrics"
            )

        raise ValueError(f"Unsupported resolved metric '{metric_name}'")

    def _resolve_metric_name(self, experiment: ExperimentRecord) -> str:
        if self.selection_metric != "auto":
            return self.selection_metric

        task_type = experiment.forest_config.task_type

        if task_type == "classification":
            return "accuracy"

        if task_type == "regression":
            return "r2"

        raise ValueError(
            f"Unsupported task_type '{task_type}' for experiment "
            f"'{experiment.experiment_id}'"
        )
