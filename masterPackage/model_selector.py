from __future__ import annotations

from typing import Sequence

from common.contracts import ExperimentRecord
from common.enums import ExperimentStatus


class ModelSelector:
    """
    Responsabilità:
    - scegliere l'esperimento migliore tra quelli validati
    - applicare una metrica di selezione consistente
    - restituire l'ExperimentRecord vincente

    Nota:
    in classificazione la metrica di default è accuracy.
    In regressione, in questa prima versione, usa r2 se disponibile
    dentro validation_metrics.classification_report["r2"].
    """

    def __init__(self, selection_metric: str = "accuracy") -> None:
        self.selection_metric = selection_metric

    def select_best(self, experiments: Sequence[ExperimentRecord]) -> ExperimentRecord:
        if not experiments:
            raise ValueError("No experiments available for selection")

        eligible = [
            experiment
            for experiment in experiments
            if experiment.status == ExperimentStatus.COMPLETED
            and experiment.validation_metrics is not None
        ]

        if not eligible:
            raise ValueError("No completed experiments with validation metrics available")

        ranked = sorted(
            eligible,
            key=self._score_experiment,
            reverse=True,
        )
        return ranked[0]

    def select_best_for_job(self, job_repository, job_id: str) -> ExperimentRecord:
        experiments = job_repository.list_experiments(job_id)
        return self.select_best(experiments)

    def _score_experiment(self, experiment: ExperimentRecord) -> float:
        metrics = experiment.validation_metrics
        if metrics is None:
            raise ValueError(
                f"Experiment '{experiment.experiment_id}' has no validation metrics"
            )

        metric_name = self.selection_metric.strip().lower()

        if metric_name == "accuracy":
            return float(metrics.accuracy)

        if metric_name == "r2":
            report = metrics.classification_report or {}
            return float(report.get("r2", float("-inf")))

        raise ValueError(f"Unsupported selection metric '{self.selection_metric}'")