from __future__ import annotations

from dataclasses import dataclass

import rf_v2_pb2 as rf_pb2

from client.training_presets import TrainingPreset


@dataclass(frozen=True)
class TrainingRequestConfig:
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    bootstrap: bool = True
    global_random_seed: int = 42

    max_depth_candidates: tuple[int, ...] = (5,)
    max_features_candidates: tuple[str, ...] = ("sqrt",)
    min_samples_split_candidates: tuple[int, ...] = (2,)
    min_samples_leaf_candidates: tuple[int, ...] = (1,)


def build_submit_training_request(
    *,
    preset: TrainingPreset,
    n_estimators_total: int,
    config: TrainingRequestConfig | None = None,
) -> rf_pb2.SubmitTrainingRequest:
    if config is None:
        config = TrainingRequestConfig()

    _validate_training_request_inputs(
        preset=preset,
        n_estimators_total=n_estimators_total,
        config=config,
    )

    request = rf_pb2.SubmitTrainingRequest(
        dataset_url=preset.dataset_path,
        target_column=preset.target_column,
        task_type=preset.task_type,
        dataset_scenario=preset.dataset_scenario,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        bootstrap=config.bootstrap,
        global_random_seed=config.global_random_seed,
        n_estimators_total=n_estimators_total,
        max_depth_candidates=list(config.max_depth_candidates),
        max_features_candidates=list(config.max_features_candidates),
        min_samples_split_candidates=list(config.min_samples_split_candidates),
        min_samples_leaf_candidates=list(config.min_samples_leaf_candidates),
        criterion_candidates=[preset.criterion],
    )

    request.leakage_columns.extend(preset.leakage_columns)

    return request


def _validate_training_request_inputs(
    *,
    preset: TrainingPreset,
    n_estimators_total: int,
    config: TrainingRequestConfig,
) -> None:
    if n_estimators_total <= 0:
        raise ValueError("n_estimators_total must be > 0")

    if not preset.dataset_path.strip():
        raise ValueError("preset.dataset_path cannot be empty")

    if not preset.target_column.strip():
        raise ValueError("preset.target_column cannot be empty")

    if preset.task_type not in {"classification", "regression"}:
        raise ValueError(
            "preset.task_type must be either 'classification' or 'regression'"
        )

    if not preset.dataset_scenario.strip():
        raise ValueError("preset.dataset_scenario cannot be empty")

    if not preset.criterion.strip():
        raise ValueError("preset.criterion cannot be empty")

    if not 0 <= config.validation_ratio < 1:
        raise ValueError("validation_ratio must be >= 0 and < 1")

    if not 0 <= config.test_ratio < 1:
        raise ValueError("test_ratio must be >= 0 and < 1")

    if config.validation_ratio + config.test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be < 1")

    if config.global_random_seed < 0:
        raise ValueError("global_random_seed must be >= 0")

    _validate_positive_int_tuple(
        "max_depth_candidates",
        config.max_depth_candidates,
    )

    _validate_non_empty_string_tuple(
        "max_features_candidates",
        config.max_features_candidates,
    )

    _validate_positive_int_tuple(
        "min_samples_split_candidates",
        config.min_samples_split_candidates,
    )

    _validate_positive_int_tuple(
        "min_samples_leaf_candidates",
        config.min_samples_leaf_candidates,
    )


def _validate_positive_int_tuple(
    field_name: str,
    values: tuple[int, ...],
) -> None:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")

    for value in values:
        if value <= 0:
            raise ValueError(f"{field_name} must contain only values > 0")


def _validate_non_empty_string_tuple(
    field_name: str,
    values: tuple[str, ...],
) -> None:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")

    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} cannot contain empty strings")