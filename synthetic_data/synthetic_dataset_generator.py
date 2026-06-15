from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SyntheticClassificationConfig:
    output_csv: str
    n_samples: int = 100_000
    n_features: int = 40
    n_informative: int = 20
    n_redundant: int = 10
    n_classes: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.01
    random_seed: int = 42
    target_column: str = "target"


@dataclass
class SyntheticRegressionConfig:
    output_csv: str
    n_samples: int = 100_000
    n_features: int = 40
    n_informative: int = 20
    noise: float = 10.0
    random_seed: int = 42
    target_column: str = "target"


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def make_feature_columns(n_features: int) -> list[str]:
    return [
        f"f_{index:03d}"
        for index in range(n_features)
    ]


def metadata_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_suffix(".metadata.json")


def write_dataset_and_metadata(
    df: pd.DataFrame,
    output_csv: Path,
    metadata: dict[str, Any],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    metadata_path = metadata_path_for_csv(output_csv)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def generate_synthetic_classification_dataset(
    config: SyntheticClassificationConfig,
) -> dict[str, Any]:
    if config.n_features <= 0:
        raise ValueError("n_features must be greater than 0")

    if config.n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")

    if config.n_informative <= 0:
        raise ValueError("n_informative must be greater than 0")

    if config.n_redundant < 0:
        raise ValueError("n_redundant cannot be negative")

    if config.n_informative + config.n_redundant > config.n_features:
        raise ValueError(
            "n_informative + n_redundant cannot be greater than n_features"
        )

    output_csv = resolve_output_path(config.output_csv)

    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_repeated=0,
        n_classes=config.n_classes,
        class_sep=config.class_sep,
        flip_y=config.flip_y,
        random_state=config.random_seed,
    )

    feature_columns = make_feature_columns(config.n_features)

    df = pd.DataFrame(X, columns=feature_columns)
    df[config.target_column] = y.astype(int)

    metadata = {
        "dataset_type": "synthetic_classification",
        "generator": "sklearn.datasets.make_classification",
        "config": asdict(config),
        "output_csv_resolved": str(output_csv),
        "n_rows": int(len(df)),
        "n_features": int(config.n_features),
        "feature_columns": feature_columns,
        "target_column": config.target_column,
        "class_distribution": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(y, return_counts=True)
            )
        },
    }

    write_dataset_and_metadata(
        df=df,
        output_csv=output_csv,
        metadata=metadata,
    )

    return metadata


def generate_synthetic_regression_dataset(
    config: SyntheticRegressionConfig,
) -> dict[str, Any]:
    if config.n_features <= 0:
        raise ValueError("n_features must be greater than 0")

    if config.n_samples <= 0:
        raise ValueError("n_samples must be greater than 0")

    if config.n_informative <= 0:
        raise ValueError("n_informative must be greater than 0")

    if config.n_informative > config.n_features:
        raise ValueError(
            "n_informative cannot be greater than n_features"
        )

    output_csv = resolve_output_path(config.output_csv)

    X, y = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        noise=config.noise,
        random_state=config.random_seed,
    )

    feature_columns = make_feature_columns(config.n_features)

    df = pd.DataFrame(X, columns=feature_columns)
    df[config.target_column] = y.astype(float)

    metadata = {
        "dataset_type": "synthetic_regression",
        "generator": "sklearn.datasets.make_regression",
        "config": asdict(config),
        "output_csv_resolved": str(output_csv),
        "n_rows": int(len(df)),
        "n_features": int(config.n_features),
        "feature_columns": feature_columns,
        "target_column": config.target_column,
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
    }

    write_dataset_and_metadata(
        df=df,
        output_csv=output_csv,
        metadata=metadata,
    )

    return metadata