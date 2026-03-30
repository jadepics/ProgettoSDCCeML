from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd


class DatasetLoader:
    """
    Responsabilità:
    - caricare un dataset da URI o path
    - astrarre il tipo di sorgente rispetto al resto del master
    - supportare i formati usati dal progetto

    Supporto attuale:
    - path locale
    - URI file://...
    - URL http/https per CSV
    - file locali CSV / Parquet

    Nota:
    questa classe NON valida il dataset.
    La validazione semantica viene delegata a DatasetValidator.
    """

    SUPPORTED_SCHEMES = {"", "file", "http", "https"}
    CSV_SUFFIXES = {".csv"}
    PARQUET_SUFFIXES = {".parquet", ".pq"}

    def __init__(self, supported_schemes: list[str] | None = None) -> None:
        self.supported_schemes = set(supported_schemes or self.SUPPORTED_SCHEMES)

    def load(self, dataset_uri: str) -> pd.DataFrame:
        """
        Carica un dataset e restituisce un DataFrame pandas.

        Esempi supportati:
        - /shared/datasets/my_dataset.csv
        - file:///shared/datasets/my_dataset.csv
        - /shared/jobs/job-1/prepared_dataset/train_features.parquet
        - https://example.com/dataset.csv
        """
        normalized_uri, scheme, is_local = self._resolve_dataset_uri(dataset_uri)

        self._validate_scheme(scheme, dataset_uri)
        dataset_format = self._detect_format(normalized_uri, scheme)

        if is_local:
            self._validate_local_path(normalized_uri)

        if dataset_format == "csv":
            return pd.read_csv(normalized_uri)

        if dataset_format == "parquet":
            if not is_local:
                raise ValueError(
                    "Remote parquet loading is not supported in this milestone. "
                    "Use a local/file:// path for parquet artifacts."
                )
            return pd.read_parquet(normalized_uri)

        raise ValueError(f"Unsupported dataset format for URI: {dataset_uri}")

    def _resolve_dataset_uri(self, dataset_uri: str) -> tuple[str, str, bool]:
        """
        Restituisce:
        - URI/path normalizzato
        - scheme
        - flag is_local
        """
        if dataset_uri is None or not dataset_uri.strip():
            raise ValueError("dataset_uri must be a non-empty string")

        dataset_uri = dataset_uri.strip()
        parsed = urlparse(dataset_uri)
        scheme = parsed.scheme.lower()

        # file://...
        if scheme == "file":
            # unquote gestisce eventuali spazi o caratteri escaped
            local_path = unquote(parsed.path)

            # Caso raro: file://hostname/path
            if parsed.netloc:
                local_path = f"//{parsed.netloc}{local_path}"

            return local_path, "file", True

        # URL remoti
        if scheme in {"http", "https"}:
            return dataset_uri, scheme, False

        # Path locale normale
        return dataset_uri, "", True

    def _validate_scheme(self, scheme: str, dataset_uri: str) -> None:
        if scheme not in self.supported_schemes:
            raise ValueError(
                f"Unsupported dataset URI scheme '{scheme}' for '{dataset_uri}'. "
                f"Supported schemes: {sorted(self.supported_schemes)}"
            )

    def _detect_format(self, normalized_uri: str, scheme: str) -> str:
        """
        Determina il formato dal suffisso del path.
        Per URL remoti usa solo il path della URL, ignorando query string.
        """
        if scheme in {"http", "https"}:
            path_part = urlparse(normalized_uri).path
        else:
            path_part = normalized_uri

        suffix = Path(path_part).suffix.lower()

        if suffix in self.CSV_SUFFIXES:
            return "csv"

        if suffix in self.PARQUET_SUFFIXES:
            return "parquet"

        raise ValueError(
            f"Unsupported dataset file extension '{suffix}'. "
            f"Supported extensions: {sorted(self.CSV_SUFFIXES | self.PARQUET_SUFFIXES)}"
        )

    def _validate_local_path(self, path_str: str) -> None:
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Dataset path is not a file: {path}")