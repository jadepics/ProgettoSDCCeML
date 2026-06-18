from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd


def normalize_uri_to_path(uri: str) -> Path:
    """
    Converte un URI file://... oppure un path locale in pathlib.Path.

    Nel progetto gli artifact sono normalmente salvati come:
      file:///mnt/efs/...

    Supporta anche path semplici:
      /mnt/efs/...
    """
    if not uri:
        raise ValueError("URI/path cannot be empty")

    parsed = urlparse(uri)

    if parsed.scheme == "":
        return Path(uri)

    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URI scheme for local file access: {uri}")

    path = url2pathname(unquote(parsed.path))

    if os.name == "nt":
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        elif path.startswith("/") and len(path) >= 3 and path[2] == ":":
            path = path[1:]

    return Path(path)


def path_to_file_uri(path: str | Path) -> str:
    """
    Converte un path locale in URI file://.
    """
    return Path(path).resolve().as_uri()


def read_features_from_uri(features_uri: str, dtype=float) -> np.ndarray:
    """
    Legge un file parquet di feature e ritorna una matrice NumPy.

    Questo verrà usato dai worker nel nuovo flusso:
      worker riceve features_uri
      worker legge le feature da EFS
      worker predice con il proprio sottoinsieme di alberi
    """
    path = normalize_uri_to_path(features_uri)
    df = pd.read_parquet(path)

    if dtype is None:
        return df.to_numpy()

    return df.to_numpy(dtype=dtype)


def read_parquet_row_count(uri: str) -> int:
    """
    Ritorna il numero di righe di un parquet.

    Implementazione semplice e robusta: legge il dataframe.
    In futuro si può ottimizzare leggendo metadata parquet.
    """
    path = normalize_uri_to_path(uri)
    df = pd.read_parquet(path)
    return int(df.shape[0])


def save_prediction_array(
    array: np.ndarray,
    output_dir_uri_or_path: str,
    prediction_id: str,
) -> str:
    """
    Salva una matrice di predizioni in formato .npy e ritorna il file URI.

    Le predizioni parziali non vengono più restituite dentro gRPC.
    Il worker le salva su EFS e ritorna solo prediction_uri.
    """
    if not prediction_id:
        raise ValueError("prediction_id cannot be empty")

    arr = np.asarray(array)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Prediction array must be 1D or 2D, got ndim={arr.ndim}")

    output_dir = normalize_uri_to_path(output_dir_uri_or_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{prediction_id}.npy"
    np.save(path, arr)

    return path_to_file_uri(path)


def load_prediction_array(prediction_uri: str) -> np.ndarray:
    """
    Carica una matrice .npy prodotta da un worker.
    """
    path = normalize_uri_to_path(prediction_uri)
    arr = np.load(path, allow_pickle=False)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Loaded prediction array must be 1D or 2D, got ndim={arr.ndim}")

    return arr


def save_final_prediction_array(
    array: np.ndarray,
    output_uri_or_path: str,
) -> str:
    """
    Salva la matrice finale di predizioni già aggregata dal master.

    Usata soprattutto per SubmitInference, dove la response gRPC ritorna
    solo prediction_uri invece di tutte le predizioni.
    """
    arr = np.asarray(array)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Final prediction array must be 1D or 2D, got ndim={arr.ndim}")

    output_path = normalize_uri_to_path(output_uri_or_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, arr)

    return path_to_file_uri(output_path)