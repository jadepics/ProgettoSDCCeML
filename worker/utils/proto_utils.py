from typing import Any
import numpy as np


def matrix_from_proto(matrix_proto: Any) -> np.ndarray:
    """
Converti un protocollo DenseMatrix in un numpy ndarray.
Argomenti:
matrice_proto: oggetto protobuf con:
- valori: lista piatta di float
- n_rows: numero di righe
- n_cols: numero di colonne
Resi:
np.ndarray con forma (n_rows, n_cols)
Rilanci:
ValueError: se la dimensione dei dati non è coerente """

    # Estrazione dati grezzi
    values = matrix_proto.values
    n_rows = matrix_proto.n_rows
    n_cols = matrix_proto.n_cols

    # Validazione dimensionale
    expected_size = n_rows * n_cols
    if len(values) != expected_size:
        raise ValueError(
            f"Inconsistent matrix size: expected {expected_size} values "
            f"but got {len(values)}"
        )

    # Conversione in numpy array
    array = np.array(values, dtype=float)

    # Reshape in matrice 2D (row-major)
    matrix = array.reshape((n_rows, n_cols))

    return matrix