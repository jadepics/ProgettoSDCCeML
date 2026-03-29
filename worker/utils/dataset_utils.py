import numpy as np
from typing import Tuple


def split_features_labels(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Utility to validate and return features and labels.

    Parameters:
    - X: feature matrix
    - y: labels vector

    Returns:
    - X, y as numpy arrays (ensured format)
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None")

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    return X, y

def get_shard_indices(
    n_samples: int,
    num_shards: int,
    shard_id: int
) -> np.ndarray:
    """
    Returns indices of samples belonging to a specific shard.
    """
    if shard_id >= num_shards:
        raise ValueError("shard_id must be < num_shards")

    indices = np.arange(n_samples)
    shard_size = n_samples // num_shards

    start = shard_id * shard_size
    end = (shard_id + 1) * shard_size if shard_id != num_shards - 1 else n_samples

    return indices[start:end]