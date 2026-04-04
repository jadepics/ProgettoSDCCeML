def tree_artifact_path(job_id: str, experiment_id: str, tree_index: int) -> str:
    return f"jobs/{job_id}/experiments/{experiment_id}/trees/tree_{tree_index}.joblib"

def worker_snapshot_path(job_id: str, experiment_id: str, worker_id: str) -> str:
    return f"jobs/{job_id}/experiments/{experiment_id}/snapshots/worker_{worker_id}.json"

def manifest_path(job_id: str, experiment_id: str) -> str:
    return f"jobs/{job_id}/experiments/{experiment_id}/manifests/manifest.json"

def dataset_path(job_id: str, dataset_name: str) -> str:
    return f"jobs/{job_id}/datasets/{dataset_name}"

def tree_metadata_path(job_id: str, experiment_id: str, tree_index: int) -> str:
    return f"jobs/{job_id}/experiments/{experiment_id}/trees/tree_{tree_index}.json"
