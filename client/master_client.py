from __future__ import annotations

from typing import Any, Optional

import grpc

import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from client.config import ClientConfig, build_grpc_options, load_client_config
from client.printer import print_info, print_master_attempt


class NoLeaderAvailableError(RuntimeError):
    """Raised when no master leader accepts the client request."""


def is_not_leader_message(message: str) -> bool:
    normalized_message = str(message or "").lower()

    return (
        "not leader" in normalized_message
        or "operation allowed only" in normalized_message
        or "leader-only operation rejected" in normalized_message
    )


def submit_training(
    request: rf_pb2.SubmitTrainingRequest,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.SubmitTrainingResponse:
    return _call_with_leader_discovery(
        rpc_name="SubmitTraining",
        request=request,
        config=config,
    )


def get_job_status(
    job_id: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.GetJobStatusResponse:
    request = rf_pb2.GetJobStatusRequest(
        job_id=job_id.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="GetJobStatus",
        request=request,
        config=config,
    )


def list_experiments(
    job_id: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.ListExperimentsResponse:
    request = rf_pb2.ListExperimentsRequest(
        job_id=job_id.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="ListExperiments",
        request=request,
        config=config,
    )


def count_saved_trees(
    job_id: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.CountSavedTreesResponse:
    request = rf_pb2.CountSavedTreesRequest(
        job_id=job_id.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="CountSavedTrees",
        request=request,
        config=config,
    )


def get_validation_metrics(
    job_id: str,
    experiment_id: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.GetValidationMetricsResponse:
    request = rf_pb2.GetValidationMetricsRequest(
        job_id=job_id.strip(),
        experiment_id=experiment_id.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="GetValidationMetrics",
        request=request,
        config=config,
    )


def run_inference_on_model_split(
    model_id: str,
    split_name: str,
    rows: int,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.RunInferenceOnModelSplitResponse:
    request = rf_pb2.RunInferenceOnModelSplitRequest(
        model_id=model_id.strip(),
        split_name=split_name.strip(),
        rows=rows,
    )

    return _call_with_leader_discovery(
        rpc_name="RunInferenceOnModelSplit",
        request=request,
        config=config,
    )


def resume_training(
    job_id: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.ResumeTrainingResponse:
    request = rf_pb2.ResumeTrainingRequest(
        job_id=job_id.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="ResumeTraining",
        request=request,
        config=config,
    )


def submit_inference(
    model_id: str,
    features_uri: str,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.SubmitInferenceResponse:
    request = rf_pb2.SubmitInferenceRequest(
        model_id=model_id.strip(),
        features_uri=features_uri.strip(),
    )

    return _call_with_leader_discovery(
        rpc_name="SubmitInference",
        request=request,
        config=config,
    )


def download_model(
    model_id: str,
    model_format: str = "joblib",
    include_bytes: bool = False,
    overwrite: bool = False,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.DownloadModelResponse:
    request = rf_pb2.DownloadModelRequest(
        model_id=model_id.strip(),
        format=model_format.strip(),
        include_bytes=include_bytes,
        overwrite=overwrite,
    )

    return _call_with_leader_discovery(
        rpc_name="DownloadModel",
        request=request,
        config=config,
    )


def reset_shared_artifacts(
    confirm: bool,
    config: Optional[ClientConfig] = None,
) -> rf_pb2.ResetSharedArtifactsResponse:
    request = rf_pb2.ResetSharedArtifactsRequest(
        confirm=confirm,
    )

    return _call_with_leader_discovery(
        rpc_name="ResetSharedArtifacts",
        request=request,
        config=config,
    )


def _call_with_leader_discovery(
    *,
    rpc_name: str,
    request: Any,
    config: Optional[ClientConfig],
) -> Any:
    if config is None:
        config = load_client_config()

    last_error: Optional[Exception] = None

    for master_address in config.master_addresses:
        try:
            print_master_attempt(master_address)

            response = _call_rpc_on_master(
                master_address=master_address,
                rpc_name=rpc_name,
                request=request,
                config=config,
            )

            if response is None:
                raise RuntimeError(f"{rpc_name} returned None")

            response_error = str(getattr(response, "error", "") or "")
            response_message = str(getattr(response, "message", "") or "")

            if is_not_leader_message(response_error) or is_not_leader_message(
                response_message
            ):
                print_info(
                    f"Master {master_address} is not leader. "
                    "Trying next candidate..."
                )
                last_error = RuntimeError(response_error or response_message)
                continue

            _print_rpc_acceptance_if_useful(
                rpc_name=rpc_name,
                master_address=master_address,
                response=response,
            )

            return response

        except grpc.RpcError as exc:
            last_error = exc
            print_info(
                f"{rpc_name} RPC failed on {master_address}: "
                f"{_format_grpc_error(exc)}"
            )
            continue

        except Exception as exc:
            last_error = exc
            print_info(f"{rpc_name} failed on {master_address}: {exc}")
            continue

    raise NoLeaderAvailableError(
        f"No master leader accepted {rpc_name}. "
        f"Last error: {last_error}"
    )


def _call_rpc_on_master(
    *,
    master_address: str,
    rpc_name: str,
    request: Any,
    config: ClientConfig,
) -> Any:
    with grpc.insecure_channel(
        master_address,
        options=build_grpc_options(config),
    ) as channel:
        stub = rf_pb2_grpc.CoordinatorServiceStub(channel)

        rpc = getattr(stub, rpc_name)

        return rpc(
            request,
            timeout=config.grpc_timeout_seconds,
        )


def _print_rpc_acceptance_if_useful(
    *,
    rpc_name: str,
    master_address: str,
    response: Any,
) -> None:
    if rpc_name == "SubmitTraining" and getattr(response, "job_id", ""):
        print_info(f"Training accepted by leader {master_address}")
        return

    if rpc_name == "ResumeTraining" and getattr(response, "job_id", ""):
        print_info(f"Resume accepted by leader {master_address}")
        return

    print_info(f"{rpc_name} handled by leader {master_address}")


def _format_grpc_error(exc: grpc.RpcError) -> str:
    code = exc.code()
    details = exc.details()

    if code is None and not details:
        return str(exc)

    if code is None:
        return str(details)

    if details:
        return f"{code.name}: {details}"

    return code.name


def get_status_name(status_value: Any) -> str:
    try:
        return rf_pb2.JobStatus.Name(status_value)
    except Exception:
        return str(status_value)