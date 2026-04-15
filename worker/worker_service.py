import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc
from common.contracts import TrainingShard, ForestConfiguration, TreeArtifactMetadata
from worker.mappers.tree_artifact_mapper import to_proto_tree_artifact
from worker.utils.dataset_utils import split_features_labels

from worker.utils.proto_utils import matrix_from_proto


class WorkerService(rf_pb2_grpc.WorkerServiceServicer):

    def __init__(
        self,
        config,
        state,
        shard_trainer,
        shard_predictor,
        progress_store,
        artifact_store,
        data_loader
    ):
        self.config = config
        self.state = state
        self.shard_trainer = shard_trainer
        self.shard_predictor = shard_predictor
        self.progress_store = progress_store
        self.artifact_store = artifact_store
        self.data_loader = data_loader

    def TrainShard(self, request, context):
        job_id = request.job_id
        experiment_id = request.experiment_id
        task_id = request.task_id

        # ----------------------------------------
        # Worker state
        # ----------------------------------------
        self.state.on_task_start(task_id)

        try:
            status = self.progress_store.start_task(
                job_id,
                experiment_id,
                task_id,
                metadata={
                    "attempt_id": request.attempt_id,
                    "worker_id": self.config.worker_id,
                },
            )

            # ----------------------------------------
            # Idempotenza task-level
            # ----------------------------------------
            if status == "ALREADY_COMPLETED":
                existing = self.progress_store.get_task(job_id, experiment_id, task_id)

                return rf_pb2.TrainShardResponse(
                    task_id=task_id,
                    attempt_id=request.attempt_id,
                    worker_id=self.config.worker_id,
                    success=True,
                    error="",
                    artifacts=[],
                    completed_tree_ids=existing.get("completed_tree_ids", []),
                    failed_tree_ids=existing.get("failed_tree_ids", []),
                    elapsed_time_seconds=0.0,
                )

            # ----------------------------------------
            # 1. Convert proto → TrainingShard
            # ----------------------------------------
            shard = TrainingShard(
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                job_id=request.job_id,
                experiment_id=request.experiment_id,
                assigned_worker_id=request.assigned_worker_id,
                tree_start_index=request.tree_start_index,
                tree_count=request.tree_count,
                forest_config=ForestConfiguration(
                    experiment_id=request.experiment_id,
                    task_type=request.task_type,
                    n_estimators=request.n_estimators,
                    max_depth=request.max_depth if request.max_depth > 0 else None,
                    max_features=request.max_features,
                    min_samples_split=request.min_samples_split,
                    min_samples_leaf=request.min_samples_leaf,
                    criterion=request.criterion,
                    bootstrap=request.bootstrap,
                    global_random_seed=request.global_random_seed,
                ),
                train_features_uri=request.train_features_uri,
                train_labels_uri=request.train_labels_uri,
                artifact_output_dir=request.artifact_output_dir,
                seed_base=request.seed_base,
                lease_expires_at_ts=request.lease_expires_at_unix_ms / 1000.0            )

            # ----------------------------------------
            # 2. Load dataset (URI → numpy)
            # ----------------------------------------
            X = self.data_loader.load_numpy(request.train_features_uri)
            y = self.data_loader.load_numpy(request.train_labels_uri)
            
            X, y = split_features_labels(X, y)

            # ----------------------------------------
            # 3. Call trainer (PURE ML)
            # ----------------------------------------
            result = self.shard_trainer.train(shard, X, y)

            if result.success:
                self.state.on_task_success(task_id)
            else:
                self.state.on_task_failure(task_id, result.error_message or "")

            # ----------------------------------------
            # 4. Convert result → proto
            # ----------------------------------------
            return rf_pb2.TrainShardResponse(
                task_id=result.task_id,
                attempt_id=result.attempt_id,
                worker_id=result.worker_id,
                success=result.success,
                error=result.error_message or "",
                artifacts=[to_proto_tree_artifact(a) for a in result.tree_artifacts],
                completed_tree_ids=result.completed_tree_ids,
                failed_tree_ids=result.failed_tree_ids,
                elapsed_time_seconds=result.elapsed_time_seconds,
            )

        except Exception as exc:
            self.state.on_task_failure(task_id, str(exc))

            self.progress_store.fail_task(
                job_id,
                experiment_id,
                task_id,
                error=str(exc),
            )

            return rf_pb2.TrainShardResponse(
                task_id=task_id,
                attempt_id=request.attempt_id,
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                artifacts=[],
                completed_tree_ids=[],
                failed_tree_ids=[],
                elapsed_time_seconds=0.0,
            )

        finally:
            self.state.on_task_end(task_id)


    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    def PredictShard(self, request, context):
        self.state.on_task_start(request.model_id)

        try:
            # 1. input
            X = matrix_from_proto(request.features)

            # 2. usare URIs dal master
            artifact_uris = list(request.tree_artifact_uris)

            if not artifact_uris:
                raise ValueError("No tree artifacts provided")

            # 3. prediction (NO aggregation finale)
            result = self.shard_predictor.predict(
                artifact_uris,
                X
            )

            self.state.on_task_success(request.model_id)

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                #precedentemente questa riga era
                #values=result.values,
                values=result.values.tolist(),
                n_rows=result.n_rows,
                n_cols=result.n_cols,
            )

        except Exception as exc:
            self.state.on_task_failure(request.model_id, str(exc))

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                values=[],
                n_rows=0,
                n_cols=0,
            )

        finally:
            self.state.on_task_end(request.model_id)
