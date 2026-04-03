import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from worker.utils.proto_utils import matrix_from_proto


class WorkerService(rf_pb2_grpc.WorkerServiceServicer):

    def __init__(
        self,
        config,
        state,
        shard_trainer,
        shard_predictor,
        progress_store,
    ):
        self.config = config
        self.state = state

        self.shard_trainer = shard_trainer
        self.shard_predictor = shard_predictor
        self.progress_store = progress_store

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def TrainShard(self, request, context):
        job_id = request.job_id
        experiment_id = request.experiment_id
        task_id = request.task_id

        # Worker state
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

            # 👉 Idempotenza forte: task già completato → skip totale
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

            # Delegate training
            result = self.shard_trainer.train(request)

            self.state.on_task_success(task_id)

            return rf_pb2.TrainShardResponse(
                task_id=result.task_id,
                attempt_id=request.attempt_id,
                worker_id=self.config.worker_id,
                success=True,
                error="",
                artifacts=result.artifacts,
                completed_tree_ids=result.completed_tree_ids,
                failed_tree_ids=result.failed_tree_ids,
                elapsed_time_seconds=result.elapsed_time_seconds,
            )

        except Exception as exc:
            self.state.on_task_failure(task_id, str(exc))

            # persist failure
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
        model_id = request.model_id

        self.state.on_task_start(model_id)

        try:
            X = matrix_from_proto(request.features)

            result = self.shard_predictor.predict(request, X)

            self.state.on_task_success(model_id)

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                values=result.values,
                n_rows=result.n_rows,
                n_cols=result.n_cols,
            )

        except Exception as exc:
            self.state.on_task_failure(model_id, str(exc))

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                values=[],
                n_rows=0,
                n_cols=0,
            )

        finally:
            self.state.on_task_end(model_id)

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------
    def GetStatus(self, request, context):
        return rf_pb2.HeartbeatResponse(ok=True)
