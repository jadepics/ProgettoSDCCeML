import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from worker.utils.proto_utils import matrix_from_proto


class WorkerService(rf_pb2_grpc.WorkerServiceServicer):

    def __init__(
        self,
        config,
        state,
        shard_trainer,
        shard_predictor
    ):
        self.config = config
        self.state = state

        self.shard_trainer = shard_trainer
        self.shard_predictor = shard_predictor

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def TrainShard(self, request, context):
        # 1. Update worker state
        self.state.on_task_start(request.task_id)

        try:
            # 2. Delegate completely to trainer
            result = self.shard_trainer.train(request)

            # 3. Update state
            self.state.on_task_success(request.task_id)

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
            # 4. Failure handling
            self.state.on_task_failure(request.task_id, str(exc))

            return rf_pb2.TrainShardResponse(
                task_id=request.task_id,
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
            self.state.on_task_end(request.task_id)

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    def PredictShard(self, request, context):
        self.state.on_task_start(request.model_id)

        try:
            # Delegate prediction entirely
            result = self.shard_predictor.predict(request)

            self.state.on_task_success(request.model_id)

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                values=result.values,
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

    # --------------------------------------------------
    # OPTIONAL: heartbeat / status
    # --------------------------------------------------
    def GetStatus(self, request, context):
        return rf_pb2.HeartbeatResponse(ok=True)