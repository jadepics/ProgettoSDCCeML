import rf_v2_pb2 as rf_pb2
import rf_v2_pb2_grpc as rf_pb2_grpc

from common.chaos import maybe_fail
from common.contracts import TrainingShard, ForestConfiguration
from common.prediction_io import read_features_from_uri, save_prediction_array

from worker.mappers.tree_artifact_mapper import to_proto_tree_artifact
from worker.utils.proto_utils import matrix_from_proto


# Facciata RPC del worker: espone via gRPC le operazioni di training e prediction,
# converte le richieste protobuf in oggetti di dominio, delega l'esecuzione ai
# componenti applicativi interni (trainer/predictor) e aggiorna lo stato locale
# del worker durante il ciclo di vita dei task.

class WorkerService(rf_pb2_grpc.WorkerServiceServicer):

    def __init__(
        self,
        config,
        state,
        shard_trainer,
        shard_predictor,
    ):
        self.config = config
        self.state = state
        self.shard_trainer = shard_trainer
        self.shard_predictor = shard_predictor

    def TrainShard(self, request, context):
        task_id = request.task_id

        self.state.on_task_start(task_id)

        try:
            # ----------------------------------------
            #  Mapping proto → domain
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
                lease_expires_at_ts=request.lease_expires_at_unix_ms / 1000.0,
            )

            maybe_fail("worker.train.before_trainer")

            # ----------------------------------------
            #  Delega totale al trainer
            # ----------------------------------------
            result = self.shard_trainer.train(
                shard,
                context=context,
                progress_callback=self.state.on_task_progress,
            )

            # ----------------------------------------
            #  Stato worker
            # ----------------------------------------
            if result.success:
                self.state.on_task_success(task_id)
            else:
                self.state.on_task_failure(task_id, result.error_message or "")

            # ----------------------------------------
            #  Mapping result → proto
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
        """
        Prediction distribuita per sottoinsiemi di alberi.


        Modalità scalabile:
          - input: request.features_uri
          - output: response.prediction_uri
          - le predizioni parziali vengono salvate su storage condiviso
        """
        task_id = request.prediction_id or request.model_id
        self.state.on_task_start(task_id)

        try:
            artifact_uris = list(request.tree_artifact_uris)

            if not artifact_uris:
                raise ValueError("No tree artifacts provided")

            # ----------------------------------------
            # Input features
            # ----------------------------------------
            if request.features_uri:
                if not request.prediction_output_dir:
                    raise ValueError(
                        "prediction_output_dir is required when features_uri is used"
                    )

                if not request.prediction_id:
                    raise ValueError(
                        "prediction_id is required when features_uri is used"
                    )

                X = read_features_from_uri(request.features_uri, dtype=float)
            else:
                # Percorso legacy/fallback.
                X = matrix_from_proto(request.features)

            maybe_fail("worker.predict.before_predictor")

            # ----------------------------------------
            #  Prediction parziale
            # ----------------------------------------
            # Classification:
            #   result = voti parziali per classe.
            #
            # Regression:
            #   result = somma parziale delle predizioni degli alberi
            #            assegnati a questo worker.
            #
            # L'aggregazione finale resta sul master.
            result = self.shard_predictor.predict(
                tree_artifact_uris=artifact_uris,
                X=X,
                task_type=request.task_type,
                class_labels=list(request.class_labels),
            )

            if result.ndim == 1:
                result = result.reshape(-1, 1)

            if result.ndim != 2:
                raise ValueError(
                    f"Shard predictor result must be 1D or 2D, got ndim={result.ndim}"
                )

            n_rows, n_cols = result.shape

            self.state.on_task_success(task_id)

            # ----------------------------------------
            # Output scalabile: salva .npy e ritorna URI
            # ----------------------------------------
            if request.prediction_output_dir:
                prediction_uri = save_prediction_array(
                    array=result,
                    output_dir_uri_or_path=request.prediction_output_dir,
                    prediction_id=request.prediction_id,
                )

                return rf_pb2.PredictShardResponse(
                    worker_id=self.config.worker_id,
                    success=True,
                    error="",
                    values=[],
                    n_rows=n_rows,
                    n_cols=n_cols,
                    prediction_uri=prediction_uri,
                )

            # ----------------------------------------
            # Output legacy: ritorna values dentro gRPC
            # ----------------------------------------
            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=True,
                error="",
                values=result.flatten().tolist(),
                n_rows=n_rows,
                n_cols=n_cols,
                prediction_uri="",
            )

        except Exception as exc:
            self.state.on_task_failure(task_id, str(exc))

            return rf_pb2.PredictShardResponse(
                worker_id=self.config.worker_id,
                success=False,
                error=str(exc),
                values=[],
                n_rows=0,
                n_cols=0,
                prediction_uri="",
            )

        finally:
            self.state.on_task_end(task_id)