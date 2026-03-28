from __future__ import annotations

import numpy as np
import joblib


class ShardPredictor:

    def __init__(self, store):
        self.store = store

    def predict(self, request):
        X = self._matrix_from_proto(request.features)

        if X.size == 0:
            raise ValueError("Empty input batch")

        if not request.tree_artifact_uris:
            raise ValueError("No tree artifact URIs provided")

        model_type = request.model_type.strip().lower()

        if model_type == "classification":
            return self._predict_classification(X, request)

        elif model_type == "regression":
            return self._predict_regression(X, request)

        else:
            raise ValueError("Unsupported model_type")

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    def _predict_classification(self, X, request):
        class_labels = list(request.class_labels)

        # ✅ VALIDAZIONE CORRETTA (reintrodotta)
        if not class_labels:
            raise ValueError("class_labels required for classification")

        class_to_idx = {label: i for i, label in enumerate(class_labels)}

        votes = np.zeros((X.shape[0], len(class_labels)), dtype=float)

        for uri in request.tree_artifact_uris:
            model = self._load_model(uri)
            pred = model.predict(X)

            for row_idx, label in enumerate(pred):
                votes[row_idx, class_to_idx[str(label)]] += 1.0

        return votes

    # --------------------------------------------------
    # Regression
    # --------------------------------------------------

    def _predict_regression(self, X, request):
        sums = np.zeros((X.shape[0], 1), dtype=float)

        for uri in request.tree_artifact_uris:
            model = self._load_model(uri)
            pred = model.predict(X)
            sums[:, 0] += pred

        return sums

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _load_model(self, uri):
        # opzionale: passare da ArtifactStore
        return joblib.load(uri)

    def _matrix_from_proto(self, msg):
        arr = np.asarray(msg.values, dtype=float)
        if msg.n_rows * msg.n_cols != arr.size:
            raise ValueError("DenseMatrix shape mismatch")
        return arr.reshape(msg.n_rows, msg.n_cols)