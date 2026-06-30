from __future__ import annotations

import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from common.contracts import ModelManifest
from common.enums import ModelStatus


SUPPORTED_EXPORT_FORMATS = {"pickle", "pkl", "joblib"}


@dataclass(slots=True)
class ModelExportResult:
    model_id: str
    export_format: str
    artifact_uri: str
    file_path: Path
    filename: str
    size_bytes: int
    created: bool


class ModelExportService:
    """
    Costruisce un pacchetto scaricabile del modello finale. implementazione opzionale per il download del modello allenato

    Formato corrente:
    - Pickle/Joblib: zip contenente model.joblib, manifest.json, metadata.json
      e uno script minimale di inferenza locale.

    Nota:
    il modello esportato e' un vero RandomForestClassifier/Regressor sklearn
    ricostruito dagli alberi DecisionTree gia' addestrati dai worker.
    """

    def __init__(self, model_repository, artifact_store) -> None:
        self.model_repository = model_repository
        self.artifact_store = artifact_store
        self.artifact_root = Path(artifact_store.layout.root)

    def export_model(
        self,
        model_id: str,
        export_format: str = "pickle",
        overwrite: bool = False,
    ) -> ModelExportResult:
        normalized_format = self._normalize_format(export_format)

        manifest = self.model_repository.load(model_id)
        if manifest is None:
            raise ValueError(f"Model '{model_id}' not found")

        if manifest.status != ModelStatus.READY:
            raise ValueError(
                f"Model '{model_id}' is not READY. Current status: {manifest.status.value}"
            )

        export_dir = self.artifact_root / "models" / model_id / "exports" / normalized_format
        bundle_path = export_dir / f"{model_id}_{normalized_format}.zip"

        if bundle_path.exists() and not overwrite:
            return ModelExportResult(
                model_id=model_id,
                export_format=normalized_format,
                artifact_uri=self._to_file_uri(bundle_path),
                file_path=bundle_path,
                filename=bundle_path.name,
                size_bytes=bundle_path.stat().st_size,
                created=False,
            )

        tmp_dir = export_dir / ".tmp_export"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            trees = self._load_trees(manifest)
            sklearn_model = self._build_sklearn_forest(manifest, trees)

            model_path = tmp_dir / "model.joblib"
            manifest_path = tmp_dir / "manifest.json"
            metadata_path = tmp_dir / "metadata.json"
            example_path = tmp_dir / "local_inference_example.py"
            readme_path = tmp_dir / "README_LOCAL_INFERENCE.md"

            joblib.dump(sklearn_model, model_path)

            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)

            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(self._build_metadata(manifest), handle, indent=2, sort_keys=True)

            example_path.write_text(
                self._local_inference_example(),
                encoding="utf-8",
            )
            readme_path.write_text(
                self._readme_text(manifest),
                encoding="utf-8",
            )

            tmp_bundle_path = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
            if tmp_bundle_path.exists():
                tmp_bundle_path.unlink()

            with zipfile.ZipFile(tmp_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in [model_path, manifest_path, metadata_path, example_path, readme_path]:
                    archive.write(path, arcname=path.name)

            tmp_bundle_path.replace(bundle_path)

            return ModelExportResult(
                model_id=model_id,
                export_format=normalized_format,
                artifact_uri=self._to_file_uri(bundle_path),
                file_path=bundle_path,
                filename=bundle_path.name,
                size_bytes=bundle_path.stat().st_size,
                created=True,
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _normalize_format(self, export_format: str) -> str:
        value = (export_format or "pickle").strip().lower()
        if value not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export format '{export_format}'. "
                f"Supported formats: {sorted(SUPPORTED_EXPORT_FORMATS)}"
            )

        if value in {"pickle", "pkl"}:
            return "pickle"
        return "joblib"

    def _load_trees(self, manifest: ModelManifest) -> list[object]:
        if not manifest.tree_artifacts:
            raise ValueError(f"Model '{manifest.model_id}' has no tree artifacts")

        trees = []
        for artifact in sorted(manifest.tree_artifacts, key=lambda item: item.tree_index):
            artifact_path = self._resolve_artifact_path(artifact.artifact_uri)
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Tree artifact not found for tree_id={artifact.tree_id}: {artifact_path}"
                )
            trees.append(joblib.load(artifact_path))

        return trees

    def _resolve_artifact_path(self, uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(urlparse(uri).path)

        path = Path(uri)
        if path.is_absolute():
            return path

        return self.artifact_root / path

    def _build_sklearn_forest(
        self,
        manifest: ModelManifest,
        trees: list[object],
    ):
        fc = manifest.forest_config

        if manifest.model_type == "classification":
            forest = RandomForestClassifier(
                n_estimators=len(trees),
                criterion=fc.criterion,
                max_depth=fc.max_depth,
                max_features=fc.max_features,
                min_samples_split=fc.min_samples_split,
                min_samples_leaf=fc.min_samples_leaf,
                bootstrap=fc.bootstrap,
                random_state=fc.global_random_seed,
            )

            first_tree = trees[0]
            forest.estimators_ = trees
            forest.n_estimators = len(trees)
            forest.n_features_in_ = getattr(first_tree, "n_features_in_", len(manifest.feature_names))
            forest.n_outputs_ = getattr(first_tree, "n_outputs_", 1)
            forest.classes_ = getattr(first_tree, "classes_", np.asarray(manifest.class_labels))
            forest.n_classes_ = getattr(first_tree, "n_classes_", len(forest.classes_))
            if manifest.feature_names:
                forest.feature_names_in_ = np.asarray(manifest.feature_names, dtype=object)
            return forest

        if manifest.model_type == "regression":
            forest = RandomForestRegressor(
                n_estimators=len(trees),
                criterion=fc.criterion,
                max_depth=fc.max_depth,
                max_features=fc.max_features,
                min_samples_split=fc.min_samples_split,
                min_samples_leaf=fc.min_samples_leaf,
                bootstrap=fc.bootstrap,
                random_state=fc.global_random_seed,
            )

            first_tree = trees[0]
            forest.estimators_ = trees
            forest.n_estimators = len(trees)
            forest.n_features_in_ = getattr(first_tree, "n_features_in_", len(manifest.feature_names))
            forest.n_outputs_ = getattr(first_tree, "n_outputs_", 1)
            if manifest.feature_names:
                forest.feature_names_in_ = np.asarray(manifest.feature_names, dtype=object)
            return forest

        raise ValueError(f"Unsupported model_type '{manifest.model_type}'")

    def _build_metadata(self, manifest: ModelManifest) -> dict:
        return {
            "model_id": manifest.model_id,
            "job_id": manifest.job_id,
            "experiment_id": manifest.experiment_id,
            "model_type": manifest.model_type,
            "target_column": manifest.target_column,
            "feature_names": manifest.feature_names,
            "class_labels": manifest.class_labels,
            "n_estimators": len(manifest.tree_artifacts),
            "forest_config": manifest.forest_config.to_dict(),
            "validation_metrics": manifest.validation_metrics.to_dict(),
            "test_metrics": manifest.test_metrics,
            "exported_at": time.time(),
        }

    def _local_inference_example(self) -> str:
        return '''from pathlib import Path
import json
import joblib
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent

model = joblib.load(BUNDLE_DIR / "model.joblib")
metadata = json.loads((BUNDLE_DIR / "metadata.json").read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]

# Esempio con CSV locale.
# Se il CSV contiene gia' le feature preparate/encodate, basta selezionare feature_names.
# Se invece contiene feature categoriche raw, get_dummies + reindex ricostruisce le colonne attese.
X_raw = pd.read_csv("input.csv")
X = pd.get_dummies(X_raw, dummy_na=True, dtype=float)
X = X.reindex(columns=feature_names, fill_value=0.0)

predictions = model.predict(X)
print(predictions.tolist())
'''

    def _readme_text(self, manifest: ModelManifest) -> str:
        return f'''# Local inference bundle

Model id: `{manifest.model_id}`
Task type: `{manifest.model_type}`
Target column: `{manifest.target_column}`

Contenuto:
- `model.joblib`: RandomForestClassifier/RandomForestRegressor sklearn esportato.
- `metadata.json`: feature names, class labels, metriche e configurazione.
- `manifest.json`: manifest completo del sistema distribuito.
- `local_inference_example.py`: esempio minimo per inferenza locale.

Uso minimo:

```python
import json
import joblib
import pandas as pd

model = joblib.load("model.joblib")
metadata = json.load(open("metadata.json"))
feature_names = metadata["feature_names"]

X_raw = pd.read_csv("input.csv")
X = pd.get_dummies(X_raw, dummy_na=True, dtype=float)
X = X.reindex(columns=feature_names, fill_value=0.0)

predictions = model.predict(X)
```
'''

    def _to_file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()
