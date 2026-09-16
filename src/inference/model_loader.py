from __future__ import annotations

import __main__
import joblib

from src.config import MODEL_REGISTRY
from src.models.classical_ml import (
    FlattenTransformer,
    HOGExtractor,
    ImagePreprocessor,
)


def _load_classical_artifact(path):
    """Load notebook and project classical artifacts across pickle sessions."""
    # Notebook-defined transformers are recorded as __main__.ClassName.
    # Register the project implementations before unpickling those artifacts.
    for name, cls in {
        "ImagePreprocessor": ImagePreprocessor,
        "HOGExtractor": HOGExtractor,
        "FlattenTransformer": FlattenTransformer,
    }.items():
        setattr(__main__, name, cls)
    return joblib.load(path)


def load_model_by_name(model_name: str):
    """Load one saved inference model from the central model registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    config = MODEL_REGISTRY[model_name]
    if not config["path"].is_file():
        raise FileNotFoundError(f"Model file not found: {config['path']}")

    if config["type"] == "cnn":
        from tensorflow import keras

        return keras.models.load_model(config["path"], compile=False)
    if config["type"] == "tflite":
        from src.inference.tflite_model import TFLiteClassifier

        return TFLiteClassifier(config["path"])
    if config["type"] == "ml":
        artifact = _load_classical_artifact(config["path"])
        if isinstance(artifact, dict):
            model = artifact.get("model")
        else:
            model = artifact
        if model is None:
            raise ValueError(
                f"Classical artifact for '{model_name}' must contain a "
                "'model' pipeline and calibrated 'threshold'."
            )
        return model
    raise ValueError(f"'{model_name}' is not an inference model.")


def load_classical_threshold(model_name: str) -> float:
    """Load the calibrated threshold saved by the classical ML notebook."""
    if model_name not in MODEL_REGISTRY or MODEL_REGISTRY[model_name]["type"] != "ml":
        raise ValueError(f"'{model_name}' is not a classical ML model.")
    path = MODEL_REGISTRY[model_name]["path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Classical model artifact is missing: {path}. Run the final cell "
            "of notebooks/02_classical_ml.ipynb first."
        )
    artifact = _load_classical_artifact(path)
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Classical artifact for '{model_name}' has no embedded threshold."
        )
    try:
        threshold = float(artifact["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"No valid calibrated threshold found inside '{path}'."
        ) from exc
    if threshold < 0:
        raise ValueError(f"Invalid negative threshold for '{model_name}'.")
    return threshold


def load_custom_cnn() -> keras.Model:
    """Backward-compatible loader for the Custom CNN."""
    return load_model_by_name("custom_cnn")


def load_mobilenet() -> keras.Model:
    """Backward-compatible loader for MobileNetV3-Small."""
    return load_model_by_name("mobilenet_v3_small_gtsrb")


def load_tflite_model(model_name: str) -> TFLiteClassifier:
    """Load a registered TFLite model with its quantization metadata."""
    from src.inference.tflite_model import TFLiteClassifier

    model = load_model_by_name(model_name)
    if not isinstance(model, TFLiteClassifier):
        raise ValueError(f"'{model_name}' is not a TFLite model.")
    return model
