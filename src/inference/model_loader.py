from tensorflow import keras

from src.config import MODEL_REGISTRY
from src.inference.tflite_model import TFLiteClassifier


def load_model_by_name(model_name: str) -> keras.Model | TFLiteClassifier:
    """Load one saved inference model from the central model registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    config = MODEL_REGISTRY[model_name]
    if not config["path"].is_file():
        raise FileNotFoundError(f"Model file not found: {config['path']}")

    if config["type"] == "cnn":
        return keras.models.load_model(config["path"], compile=False)
    if config["type"] == "tflite":
        return TFLiteClassifier(config["path"])
    raise ValueError(f"'{model_name}' is not an inference model.")


def load_custom_cnn() -> keras.Model:
    """Backward-compatible loader for the Custom CNN."""
    return load_model_by_name("custom_cnn")


def load_mobilenet() -> keras.Model:
    """Backward-compatible loader for MobileNetV3-Small."""
    return load_model_by_name("mobilenet_v3_small_gtsrb")


def load_tflite_model(model_name: str) -> TFLiteClassifier:
    """Load a registered TFLite model with its quantization metadata."""
    model = load_model_by_name(model_name)
    if not isinstance(model, TFLiteClassifier):
        raise ValueError(f"'{model_name}' is not a TFLite model.")
    return model
