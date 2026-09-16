from tensorflow import keras

from src.config import MODEL_REGISTRY


def load_model_by_name(model_name: str) -> keras.Model:
    """Load one saved Keras model from the central model registry."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    config = MODEL_REGISTRY[model_name]
    if config["type"] != "cnn":
        raise ValueError(f"'{model_name}' is not a Keras model.")
    if not config["path"].is_file():
        raise FileNotFoundError(f"Model file not found: {config['path']}")

    return keras.models.load_model(config["path"], compile=False)


def load_custom_cnn() -> keras.Model:
    """Backward-compatible loader for the Custom CNN."""
    return load_model_by_name("custom_cnn")


def load_mobilenet() -> keras.Model:
    """Backward-compatible loader for MobileNetV3-Small."""
    return load_model_by_name("mobilenet_v3_small_gtsrb")
