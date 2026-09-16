from pathlib import Path

import numpy as np
from tensorflow import keras

from .evaluation_utils import format_size, get_file_size



def get_dl_model_complexity(
    model: keras.Model,
    model_path: Path,
) -> dict:
    """Collect size and parameter information for a Keras model."""
    trainable_params = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(np.prod(variable.shape) for variable in model.non_trainable_weights)
    )
    total_params = trainable_params + non_trainable_params
    size_bytes = get_file_size(model_path)

    return {
        "model_size_bytes": size_bytes,
        "model_size": format_size(size_bytes),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
    }


def get_tflite_model_complexity(model_path: Path) -> dict:
    """Collect file-size metadata for a TFLite deployment artifact."""
    size_bytes = get_file_size(model_path)
    return {
        "model_size_bytes": size_bytes,
        "model_size": format_size(size_bytes),
    }



def get_ml_model_complexity(
    model_name: str,
    model,
    model_path: Path,
) -> dict:
    """Collect size and model-specific complexity information."""
    size_bytes = get_file_size(model_path)

    complexity = {
        "model_size_bytes": size_bytes,
        "model_size": format_size(size_bytes),
    }

    if model_name == "svm":
        if hasattr(model, "support_vectors_"):
            complexity["support_vectors"] = int(
                model.support_vectors_.shape[0]
            )
            complexity["n_features"] = int(
                model.support_vectors_.shape[1]
            )

        if hasattr(model, "n_support_"):
            complexity["support_vectors_per_class"] = (
                model.n_support_.tolist()
            )

    elif model_name == "random_forest":
        if hasattr(model, "n_estimators"):
            complexity["n_estimators"] = int(model.n_estimators)

        if hasattr(model, "estimators_"):
            node_counts = [
                estimator.tree_.node_count
                for estimator in model.estimators_
            ]

            if node_counts:
                complexity["total_nodes"] = int(sum(node_counts))
                complexity["average_nodes_per_tree"] = float(
                    np.mean(node_counts)
                )
                complexity["max_nodes_per_tree"] = int(max(node_counts))

    return complexity
