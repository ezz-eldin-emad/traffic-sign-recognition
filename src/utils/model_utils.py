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

    estimator = (
        model.named_steps.get("clf", model)
        if hasattr(model, "named_steps")
        else model
    )

    if model_name == "svm":
        if hasattr(estimator, "n_features_in_"):
            complexity["n_features"] = int(estimator.n_features_in_)
        if hasattr(estimator, "support_vectors_"):
            complexity["support_vectors"] = int(
                estimator.support_vectors_.shape[0]
            )
            complexity["n_features"] = int(
                estimator.support_vectors_.shape[1]
            )

        if hasattr(estimator, "n_support_"):
            complexity["support_vectors_per_class"] = (
                estimator.n_support_.tolist()
            )

    elif model_name == "random_forest":
        if hasattr(estimator, "n_estimators"):
            complexity["n_estimators"] = int(estimator.n_estimators)

        if hasattr(estimator, "estimators_"):
            node_counts = [
                tree.tree_.node_count
                for tree in estimator.estimators_
            ]

            if node_counts:
                complexity["total_nodes"] = int(sum(node_counts))
                complexity["average_nodes_per_tree"] = float(
                    np.mean(node_counts)
                )
                complexity["max_nodes_per_tree"] = int(max(node_counts))

    return complexity
