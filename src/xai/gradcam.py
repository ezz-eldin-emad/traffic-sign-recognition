import numpy as np
import tensorflow as tf
from tensorflow import keras


def _find_target_parent(
    model: keras.Model,
    target_layer: keras.layers.Layer,
) -> tuple[int, keras.Model | None]:
    """Return the outer-layer index and nested parent for a target layer."""
    for index, layer in enumerate(model.layers):
        if layer is target_layer:
            return index, None

        if isinstance(layer, keras.Model) and any(
            child is target_layer for child in layer.layers
        ):
            return index, layer

    raise ValueError(f"Target layer '{target_layer.name}' is not in the model.")


def _build_gradcam_model(
    model: keras.Model,
    target_layer: keras.layers.Layer,
) -> keras.Model:
    """Build one connected model that returns activations and predictions."""
    input_shape = model.input_shape[1:]
    inputs = keras.Input(shape=input_shape, name="gradcam_input")
    target_index, parent = _find_target_parent(model, target_layer)

    if parent is None:
        features = inputs
        activations = None
        for layer in model.layers:
            features = layer(features, training=False)
            if layer is target_layer:
                activations = features
        if activations is None:
            raise ValueError(f"Could not connect target layer '{target_layer.name}'.")
    else:
        features = inputs
        for layer in model.layers[:target_index]:
            features = layer(features, training=False)

        nested_model = keras.Model(
            inputs=parent.inputs,
            outputs=[target_layer.output, parent.output],
            name="gradcam_feature_extractor",
        )
        activations, features = nested_model(features, training=False)

        for layer in model.layers[target_index + 1:]:
            features = layer(features, training=False)

    return keras.Model(inputs=inputs, outputs=[activations, features])


def make_gradcam_heatmap(
    image: tf.Tensor,
    model: keras.Model,
    target_layer: keras.layers.Layer,
    class_index: int | None = None,
) -> np.ndarray:
    """Generate a normalized Grad-CAM heatmap for a model prediction."""
    grad_model = _build_gradcam_model(model, target_layer)

    with tf.GradientTape() as tape:
        activations, predictions = grad_model(image, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError(f"Gradients are unavailable for '{target_layer.name}'.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(activations * weights[:, None, None, :], axis=-1)
    heatmap = tf.maximum(heatmap[0], 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, tf.zeros_like(heatmap))
    return heatmap.numpy()
