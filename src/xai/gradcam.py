import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_gradcam_heatmap(
    image: tf.Tensor,
    model: keras.Model,
    target_layer: keras.layers.Layer,
    class_index: int | None = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a target class.

    Args:
        image: Input image with shape (1, H, W, 3).
        model: Trained Keras classification model.
        target_layer: Convolutional layer used for Grad-CAM.
        class_index: Target class index. If None, use predicted class.

    Returns:
        Normalized heatmap with values in the range [0, 1].
    """

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            image,
            training=False,
        )

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs,
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy()