import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> tf.Tensor:
    """Convert a PIL image to the batched float tensor expected by a model."""
    resized = image.convert("RGB").resize(image_size)
    array = np.asarray(resized, dtype=np.float32)
    return tf.convert_to_tensor(array[None, ...], dtype=tf.float32)
