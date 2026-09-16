from tensorflow import keras


def _last_convolution(layer: keras.layers.Layer) -> keras.layers.Layer | None:
    """Find the last Conv2D layer in a model or nested Keras model."""
    if isinstance(layer, keras.layers.Conv2D):
        return layer

    if isinstance(layer, keras.Model):
        for child in reversed(layer.layers):
            target = _last_convolution(child)
            if target is not None:
                return target

    return None


def get_target_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the last convolutional layer without relying on generated names."""
    for layer in reversed(model.layers):
        target = _last_convolution(layer)
        if target is not None:
            return target

    raise ValueError(f"No Conv2D layer found in model '{model.name}'.")


def get_custom_cnn_target_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the last convolutional layer in the Custom CNN."""
    return get_target_layer(model)


def get_mobilenet_target_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the last convolutional layer inside MobileNetV3-Small."""
    return get_target_layer(model)
