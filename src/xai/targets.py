from tensorflow import keras


def get_custom_cnn_target_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the convolutional layer used for Custom CNN Grad-CAM."""

    return model.get_layer("conv2d_18")


def get_mobilenet_target_layer(model: keras.Model) -> keras.layers.Layer:
    """Return the final convolutional layer used for MobileNet Grad-CAM."""

    base_model = model.get_layer("MobileNetV3Small")

    return base_model.get_layer("conv_1")