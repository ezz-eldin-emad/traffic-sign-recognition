from tensorflow import keras
from tensorflow.keras import layers

data_augmentation_custom_cnn = keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="data_augmentation_custom_cnn",
)
data_augmentation_mobilenet = keras.Sequential(
    [
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation_mobilenet",
)
