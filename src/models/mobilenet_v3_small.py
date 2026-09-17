from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from src.utils.data_augmentation import data_augmentation_mobilenet

def build_mobilenet_model(
    num_classes: int,
    train_base_model: bool = False,
) -> tuple[keras.Model, keras.Model]:
    """Build the notebook's MobileNetV3-Small transfer-learning model."""

    base_model = MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = train_base_model

    model = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 3)),
            data_augmentation_mobilenet,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mobilenet_v3_small_gtsrb",
    )

    return model, base_model