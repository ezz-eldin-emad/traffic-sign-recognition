import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    BATCH_SIZE,
    CNN_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    HEAD_EPOCHS,
    IMG_SIZE_CUSTOM_CNN,
    IMG_SIZE_MOBILENET,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_CLASSES,
    SEED,
)
from src.data.dataset_loader import load_datasets
from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenet_v3_small import build_mobilenet_model


MODEL_CONFIGS = {
    "custom_cnn": {
        "image_size": IMG_SIZE_CUSTOM_CNN,
        "output": "cnn_model_customized.keras",
    },
    "mobilenet_v3_small_gtsrb": {
        "image_size": IMG_SIZE_MOBILENET,
        "output": "mobilenet_v3_small_gtsrb.keras",
    },
}


def create_callbacks(checkpoint_name: str) -> list[keras.callbacks.Callback]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / checkpoint_name,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
    ]


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_custom_cnn(train_ds, val_ds, test_ds, class_names) -> None:
    model = build_custom_cnn(len(class_names))
    compile_model(model, LEARNING_RATE)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CNN_EPOCHS,
        callbacks=create_callbacks("custom_cnn_best.keras"),
    )
    model.save(MODEL_DIR / "cnn_model_customized.keras")
    print(f"Test metrics: {model.evaluate(test_ds, return_dict=True)}")


def train_mobilenet(train_ds, val_ds, test_ds, class_names) -> None:
    model, base_model = build_mobilenet_model(len(class_names), False)
    callbacks = create_callbacks("mobilenet_v3_small_best.keras")

    compile_model(model, LEARNING_RATE)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, FINE_TUNE_LEARNING_RATE)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )
    model.save(MODEL_DIR / "mobilenet_v3_small_gtsrb.keras")
    print(f"Test metrics: {model.evaluate(test_ds, return_dict=True)}")


def train(model_name: str) -> None:
    tf.keras.utils.set_random_seed(SEED)
    config = MODEL_CONFIGS[model_name]
    train_ds, val_ds, test_ds, class_names = load_datasets(
        image_size=config["image_size"],
        batch_size=BATCH_SIZE,
    )
    print(f"Number of classes: {len(class_names)}")
    if model_name == "custom_cnn":
        train_custom_cnn(train_ds, val_ds, test_ds, class_names)
    elif model_name == "mobilenet_v3_small_gtsrb":
        train_mobilenet(train_ds, val_ds, test_ds, class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a traffic-sign model")
    parser.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()