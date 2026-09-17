import argparse
import joblib

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
    MODEL_REGISTRY,
    CLASSICAL_ML_MODELS,
    TRAINABLE_MODELS,
)
from src.data.classical_dataset import load_classical_training_data
from src.data.dataset_loader import load_datasets
from src.models.custom_cnn import build_custom_cnn
from src.models.classical_ml import build_classical_pipeline
from src.models.mobilenet_v3_small import build_mobilenet_model


MODEL_CONFIGS = {
    name: {
        **MODEL_REGISTRY[name],
        "output": MODEL_REGISTRY[name]["path"].name,
    }
    for name in TRAINABLE_MODELS
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


def _scores_for_classical_model(model, images: object) -> object:
    """Return ranking scores for validation Top-k metrics."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(images)
    return model.decision_function(images)


def train_classical(model_name: str) -> None:
    """Train and save one notebook-compatible classical ML pipeline."""
    X_train, y_train, X_val, y_val = load_classical_training_data()
    model = build_classical_pipeline(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    scores = _scores_for_classical_model(model, X_val)
    top5 = (
        (scores.argsort(axis=1)[:, -5:] == y_val[:, None]).any(axis=1).mean()
    )
    print(f"Validation accuracy ({model_name}): {(y_pred == y_val).mean():.4f}")
    print(f"Validation Top-5 accuracy ({model_name}): {top5:.4f}")

    output_path = MODEL_REGISTRY[model_name]["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved {model_name} to {output_path}")


def train(model_name: str) -> None:
    tf.keras.utils.set_random_seed(SEED)
    if model_name in CLASSICAL_ML_MODELS:
        train_classical(model_name)
        return

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
