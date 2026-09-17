import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

from src.config import (
    BATCH_SIZE,
    IMG_SIZE_MOBILENET,
    PROCESSED_TEST_DIR,
    SEED,
    TRAIN_DIR,
)


def load_datasets(
    image_size=IMG_SIZE_MOBILENET,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
):
    """Load train, validation, and test datasets."""

    train_ds = image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_ds = image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        shuffle=True,
    )

    test_ds = image_dataset_from_directory(
        PROCESSED_TEST_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names