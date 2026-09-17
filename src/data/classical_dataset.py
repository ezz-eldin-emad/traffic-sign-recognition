"""CSV/image loading for the classical GTSRB baselines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASSICAL_VALIDATION_SPLIT,
    CLASS_NAMES,
    IMG_SIZE_CLASSICAL,
    RAW_DATA_DIR,
    SEED,
)


def _load_split_frame(data_dir: Path, csv_name: str) -> pd.DataFrame:
    csv_path = data_dir / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"GTSRB split CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    frame["AbsPath"] = frame["Path"].map(lambda value: data_dir / str(value))
    return frame


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Traffic-sign image not found: {path}")
    return cv2.resize(image, IMG_SIZE_CLASSICAL, interpolation=cv2.INTER_AREA)


def _load_images(frame: pd.DataFrame) -> np.ndarray:
    return np.stack([_read_image(path) for path in frame["AbsPath"]])


def load_classical_training_data(
    data_dir: Path = RAW_DATA_DIR,
    validation_split: float = CLASSICAL_VALIDATION_SPLIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the notebook-compatible train/validation images and labels."""
    frame = _load_split_frame(data_dir, "Train.csv")
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_split,
        stratify=frame["ClassId"],
        random_state=SEED,
    )
    return (
        _load_images(train_frame),
        train_frame["ClassId"].to_numpy(dtype=np.int64),
        _load_images(validation_frame),
        validation_frame["ClassId"].to_numpy(dtype=np.int64),
    )


def load_classical_test_data(
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the GTSRB Test.csv images in the classical model format."""
    frame = _load_split_frame(data_dir, "Test.csv")
    return (
        _load_images(frame),
        frame["ClassId"].to_numpy(dtype=np.int64),
        list(CLASS_NAMES),
    )
