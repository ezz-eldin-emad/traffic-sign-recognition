"""Classical GTSRB models and the preprocessing contract used by the notebook."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import (
    CLASSICAL_N_JOBS,
    IMG_SIZE_CLASSICAL,
    RF_ESTIMATORS,
    SEED,
    SVM_C,
    SVM_MAX_ITER,
    SVM_TOL,
)


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """Convert BGR images to equalized 100x100 grayscale images."""

    def __init__(self, size: tuple[int, int] = IMG_SIZE_CLASSICAL):
        self.size = tuple(size)

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[np.ndarray]) -> np.ndarray:
        size = (self.size, self.size) if isinstance(self.size, int) else self.size
        processed = []
        for image in X:
            image = np.asarray(image)
            resized = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
            resized = np.clip(resized, 0, 255).astype(np.uint8)
            if resized.ndim == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            processed.append(cv2.equalizeHist(gray))
        if not processed:
            return np.empty((0, *tuple(size)), dtype=np.uint8)
        return np.stack(processed).astype(np.uint8)


class HOGExtractor(BaseEstimator, TransformerMixin):
    """Extract the notebook's HOG descriptor from grayscale images."""

    def __init__(
        self,
        pixels_per_cell: tuple[int, int] = (16, 16),
        n_jobs: int = CLASSICAL_N_JOBS,
    ):
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X: Iterable[np.ndarray]) -> np.ndarray:
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self._extract_one)(image) for image in X
        )
        return np.asarray(features, dtype=np.float32)

    def _extract_one(self, image: np.ndarray) -> np.ndarray:
        return hog(
            image,
            orientations=9,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Flatten equalized grayscale images for the Random Forest baseline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X).reshape(len(X), -1)


def build_svm_pipeline() -> Pipeline:
    """Build the HOG + StandardScaler + LinearSVC pipeline."""
    return Pipeline(
        [
            ("preprocess", ImagePreprocessor()),
            ("hog", HOGExtractor()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=SVM_C,
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=SVM_MAX_ITER,
                    tol=SVM_TOL,
                    dual="auto",
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Build the equalized-grayscale + flatten + Random Forest pipeline."""
    return Pipeline(
        [
            ("preprocess", ImagePreprocessor()),
            ("flatten", FlattenTransformer()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_ESTIMATORS,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=CLASSICAL_N_JOBS,
                ),
            ),
        ]
    )


def build_classical_pipeline(model_name: str) -> Pipeline:
    """Return the trainable classical pipeline for a registered model name."""
    builders = {
        "svm": build_svm_pipeline,
        "random_forest": build_random_forest_pipeline,
    }
    try:
        return builders[model_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported classical model: {model_name}") from exc
