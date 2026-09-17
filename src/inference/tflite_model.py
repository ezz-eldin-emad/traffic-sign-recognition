"""TensorFlow Lite inference with explicit quantization handling."""

from pathlib import Path

import numpy as np
import tensorflow as tf


class TFLiteClassifier:
    """Run a TFLite image classifier and return float probabilities."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self._refresh_tensor_details()

    def _refresh_tensor_details(self) -> None:
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    @staticmethod
    def _quantize(values: np.ndarray, details: dict) -> np.ndarray:
        dtype = details["dtype"]
        if not np.issubdtype(dtype, np.integer):
            return values.astype(dtype, copy=False)

        scale, zero_point = details["quantization"]
        if scale == 0:
            raise ValueError("Quantized TFLite input is missing a scale.")
        quantized = np.rint(values / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)

    @staticmethod
    def _dequantize(values: np.ndarray, details: dict) -> np.ndarray:
        if not np.issubdtype(values.dtype, np.integer):
            return values.astype(np.float32, copy=False)

        scale, zero_point = details["quantization"]
        if scale == 0:
            raise ValueError("Quantized TFLite output is missing a scale.")
        return (values.astype(np.float32) - zero_point) * scale

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict a batch of RGB images in the model's original value range."""
        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim != 4:
            raise ValueError(f"Expected a rank-4 image batch, got {batch.shape}.")

        required_shape = list(batch.shape)
        if self.input_details["shape"].tolist() != required_shape:
            self.interpreter.resize_tensor_input(
                self.input_details["index"], required_shape, strict=False
            )
            self.interpreter.allocate_tensors()
            self._refresh_tensor_details()

        self.interpreter.set_tensor(
            self.input_details["index"],
            self._quantize(batch, self.input_details),
        )
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details["index"])
        probabilities = self._dequantize(raw_output, self.output_details)
        # Integer softmax outputs are rounded to fixed bins, so their values can
        # sum slightly above or below one after dequantization.
        totals = probabilities.sum(axis=1, keepdims=True)
        return np.divide(
            probabilities,
            totals,
            out=np.zeros_like(probabilities),
            where=totals > 0,
        )
