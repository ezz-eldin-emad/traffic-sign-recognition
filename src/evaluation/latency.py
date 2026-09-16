import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.inference.tflite_model import TFLiteClassifier


def benchmark_ml_latency(
    model,
    sample: np.ndarray,
    warmup_runs: int = 10,
    measured_runs: int = 100,
) -> dict:
    """Measure single-image inference latency for a scikit-learn pipeline."""
    for _ in range(warmup_runs):
        model.predict(sample)

    durations = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        model.predict(sample)
        durations.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(durations)
    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(durations),
        "p95_ms": sorted(durations)[int(0.95 * len(durations)) - 1],
        "throughput_images_per_second": 1000.0 / mean_ms,
        "runtime": "scikit-learn",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def benchmark_latency(
    model: tf.keras.Model,
    sample: tf.Tensor,
    warmup_runs: int = 10,
    measured_runs: int = 100,
) -> dict:
    """Measure single-image model inference latency after warm-up."""
    for _ in range(warmup_runs):
        model(sample, training=False).numpy()

    durations = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        model(sample, training=False).numpy()
        durations.append((time.perf_counter() - start) * 1000)

    median_ms = statistics.median(durations)
    mean_ms = statistics.mean(durations)
    p95_ms = sorted(durations)[int(0.95 * len(durations)) - 1]

    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "throughput_images_per_second": 1000.0 / mean_ms,
        "device": ", ".join(device.name for device in tf.config.list_logical_devices()),
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def benchmark_tflite_latency(
    model: TFLiteClassifier,
    sample: tf.Tensor,
    warmup_runs: int = 10,
    measured_runs: int = 100,
) -> dict:
    """Measure single-image TensorFlow Lite inference latency after warm-up."""
    image = sample.numpy()
    for _ in range(warmup_runs):
        model.predict(image)

    durations = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        model.predict(image)
        durations.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(durations)
    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(durations),
        "p95_ms": sorted(durations)[int(0.95 * len(durations)) - 1],
        "throughput_images_per_second": 1000.0 / mean_ms,
        "runtime": "TensorFlow Lite",
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def save_latency_result(result: dict, report_dir: Path) -> None:
    """Save one model's latency benchmark."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latency.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
