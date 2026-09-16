import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROBUSTNESS_LEVELS = (1, 2, 3)


def _blur(images: tf.Tensor, level: int) -> tf.Tensor:
    sigma = float(level)
    radius = int(np.ceil(3 * sigma))
    coordinates = tf.range(-radius, radius + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d /= tf.reduce_sum(kernel_1d)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = tf.tile(kernel_2d[:, :, None, None], [1, 1, 3, 1])

    padding = [[0, 0], [radius, radius], [radius, radius], [0, 0]]
    padded = tf.pad(images, padding, mode="REFLECT")
    return tf.nn.depthwise_conv2d(
        padded,
        kernel,
        strides=[1, 1, 1, 1],
        padding="VALID",
    )


def _illumination(images: tf.Tensor, level: int) -> tf.Tensor:
    # The three fixed conditions are increasingly severe darkening.
    factors = {1: 0.85, 2: 0.65, 3: 0.45}
    return tf.clip_by_value(images * factors[level], 0.0, 255.0)


def _perspective(images: tf.Tensor, level: int) -> tf.Tensor:
    # A deterministic projective warp; coefficients are in pixel coordinates.
    strength = {1: 0.00035, 2: 0.0007, 3: 0.00105}[level]
    transforms = tf.tile(
        tf.constant(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, strength, strength]],
            dtype=tf.float32,
        ),
        [tf.shape(images)[0], 1],
    )
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )


def transform_images(
    images: tf.Tensor,
    transform_name: str,
    level: int,
) -> tf.Tensor:
    """Apply one deterministic robustness transformation."""
    if level not in ROBUSTNESS_LEVELS:
        raise ValueError(f"Robustness level must be one of {ROBUSTNESS_LEVELS}.")

    transforms = {
        "blur": _blur,
        "illumination": _illumination,
        "perspective": _perspective,
    }
    if transform_name not in transforms:
        raise ValueError(f"Unknown robustness transform: {transform_name}")
    return transforms[transform_name](images, level)


def save_robustness_results(
    rows: list[dict],
    report_dir: Path,
) -> None:
    """Save detailed robustness rows and a compact summary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "robustness.csv"
    fields = ["transform", "level", "test_samples", "top1_accuracy", "top5_accuracy", "macro_f1"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "conditions": len(rows),
        "mean_top1_accuracy": float(np.mean([row["top1_accuracy"] for row in rows])),
        "mean_top5_accuracy": float(np.mean([row["top5_accuracy"] for row in rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
    }
    (report_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
