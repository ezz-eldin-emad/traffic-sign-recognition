from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.config import EVAL_BATCH_SIZE, NUM_CLASSES
from src.data.dataset_loader import load_datasets
from src.evaluation.latency import benchmark_latency, save_latency_result
from src.evaluation.robustness import (
    ROBUSTNESS_LEVELS,
    save_robustness_results,
    transform_images,
)
from src.utils.evaluation_utils import get_report_dir, save_evaluation_artifacts
from src.utils.metrics_utils import calculate_macro_f1, calculate_top_k_accuracy
from src.utils.model_utils import get_dl_model_complexity, get_ml_model_complexity


def collect_cnn_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    transform_name: str | None = None,
    level: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect labels and probabilities, optionally after one transformation."""
    y_true = np.concatenate(
        [labels.numpy() for _, labels in dataset],
        axis=0,
    )
    prediction_dataset = dataset
    if transform_name is not None:
        prediction_dataset = dataset.map(
            lambda images, labels: (
                transform_images(images, transform_name, level or 1),
                labels,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    y_prob = model.predict(prediction_dataset, verbose=0)
    return y_true, np.asarray(y_prob)


def _calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "test_samples": int(len(y_true)),
        "top1_accuracy": calculate_top_k_accuracy(y_true, y_prob, k=1),
        "top5_accuracy": calculate_top_k_accuracy(y_true, y_prob, k=5),
        "macro_f1": calculate_macro_f1(y_true, y_pred),
        "y_pred": y_pred,
    }


def _evaluate_robustness(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    report_dir: Path,
) -> dict:
    rows = []
    for transform_name in ("blur", "illumination", "perspective"):
        for level in ROBUSTNESS_LEVELS:
            print(f"  Robustness: {transform_name} level {level}/3", flush=True)
            y_true, y_prob = collect_cnn_predictions(
                model,
                test_ds,
                transform_name=transform_name,
                level=level,
            )
            metrics = _calculate_metrics(y_true, y_prob)
            rows.append(
                {
                    "transform": transform_name,
                    "level": level,
                    "test_samples": metrics["test_samples"],
                    "top1_accuracy": metrics["top1_accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    save_robustness_results(rows, report_dir)
    return {
        "robustness_mean_top1": float(np.mean([row["top1_accuracy"] for row in rows])),
        "robustness_mean_top5": float(np.mean([row["top5_accuracy"] for row in rows])),
        "robustness_mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
    }


def evaluate_cnn(
    model_name: str,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    model_configs: dict,
    reports_dir: Path,
) -> dict:
    """Evaluate one saved Keras model and write all post-training artifacts."""
    config = model_configs[model_name]
    model_path = config["path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(model_path, compile=False)
    report_dir = get_report_dir(reports_dir, model_name)
    y_true, y_prob = collect_cnn_predictions(model, test_ds)
    metrics = _calculate_metrics(y_true, y_prob)
    y_pred = metrics.pop("y_pred")
    complexity = get_dl_model_complexity(model, model_path)

    robustness = _evaluate_robustness(model, test_ds, report_dir)
    sample_images, _ = next(iter(test_ds.take(1)))
    latency = benchmark_latency(model, sample_images[:1])
    save_latency_result(latency, report_dir)

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "model_type": "deep_learning",
        **metrics,
        **complexity,
        **robustness,
        "latency_mean_ms": latency["mean_ms"],
        "latency_median_ms": latency["median_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "throughput_images_per_second": latency["throughput_images_per_second"],
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=class_names,
        digits=4,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    save_evaluation_artifacts(
        reports_dir=reports_dir,
        model_name=model_name,
        metrics=metrics,
        report=report,
        confusion=confusion,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
    )

    _print_results(config["display_name"], metrics, report_dir)
    return metrics


def load_ml_test_data(model_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load test features for a classical ML model when that pipeline is ready."""
    raise NotImplementedError(
        f"The test-feature loader for {model_name} is not connected yet."
    )


def evaluate_ml(model_name: str, model_configs: dict, reports_dir: Path) -> dict:
    """Evaluate one classical ML model when its artifacts are available."""
    config = model_configs[model_name]
    model_path = config["path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    X_test, y_true, class_names = load_ml_test_data(model_name)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "model_type": "classical_ml",
        "test_samples": int(len(y_true)),
        "top1_accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": calculate_macro_f1(y_true, y_pred),
        **get_ml_model_complexity(model_name, model, model_path),
    }
    if y_prob is not None:
        metrics["top5_accuracy"] = calculate_top_k_accuracy(y_true, y_prob, k=5)

    save_evaluation_artifacts(
        reports_dir=reports_dir,
        model_name=model_name,
        metrics=metrics,
        report=report,
        confusion=confusion,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
    )
    return metrics


def evaluate_model(model_name: str, model_configs: dict, reports_dir: Path) -> dict:
    """Evaluate one model according to its registry type."""
    config = model_configs[model_name]
    if config["type"] == "cnn":
        _, _, test_ds, class_names = load_datasets(
            image_size=config["image_size"],
            batch_size=EVAL_BATCH_SIZE,
        )
        return evaluate_cnn(model_name, test_ds, class_names, model_configs, reports_dir)
    if config["type"] == "ml":
        return evaluate_ml(model_name, model_configs, reports_dir)
    raise ValueError(f"Unsupported model type: {config['type']}")


def _print_results(display_name: str, metrics: dict, report_dir: Path) -> None:
    print(f"\n{display_name}")
    print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"  Top-5 accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Latency p95:    {metrics['latency_p95_ms']:.2f} ms")
    print(f"  Reports:        {report_dir}")
