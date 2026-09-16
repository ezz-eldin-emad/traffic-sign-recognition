from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.config import BATCH_SIZE, NUM_CLASSES
from src.data.dataset_loader import load_datasets
from src.utils.evaluation_utils import (
    get_report_dir,
    save_evaluation_artifacts,
)
from src.utils.metrics_utils import calculate_macro_f1, calculate_top_k_accuracy
from src.utils.model_utils import get_dl_model_complexity, get_ml_model_complexity



def collect_cnn_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true labels and prediction probabilities."""
    y_true = []
    y_prob = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_prob.extend(predictions)

    return np.asarray(y_true), np.asarray(y_prob)



def evaluate_cnn(
    model_name: str,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    model_configs: dict,
    reports_dir: Path,
) -> dict:
    """Evaluate and save all results for a Keras model."""
    config = model_configs[model_name]
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(model_path, compile=False)
    complexity = get_dl_model_complexity(model, model_path)

    y_true, y_prob = collect_cnn_predictions(model, test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    top1_accuracy = calculate_top_k_accuracy(y_true, y_prob, k=1)
    top5_accuracy = calculate_top_k_accuracy(y_true, y_prob, k=5)
    macro_f1 = calculate_macro_f1(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=class_names,
        digits=4,
    )
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
    )

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "model_type": "deep_learning",
        "test_samples": int(len(y_true)),
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_f1": macro_f1,
        **complexity,
    }

    report_dir = get_report_dir(reports_dir, model_name)
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

    _print_cnn_results(
        config["display_name"],
        report_dir,
        y_true,
        complexity,
        top1_accuracy,
        top5_accuracy,
        macro_f1,
        report,
    )

    return metrics



def load_ml_test_data(
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load test features and labels used by classical ML models."""
    raise NotImplementedError(
        "The ML test-feature loader is not connected yet."
    )



def evaluate_ml(
    model_name: str,
    model_configs: dict,
    reports_dir: Path,
) -> dict:
    """Evaluate and save all results for a classical ML model."""
    config = model_configs[model_name]
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    X_test, y_true, class_names = load_ml_test_data(model_name)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    top1_accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = calculate_macro_f1(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
    )
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names)),
    )

    complexity = get_ml_model_complexity(model_name, model, model_path)

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "model_type": "classical_ml",
        "test_samples": int(len(y_true)),
        "top1_accuracy": top1_accuracy,
        "macro_f1": macro_f1,
        **complexity,
    }

    if y_prob is not None:
        metrics["top5_accuracy"] = calculate_top_k_accuracy(
            y_true,
            y_prob,
            k=5,
        )

    report_dir = get_report_dir(reports_dir, model_name)
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

    _print_ml_results(
        config["display_name"],
        report_dir,
        y_true,
        complexity,
        top1_accuracy,
        metrics.get("top5_accuracy"),
        macro_f1,
        report,
    )

    return metrics



def evaluate_model(
    model_name: str,
    model_configs: dict,
    reports_dir: Path,
) -> dict:
    """Evaluate one model based on its configured type."""
    config = model_configs[model_name]

    if config["type"] == "cnn":
        _, _, test_ds, class_names = load_datasets(
            image_size=config["image_size"],
            batch_size=BATCH_SIZE,
        )
        return evaluate_cnn(
            model_name,
            test_ds,
            class_names,
            model_configs,
            reports_dir,
        )

    if config["type"] == "ml":
        return evaluate_ml(
            model_name,
            model_configs,
            reports_dir,
        )

    raise ValueError(f"Unsupported model type: {config['type']}")



def _print_cnn_results(
    display_name: str,
    report_dir: Path,
    y_true: np.ndarray,
    complexity: dict,
    top1_accuracy: float,
    top5_accuracy: float,
    macro_f1: float,
    report: str,
) -> None:
    print("\n" + "=" * 70)
    print(f"{display_name} Test Results")
    print("=" * 70)
    print(f"Test samples:   {len(y_true):,}")
    print(f"Model Size:     {complexity['model_size']}")
    print(f"Total Params:   {complexity['total_parameters']:,}")
    print(f"Trainable:      {complexity['trainable_parameters']:,}")
    print(f"Non-trainable:  {complexity['non_trainable_parameters']:,}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print("\nClassification Report")
    print("-" * 70)
    print(report)
    print(f"\nReports saved to: {report_dir}")



def _print_ml_results(
    display_name: str,
    report_dir: Path,
    y_true: np.ndarray,
    complexity: dict,
    top1_accuracy: float,
    top5_accuracy: float | None,
    macro_f1: float,
    report: str,
) -> None:
    print("\n" + "=" * 70)
    print(f"{display_name} Test Results")
    print("=" * 70)
    print(f"Test samples:   {len(y_true):,}")
    print(f"Model Size:     {complexity['model_size']}")

    if "support_vectors" in complexity:
        print(f"Support Vectors: {complexity['support_vectors']:,}")
    if "n_features" in complexity:
        print(f"Features:        {complexity['n_features']:,}")
    if "n_estimators" in complexity:
        print(f"Trees:           {complexity['n_estimators']:,}")
    if "total_nodes" in complexity:
        print(f"Total Nodes:     {complexity['total_nodes']:,}")
    if "average_nodes_per_tree" in complexity:
        print(f"Avg Nodes/Tree:  {complexity['average_nodes_per_tree']:.2f}")
    if "max_nodes_per_tree" in complexity:
        print(f"Max Nodes/Tree:  {complexity['max_nodes_per_tree']:,}")

    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    if top5_accuracy is not None:
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print("\nClassification Report")
    print("-" * 70)
    print(report)
    print(f"\nReports saved to: {report_dir}")
