import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from src.config import (
    BATCH_SIZE,
    IMG_SIZE_CUSTOM_CNN,
    IMG_SIZE_MOBILENET,
    MODEL_DIR,
    NUM_CLASSES,
)
from src.data.dataset_loader import load_datasets


REPORTS_DIR = Path("reports")
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"


MODEL_CONFIGS = {
    "custom_cnn": {
        "type": "cnn",
        "image_size": IMG_SIZE_CUSTOM_CNN,
        "path": MODEL_DIR / "cnn_model_customized.keras",
        "display_name": "Custom CNN",
    },
    "mobilenet_v3_small_gtsrb": {
        "type": "cnn",
        "image_size": IMG_SIZE_MOBILENET,
        "path": MODEL_DIR / "mobilenet_v3_small_gtsrb.keras",
        "display_name": "MobileNetV3-Small",
    },
    "svm": {
        "type": "ml",
        "path": MODEL_DIR / "svm_model.joblib",
        "display_name": "SVM",
    },
    "random_forest": {
        "type": "ml",
        "path": MODEL_DIR / "random_forest_model.joblib",
        "display_name": "Random Forest",
    },
}


COMPARISON_FIELDS = [
    "model",
    "display_name",
    "model_type",
    "model_size",
    "model_size_bytes",
    "total_parameters",
    "trainable_parameters",
    "non_trainable_parameters",
    "support_vectors",
    "n_features",
    "n_estimators",
    "total_nodes",
    "average_nodes_per_tree",
    "max_nodes_per_tree",
    "test_samples",
    "top1_accuracy",
    "top5_accuracy",
    "macro_f1",
]


# ============================================================
# Report helpers
# ============================================================

def get_report_dir(model_name: str) -> Path:
    """Return and create the report directory for a model."""

    report_dir = REPORTS_DIR / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    return report_dir


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""

    serializable = {}

    for key, value in data.items():
        if isinstance(value, np.generic):
            serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            continue
        else:
            serializable[key] = value

    path.write_text(
        json.dumps(
            serializable,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_classification_report(
    report: str,
    path: Path,
) -> None:
    """Save the sklearn classification report."""

    path.write_text(
        report,
        encoding="utf-8",
    )


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    path: Path,
) -> None:
    """Save a confusion matrix plot."""

    figure_size = max(
        14,
        len(class_names) * 0.35,
    )

    plt.figure(
        figsize=(figure_size, figure_size),
    )

    plt.imshow(matrix)

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90,
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save true and predicted labels for later error analysis."""

    report_dir = get_report_dir(model_name)

    np.save(
        report_dir / "y_true.npy",
        y_true,
    )

    np.save(
        report_dir / "y_pred.npy",
        y_pred,
    )


def get_file_size(path: Path) -> int:
    """Return file size in bytes."""

    return path.stat().st_size


def format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable size."""

    size = float(size_bytes)
    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


# ============================================================
# Deep learning complexity
# ============================================================

def get_dl_model_complexity(
    model: keras.Model,
    model_path: Path,
) -> dict:
    """Collect size and parameter information for a Keras model."""

    trainable_params = int(
        sum(
            np.prod(variable.shape)
            for variable in model.trainable_weights
        )
    )

    non_trainable_params = int(
        sum(
            np.prod(variable.shape)
            for variable in model.non_trainable_weights
        )
    )

    total_params = (
        trainable_params
        + non_trainable_params
    )

    size_bytes = get_file_size(model_path)

    return {
        "model_size_bytes": size_bytes,
        "model_size": format_size(size_bytes),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
    }


# ============================================================
# CNN evaluation
# ============================================================

def collect_cnn_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true labels and prediction probabilities."""

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(labels.numpy())
        y_prob.extend(predictions)

    return (
        np.asarray(y_true),
        np.asarray(y_prob),
    )


def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> float:
    """Calculate top-k classification accuracy."""

    k = min(
        k,
        y_prob.shape[1],
    )

    top_k_predictions = np.argsort(
        y_prob,
        axis=1,
    )[:, -k:]

    return float(
        np.mean(
            np.any(
                top_k_predictions
                == y_true[:, None],
                axis=1,
            )
        )
    )


def evaluate_cnn(
    model_name: str,
    test_ds: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    """Evaluate and save all results for a Keras model."""

    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = keras.models.load_model(
        model_path,
        compile=False,
    )

    complexity = get_dl_model_complexity(
        model,
        model_path,
    )

    y_true, y_prob = collect_cnn_predictions(
        model,
        test_ds,
    )

    y_pred = np.argmax(
        y_prob,
        axis=1,
    )

    top1_accuracy = calculate_top_k_accuracy(
        y_true,
        y_prob,
        k=1,
    )

    top5_accuracy = calculate_top_k_accuracy(
        y_true,
        y_prob,
        k=5,
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
        )
    )

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

    report_dir = get_report_dir(model_name)

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

    save_json(
        metrics,
        report_dir / "metrics.json",
    )

    save_classification_report(
        report,
        report_dir / "classification_report.txt",
    )

    save_confusion_matrix(
        confusion,
        class_names,
        f"{config['display_name']} Test Confusion Matrix",
        report_dir / "confusion_matrix.png",
    )

    save_predictions(
        model_name,
        y_true,
        y_pred,
    )

    np.save(
        report_dir / "confusion_matrix.npy",
        confusion,
    )

    print()
    print("=" * 70)
    print(f"{config['display_name']} Test Results")
    print("=" * 70)

    print(
        f"Test samples:   "
        f"{len(y_true):,}"
    )

    print(
        f"Model Size:     "
        f"{complexity['model_size']}"
    )

    print(
        f"Total Params:   "
        f"{complexity['total_parameters']:,}"
    )

    print(
        f"Trainable:      "
        f"{complexity['trainable_parameters']:,}"
    )

    print(
        f"Non-trainable:  "
        f"{complexity['non_trainable_parameters']:,}"
    )

    print(
        f"Top-1 Accuracy: "
        f"{top1_accuracy:.4f}"
    )

    print(
        f"Top-5 Accuracy: "
        f"{top5_accuracy:.4f}"
    )

    print(
        f"Macro F1:       "
        f"{macro_f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)
    print(report)

    print()
    print(
        f"Reports saved to: "
        f"{report_dir}"
    )

    return metrics


# ============================================================
# Classical ML evaluation
# ============================================================

def load_ml_test_data(
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load test features and labels used by classical ML models.

    Must return:

        X_test
        y_test
        class_names
    """

    raise NotImplementedError(
        "The ML test-feature loader is not connected yet."
    )


def get_ml_model_complexity(
    model_name: str,
    model,
    model_path: Path,
) -> dict:
    """Collect size and model-specific complexity information."""

    size_bytes = get_file_size(model_path)

    complexity = {
        "model_size_bytes": size_bytes,
        "model_size": format_size(size_bytes),
    }

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    if model_name == "svm":
        if hasattr(
            model,
            "support_vectors_",
        ):
            complexity["support_vectors"] = int(
                model.support_vectors_.shape[0]
            )

            complexity["n_features"] = int(
                model.support_vectors_.shape[1]
            )

        if hasattr(
            model,
            "n_support_",
        ):
            complexity[
                "support_vectors_per_class"
            ] = model.n_support_.tolist()

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    elif model_name == "random_forest":
        if hasattr(
            model,
            "n_estimators",
        ):
            complexity["n_estimators"] = int(
                model.n_estimators
            )

        if hasattr(
            model,
            "estimators_",
        ):
            node_counts = [
                estimator.tree_.node_count
                for estimator in model.estimators_
            ]

            if node_counts:
                complexity["total_nodes"] = int(
                    sum(node_counts)
                )

                complexity[
                    "average_nodes_per_tree"
                ] = float(
                    np.mean(node_counts)
                )

                complexity[
                    "max_nodes_per_tree"
                ] = int(
                    max(node_counts)
                )

    return complexity


def evaluate_ml(
    model_name: str,
) -> dict:
    """Evaluate and save all results for a classical ML model."""

    import joblib

    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    X_test, y_true, class_names = load_ml_test_data(
        model_name
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = None

    if hasattr(
        model,
        "predict_proba",
    ):
        y_prob = model.predict_proba(
            X_test
        )

    top1_accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        ),
        target_names=class_names,
        digits=4,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=range(
            len(class_names)
        ),
    )

    complexity = get_ml_model_complexity(
        model_name,
        model,
        model_path,
    )

    metrics = {
        "model": model_name,
        "display_name": config["display_name"],
        "model_type": "classical_ml",
        "test_samples": int(
            len(y_true)
        ),
        "top1_accuracy": top1_accuracy,
        "macro_f1": macro_f1,
        **complexity,
    }

    if y_prob is not None:
        metrics["top5_accuracy"] = (
            calculate_top_k_accuracy(
                y_true,
                y_prob,
                k=5,
            )
        )

    report_dir = get_report_dir(
        model_name
    )

    save_json(
        metrics,
        report_dir / "metrics.json",
    )

    save_classification_report(
        report,
        report_dir / "classification_report.txt",
    )

    save_confusion_matrix(
        confusion,
        class_names,
        f"{config['display_name']} Test Confusion Matrix",
        report_dir / "confusion_matrix.png",
    )

    save_predictions(
        model_name,
        y_true,
        y_pred,
    )

    np.save(
        report_dir / "confusion_matrix.npy",
        confusion,
    )

    print()
    print("=" * 70)
    print(
        f"{config['display_name']} "
        f"Test Results"
    )
    print("=" * 70)

    print(
        f"Test samples:   "
        f"{len(y_true):,}"
    )

    print(
        f"Model Size:     "
        f"{complexity['model_size']}"
    )

    if "support_vectors" in complexity:
        print(
            f"Support Vectors: "
            f"{complexity['support_vectors']:,}"
        )

    if "n_features" in complexity:
        print(
            f"Features:        "
            f"{complexity['n_features']:,}"
        )

    if "n_estimators" in complexity:
        print(
            f"Trees:           "
            f"{complexity['n_estimators']:,}"
        )

    if "total_nodes" in complexity:
        print(
            f"Total Nodes:     "
            f"{complexity['total_nodes']:,}"
        )

    if "average_nodes_per_tree" in complexity:
        print(
            f"Avg Nodes/Tree:  "
            f"{complexity['average_nodes_per_tree']:.2f}"
        )

    if "max_nodes_per_tree" in complexity:
        print(
            f"Max Nodes/Tree:  "
            f"{complexity['max_nodes_per_tree']:,}"
        )

    print(
        f"Top-1 Accuracy: "
        f"{top1_accuracy:.4f}"
    )

    if "top5_accuracy" in metrics:
        print(
            f"Top-5 Accuracy: "
            f"{metrics['top5_accuracy']:.4f}"
        )

    print(
        f"Macro F1:       "
        f"{macro_f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)
    print(report)

    print()
    print(
        f"Reports saved to: "
        f"{report_dir}"
    )

    return metrics


# ============================================================
# Dispatcher
# ============================================================

def evaluate_model(
    model_name: str,
) -> dict:
    """Evaluate one model based on its configured type."""

    config = MODEL_CONFIGS[model_name]

    if config["type"] == "cnn":
        _, _, test_ds, class_names = (
            load_datasets(
                image_size=config["image_size"],
                batch_size=BATCH_SIZE,
            )
        )

        return evaluate_cnn(
            model_name,
            test_ds,
            class_names,
        )

    if config["type"] == "ml":
        return evaluate_ml(
            model_name
        )

    raise ValueError(
        f"Unsupported model type: "
        f"{config['type']}"
    )


# ============================================================
# Comparison report
# ============================================================

def load_existing_comparison() -> dict[str, dict]:
    """
    Load valid existing comparison rows.

    Invalid/corrupted rows are ignored so an old broken CSV
    cannot contaminate the new comparison file.
    """

    if not COMPARISON_PATH.exists():
        return {}

    existing = {}

    try:
        with COMPARISON_PATH.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                return {}

            for row in reader:
                model_name = (
                    row.get("model") or ""
                ).strip()

                # Only accept known model names.
                if model_name not in MODEL_CONFIGS:
                    continue

                # Reject obviously corrupted rows.
                if (
                    row.get("display_name")
                    == "display_name"
                    or row.get("model_type")
                    == "model_type"
                ):
                    continue

                existing[model_name] = {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in COMPARISON_FIELDS
                }

    except (
        OSError,
        csv.Error,
    ):
        return {}

    return existing


def normalize_comparison_row(
    result: dict,
) -> dict:
    """Convert metrics into a stable CSV row."""

    row = {}

    for field in COMPARISON_FIELDS:
        value = result.get(
            field,
            "",
        )

        if value is None:
            value = ""

        row[field] = value

    return row


def save_comparison(
    results: list[dict],
) -> None:
    """
    Upsert model results into model_comparison.csv.

    Running a model again replaces only that model's row.
    Different models are preserved.
    """

    if not results:
        return

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = load_existing_comparison()

    for result in results:
        model_name = result.get(
            "model"
        )

        if model_name not in MODEL_CONFIGS:
            continue

        comparison[model_name] = (
            normalize_comparison_row(
                result
            )
        )

    # Keep models in the same order as MODEL_CONFIGS.
    ordered_rows = []

    for model_name in MODEL_CONFIGS:
        if model_name in comparison:
            ordered_rows.append(
                comparison[model_name]
            )

    temp_path = COMPARISON_PATH.with_suffix(
        ".tmp"
    )

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COMPARISON_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in ordered_rows:
            writer.writerow(row)

    temp_path.replace(
        COMPARISON_PATH
    )

    print()
    print(
        f"Model comparison saved to: "
        f"{COMPARISON_PATH}"
    )


def print_comparison(
    results: list[dict],
) -> None:
    """Print a compact comparison for current run."""

    print()
    print("=" * 120)
    print("CURRENT RUN MODEL COMPARISON")
    print("=" * 120)

    print(
        f"{'Model':30}"
        f"{'Size':>12}"
        f"{'Complexity':>18}"
        f"{'Top-1':>12}"
        f"{'Top-5':>12}"
        f"{'Macro F1':>12}"
    )

    print("-" * 120)

    for result in results:
        model_type = result.get(
            "model_type"
        )

        if model_type == "deep_learning":
            total_params = result.get(
                "total_parameters"
            )

            complexity_text = (
                f"Params={total_params:,}"
                if total_params is not None
                else "N/A"
            )

        elif result["model"] == "svm":
            support_vectors = result.get(
                "support_vectors"
            )

            complexity_text = (
                f"SV={support_vectors:,}"
                if support_vectors is not None
                else "N/A"
            )

        elif result["model"] == "random_forest":
            total_nodes = result.get(
                "total_nodes"
            )

            complexity_text = (
                f"Nodes={total_nodes:,}"
                if total_nodes is not None
                else "N/A"
            )

        else:
            complexity_text = "N/A"

        top5 = result.get(
            "top5_accuracy"
        )

        top5_text = (
            f"{top5:.4f}"
            if top5 is not None
            else "N/A"
        )

        print(
            f"{result['display_name']:30}"
            f"{result['model_size']:>12}"
            f"{complexity_text:>18}"
            f"{result['top1_accuracy']:>12.4f}"
            f"{top5_text:>12}"
            f"{result['macro_f1']:>12.4f}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate traffic-sign "
            "recognition models."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            *MODEL_CONFIGS.keys(),
            "all",
        ],
        required=True,
        help="Model to evaluate.",
    )

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(
        42
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    if args.model == "all":
        model_names = list(
            MODEL_CONFIGS.keys()
        )
    else:
        model_names = [
            args.model
        ]

    for model_name in model_names:
        try:
            result = evaluate_model(
                model_name
            )

            results.append(
                result
            )

        except FileNotFoundError as exc:
            print()
            print(
                f"[SKIPPED] "
                f"{model_name}"
            )
            print(exc)

        except NotImplementedError as exc:
            print()
            print(
                f"[SKIPPED] "
                f"{model_name}"
            )
            print(exc)

    if results:
        print_comparison(
            results
        )

        save_comparison(
            results
        )


if __name__ == "__main__":
    main()
