from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def get_report_dir(reports_dir: Path, model_name: str) -> Path:
    """Return and create the report directory for a model."""
    report_dir = reports_dir / model_name
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
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )



def save_classification_report(report: str, path: Path) -> None:
    """Save the sklearn classification report."""
    path.write_text(report, encoding="utf-8")



def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
    path: Path,
) -> None:
    """Save a clear annotated confusion matrix plot."""


    figure_size = max(20, len(class_names) * 0.5)

    plt.figure(
        figsize=(figure_size, figure_size),
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )

    plt.title(
        title,
        fontsize=18,
    )

    plt.xlabel(
        "Predicted Class",
        fontsize=14,
    )

    plt.ylabel(
        "True Class",
        fontsize=14,
    )

    plt.xticks(
        fontsize=10,
        rotation=90,
    )

    plt.yticks(
        fontsize=10,
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()



def save_predictions(
    reports_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save true and predicted labels for later error analysis."""
    report_dir = get_report_dir(reports_dir, model_name)

    np.save(report_dir / "y_true.npy", y_true)
    np.save(report_dir / "y_pred.npy", y_pred)



def get_file_size(path: Path) -> int:
    """Return file size in bytes."""
    return path.stat().st_size



def format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable size."""
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"



def save_evaluation_artifacts(
    *,
    reports_dir: Path,
    model_name: str,
    metrics: dict,
    report: str,
    confusion: np.ndarray,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Save the common evaluation artifacts for a model."""
    report_dir = get_report_dir(reports_dir, model_name)

    save_json(metrics, report_dir / "metrics.json")
    save_classification_report(
        report,
        report_dir / "classification_report.txt",
    )
    save_confusion_matrix(
        confusion,
        class_names,
        f"{metrics['display_name']} Test Confusion Matrix",
        report_dir / "confusion_matrix.png",
    )
    save_predictions(
        reports_dir,
        model_name,
        y_true,
        y_pred,
    )
    np.save(report_dir / "confusion_matrix.npy", confusion)

    return report_dir



def load_existing_comparison(
    comparison_path: Path,
    model_configs: dict,
    comparison_fields: list[str],
) -> dict[str, dict]:
    """Load valid existing comparison rows."""
    if not comparison_path.exists():
        return {}

    existing = {}

    try:
        with comparison_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                return {}

            for row in reader:
                model_name = (row.get("model") or "").strip()

                if model_name not in model_configs:
                    continue

                if (
                    row.get("display_name") == "display_name"
                    or row.get("model_type") == "model_type"
                ):
                    continue

                existing[model_name] = {
                    field: row.get(field, "")
                    for field in comparison_fields
                }

    except (OSError, csv.Error):
        return {}

    return existing



def normalize_comparison_row(result: dict, comparison_fields: list[str]) -> dict:
    """Convert metrics into a stable CSV row."""
    return {
        field: "" if result.get(field) is None else result.get(field, "")
        for field in comparison_fields
    }



def save_comparison(
    results: list[dict],
    *,
    reports_dir: Path,
    comparison_path: Path,
    model_configs: dict,
    comparison_fields: list[str],
) -> None:
    """Upsert model results into model_comparison.csv."""
    if not results:
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = load_existing_comparison(
        comparison_path,
        model_configs,
        comparison_fields,
    )

    for result in results:
        model_name = result.get("model")
        if model_name not in model_configs:
            continue

        comparison[model_name] = normalize_comparison_row(
            result,
            comparison_fields,
        )

    ordered_rows = [
        comparison[model_name]
        for model_name in model_configs
        if model_name in comparison
    ]

    temp_path = comparison_path.with_suffix(".tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ordered_rows)

    temp_path.replace(comparison_path)

    print(f"\nModel comparison saved to: {comparison_path}")



def print_comparison(results: list[dict]) -> None:
    """Print a compact comparison for the current run."""
    print("\n" + "=" * 120)
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
        model_type = result.get("model_type")

        if model_type == "deep_learning":
            total_params = result.get("total_parameters")
            complexity_text = (
                f"Params={total_params:,}"
                if total_params is not None
                else "N/A"
            )
        elif result["model"] == "svm":
            support_vectors = result.get("support_vectors")
            complexity_text = (
                f"SV={support_vectors:,}"
                if support_vectors is not None
                else "N/A"
            )
        elif result["model"] == "random_forest":
            total_nodes = result.get("total_nodes")
            complexity_text = (
                f"Nodes={total_nodes:,}"
                if total_nodes is not None
                else "N/A"
            )
        else:
            complexity_text = "N/A"

        top5 = result.get("top5_accuracy")
        top5_text = f"{top5:.4f}" if top5 is not None else "N/A"

        print(
            f"{result['display_name']:30}"
            f"{result['model_size']:>12}"
            f"{complexity_text:>18}"
            f"{result['top1_accuracy']:>12.4f}"
            f"{top5_text:>12}"
            f"{result['macro_f1']:>12.4f}"
        )
