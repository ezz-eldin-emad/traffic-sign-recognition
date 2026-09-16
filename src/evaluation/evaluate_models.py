import argparse

import tensorflow as tf

from src.config import COMPARISON_PATH, MODEL_REGISTRY, REPORTS_DIR
from src.evaluation.evaluation import evaluate_model
from src.utils.evaluation_utils import print_comparison, save_comparison


# Kept as an alias for existing imports and command-line behavior.
MODEL_CONFIGS = MODEL_REGISTRY

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
    "robustness_mean_top1",
    "robustness_mean_top5",
    "robustness_mean_macro_f1",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "throughput_images_per_second",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved traffic-sign models.")
    parser.add_argument(
        "--model",
        choices=[*MODEL_CONFIGS, "all"],
        required=True,
        help="Model to evaluate. No training is performed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    model_names = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    results = []

    for model_name in model_names:
        try:
            results.append(evaluate_model(model_name, MODEL_CONFIGS, REPORTS_DIR))
        except (FileNotFoundError, NotImplementedError) as exc:
            print(f"\n[SKIPPED] {model_name}: {exc}")

    if results:
        print_comparison(results)
        save_comparison(
            results,
            reports_dir=REPORTS_DIR,
            comparison_path=COMPARISON_PATH,
            model_configs=MODEL_CONFIGS,
            comparison_fields=COMPARISON_FIELDS,
        )


if __name__ == "__main__":
    main()
