import argparse
from pathlib import Path

import tensorflow as tf

from src.config import (
    IMG_SIZE_CUSTOM_CNN,
    IMG_SIZE_MOBILENET,
    MODEL_DIR,
)
from src.evaluation.evaluation import evaluate_model
from src.utils.evaluation_utils import print_comparison, save_comparison


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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate traffic-sign recognition models."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_CONFIGS.keys(), "all"],
        required=True,
        help="Model to evaluate.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        model_names = list(MODEL_CONFIGS.keys())
    else:
        model_names = [args.model]

    results = []

    for model_name in model_names:
        try:
            result = evaluate_model(
                model_name,
                MODEL_CONFIGS,
                REPORTS_DIR,
            )
            results.append(result)
        except FileNotFoundError as exc:
            print(f"\n[SKIPPED] {model_name}")
            print(exc)
        except NotImplementedError as exc:
            print(f"\n[SKIPPED] {model_name}")
            print(exc)

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
