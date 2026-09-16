import numpy as np

from sklearn.metrics import f1_score



def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
    class_labels: np.ndarray | None = None,
) -> float:
    """Calculate top-k classification accuracy."""
    k = min(k, y_prob.shape[1])
    if class_labels is None:
        class_labels = np.arange(y_prob.shape[1])
    class_labels = np.asarray(class_labels)

    top_k_predictions = np.argsort(y_prob, axis=1)[:, -k:]
    top_k_predictions = class_labels[top_k_predictions]

    return float(
        np.mean(
            np.any(top_k_predictions == y_true[:, None], axis=1)
        )
    )



def calculate_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))
