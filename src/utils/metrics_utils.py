import numpy as np

from sklearn.metrics import f1_score



def calculate_top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> float:
    """Calculate top-k classification accuracy."""
    k = min(k, y_prob.shape[1])

    top_k_predictions = np.argsort(y_prob, axis=1)[:, -k:]

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
