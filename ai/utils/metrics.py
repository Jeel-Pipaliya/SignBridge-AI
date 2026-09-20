"""
SignBridge AI - Evaluation Metrics Utilities
Calculates quantitative metrics, per-class stats, and misclassification pairs.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, any]:
    """
    Computes overall accuracy, macro and weighted precision, recall, and F1.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=range(len(class_names)),
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report_dict": report_dict,
    }


def find_top_misclassifications(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, str, int]]:
    """
    Identifies the most frequent off-diagonal errors in the confusion matrix.
    Returns: List of (TrueClass, PredClass, Count)
    """
    errors = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], int(cm[i, j])))

    # Sort descending by error count
    errors.sort(key=lambda x: x[2], reverse=True)
    return errors[:top_k]


def classification_report_to_dataframe(
    report_dict: dict,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Converts sklearn classification report dictionary to a clean tabular DataFrame.
    """
    rows = []
    for cls in class_names:
        if cls in report_dict:
            stats = report_dict[cls]
            rows.append({
                "Class": cls,
                "Precision": round(stats["precision"], 4),
                "Recall": round(stats["recall"], 4),
                "F1": round(stats["f1-score"], 4),
                "Support": int(stats["support"]),
            })
    return pd.DataFrame(rows)
