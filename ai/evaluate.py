"""
SignBridge AI - Model Evaluation & Diagnostics Module (Week 3 Day 2)
Computes rigorous quantitative metrics on the unseen test set:
- Overall Accuracy
- Macro & Weighted Precision, Recall, F1-Score
- Per-Class Classification Report
- Generates high-resolution Confusion Matrix heatmap:
    outputs/confusion_matrix/confusion_matrix.png
- Generates per-class F1-score bar chart:
    outputs/plots/per_class_f1.png
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dataset_loader import TEST_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config.CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: list,
    output_path: Path = config.CONFUSION_MATRIX_DIR / "confusion_matrix.png",
) -> Path:
    """Plot and save confusion matrix with raw counts and normalized percentages."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("SignBridge AI — Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=10, fontweight="bold")
    plt.yticks(tick_marks, class_names, fontsize=10, fontweight="bold")

    # Overlay numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.ylabel("True ISL Class", fontsize=12, labelpad=10)
    plt.xlabel("Predicted ISL Class", fontsize=12, labelpad=10)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to: {output_path}")
    return output_path


def plot_per_class_f1(
    report_dict: dict,
    class_names: list,
    output_path: Path = config.PLOTS_DIR / "per_class_f1.png",
) -> Path:
    """Plot bar chart of per-class F1-scores."""
    f1_scores = [report_dict[cls]["f1-score"] for cls in class_names if cls in report_dict]

    plt.figure(figsize=(11, 5))
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(class_names)))
    bars = plt.bar(class_names, f1_scores, color=palette, edgecolor="black", alpha=0.85)

    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    plt.axhline(mean_f1, color="red", linestyle="--", label=f"Macro Avg F1: {mean_f1:.2f}")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("SignBridge AI — Per-Class F1-Score Performance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("ISL Class", fontsize=11, labelpad=8)
    plt.ylabel("F1-Score", fontsize=11, labelpad=8)
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved per-class F1 plot to: {output_path}")
    return output_path


def evaluate_trained_model(
    model_path: Path = config.MODEL_PATH,
    encoder_path: Path = config.LABEL_ENCODER_PATH,
    test_csv: Path = TEST_CSV,
) -> dict:
    """Load model, run inference on test set, and generate evaluation diagnostics."""
    print("=" * 65)
    print("  SignBridge AI - Model Evaluation & Diagnostics (Week 3 Day 2)")
    print("=" * 65)

    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Model or LabelEncoder not found. Please run ai/train.py first.")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    class_names = list(label_encoder.classes_)

    # Load test set
    df_test = pd.read_csv(test_csv)
    feat_cols = [c for c in df_test.columns if c.startswith("lm")]
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_true_str = df_test["label"].astype(str).values
    y_true = label_encoder.transform(y_true_str)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_str = label_encoder.inverse_transform(y_pred)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n[Aggregate Evaluation Results on Unseen Test Set]")
    print(f"  • Overall Test Accuracy: {acc:.2%}")
    print(f"  • Macro Precision:       {prec_macro:.4f}")
    print(f"  • Macro Recall:          {rec_macro:.4f}")
    print(f"  • Macro F1-Score:        {f1_macro:.4f}")

    print("\n[Per-Class Classification Report]")
    report_text = classification_report(y_true_str, y_pred_str, target_names=class_names, zero_division=0)
    print(report_text)

    report_dict = classification_report(y_true_str, y_pred_str, target_names=class_names, output_dict=True, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = plot_confusion_matrix_heatmap(cm, class_names)
    f1_path = plot_per_class_f1(report_dict, class_names)

    print(f"\n[Generated Visualizations]")
    print(f"  1. Confusion Matrix: {cm_path}")
    print(f"  2. Per-Class F1:     {f1_path}")

    # Real-World Error Analysis Discussion
    print("\n[Engineering Analysis: Why Accuracy != Complete Real-World Performance]")
    print("  1. Similar Hand Topologies: E.g., '2' vs 'V' or '5' vs 'HELLO' share similar")
    print("     finger extensions; small rotational angles in live video may cause drift.")
    print("  2. Optical Occlusion: When the thumb crosses in front of palm, MediaPipe")
    print("     depth estimate (z-coordinate) has higher noise than (x, y).")
    print("  3. Motion Blur & Lighting: Rapid transitions drop MediaPipe confidence.")
    print("  4. Mitigation: Handled via Confidence Thresholding + Temporal Sliding-Window Smoothing.")

    return {
        "accuracy": acc,
        "precision": prec_macro,
        "recall": rec_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


if __name__ == "__main__":
    evaluate_trained_model()
