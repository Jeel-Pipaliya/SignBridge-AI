"""
SignBridge AI - Rigorous Model Evaluation Module (Week 4 Phase 5 & 6)
Evaluates trained ISL model on untouched test set.
Generates:
  1. reports/confusion_matrix.png
  2. reports/classification_report.csv
  3. Misclassification error analysis
  4. Appends run entry to experiments/experiment_log.csv
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.utils.metrics import compute_metrics, find_top_misclassifications, classification_report_to_dataframe
from ai.utils.logger import setup_logger
from ai.utils.config import load_config, get_resolved_path

logger = setup_logger("ModelEvaluation")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> Path:
    """
    Renders high-resolution confusion matrix heatmap for 32 ISL classes.
    """
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("SignBridge AI — 32-Class Confusion Matrix (Unseen Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=8, fontweight="bold")
    plt.yticks(ticks, class_names, fontsize=8, fontweight="bold")

    # Annotate counts inside cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > thresh else "black"
                plt.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=7, fontweight="bold")

    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=8)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_path}")
    return output_path


def log_experiment(
    exp_id: str,
    model_name: str,
    dataset_version: str,
    features: str,
    epochs: int,
    lr: float,
    val_acc: float,
    test_acc: float,
    f1: float,
    notes: str,
    log_file: Path = PROJECT_ROOT / "experiments" / "experiment_log.csv",
) -> None:
    """
    Records training and evaluation metrics into experiment tracking CSV.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    header = not log_file.exists()

    df_entry = pd.DataFrame([{
        "Experiment ID": exp_id,
        "Model": model_name,
        "Dataset version": dataset_version,
        "Features": features,
        "Epochs": epochs,
        "Learning rate": lr,
        "Validation accuracy": f"{val_acc:.4f}",
        "Test accuracy": f"{test_acc:.4f}",
        "F1": f"{f1:.4f}",
        "Notes": notes,
    }])

    df_entry.to_csv(log_file, mode="a", index=False, header=header)
    logger.info(f"Appended experiment record to: {log_file}")


def evaluate_model() -> Dict[str, Any]:
    """
    Main evaluation pipeline on unseen test data.
    """
    print("=" * 75)
    print("  SIGNBRIDGE AI — Model Evaluation & Diagnostics (Week 4)")
    print("=" * 75)

    cfg = load_config()
    test_csv = get_resolved_path(cfg.get("paths", {}).get("test_data", "data/processed/test.csv"))
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model and class mapping
    model_path = config.MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")

    model = joblib.load(model_path)

    metadata_path = PROJECT_ROOT / "models" / "metadata" / "class_names.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            class_dict = json.load(f)
        class_names = [class_dict[str(i)] for i in range(len(class_dict))]
    else:
        label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        class_names = list(label_encoder.classes_)

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    # 2. Load test set
    df_test = pd.read_csv(test_csv)
    feat_cols = [c for c in df_test.columns if c.startswith("lm")]
    X_test = df_test[feat_cols].values.astype(np.float32)
    y_test = np.array([class_to_idx[str(lbl)] for lbl in df_test["label"]], dtype=np.int64)

    # 3. Model predictions
    y_pred = model.predict(X_test)
    if y_pred.dtype == object or isinstance(y_pred[0], str):
        y_pred = np.array([class_to_idx[str(lbl)] for lbl in y_pred], dtype=np.int64)

    # 4. Compute metrics
    metrics = compute_metrics(y_test, y_pred, class_names)
    acc = metrics["accuracy"]
    prec_macro = metrics["precision_macro"]
    rec_macro = metrics["recall_macro"]
    f1_macro = metrics["f1_macro"]

    print("\n[Aggregate Evaluation Results on Unseen Test Set]")
    print(f"  • Overall Test Accuracy:  {acc:.2%}")
    print(f"  • Macro Precision:        {prec_macro:.4f}")
    print(f"  • Macro Recall:           {rec_macro:.4f}")
    print(f"  • Macro F1-Score:         {f1_macro:.4f}")
    print(f"  • Weighted F1-Score:      {metrics['f1_weighted']:.4f}")

    # 5. Save Classification Report CSV
    report_df = classification_report_to_dataframe(metrics["classification_report_dict"], class_names)
    report_csv = reports_dir / "classification_report.csv"
    report_df.to_csv(report_csv, index=False)
    logger.info(f"Saved classification report to: {report_csv}")

    # 6. Plot Confusion Matrix
    cm_plot = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_plot)

    # 7. Misclassification Analysis
    top_errors = find_top_misclassifications(metrics["confusion_matrix"], class_names, top_k=7)
    # Safe encoding for Windows consoles
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n[Misclassification Error Analysis - Most Confused Class Pairs]")
    if top_errors:
        for true_cls, pred_cls, count in top_errors:
            print(f"  * True: '{true_cls}' -> Predicted: '{pred_cls}' ({count} occurrences)")
    else:
        print("  * No misclassifications found on test set.")

    # 8. Experiment Logging
    meta_path = PROJECT_ROOT / "models" / "metadata" / "model_metadata.json"
    val_acc = 0.0
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            val_acc = meta.get("validation_accuracy", 0.0)

    log_experiment(
        exp_id="EXP-001",
        model_name=type(getattr(model, "model", model)).__name__,
        dataset_version="ISL-32-Classes-v1",
        features="Landmarks-63D-Normalized",
        epochs=100,
        lr=0.001,
        val_acc=val_acc,
        test_acc=acc,
        f1=f1_macro,
        notes="Week 4 Baseline ISL Classifier evaluation on 32 classes",
    )

    print("\n" + "=" * 75)
    print("  Evaluation Complete")
    print("=" * 75 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_model()
