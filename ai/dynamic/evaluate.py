"""
SignBridge AI - Dynamic Model Evaluation Suite (Week 5)
Evaluates trained Bi-LSTM dynamic gesture model on test sequences.
Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1, and confusion matrix.

Usage:
  python -m ai.dynamic.evaluate
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dynamic.model import BiLSTMClassifier
from ai.dynamic.dataset import load_dynamic_dataset, split_dataset, DYNAMIC_CLASSES
from ai.utils.logger import setup_logger

logger = setup_logger("DynamicEvaluation")


def evaluate_dynamic_model(
    model_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates dynamic Bi-LSTM model on unseen test set.
    """
    model_path = Path(model_path) if model_path is not None else config.DYNAMIC_MODEL_PATH
    data_dir = Path(data_dir) if data_dir is not None else config.DYNAMIC_SEQUENCES_DIR

    if not model_path.exists():
        msg = f"Dynamic model checkpoint not found at {model_path}. Please run 'python -m ai.dynamic.train' first."
        logger.error(msg)
        print(f"\n[ERROR] {msg}")
        return {"error": msg}

    print("=" * 68)
    print("  SIGNBRIDGE AI — Dynamic Model Test Evaluation (Week 5)")
    print(f"  Model: {model_path}")
    print("=" * 68)

    # 1. Load Model
    model = BiLSTMClassifier.load_checkpoint(model_path)

    # 2. Load Dataset
    dataset = load_dynamic_dataset(data_dir=data_dir, fallback_synthetic=True)
    _, _, test_ds = split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # 3. Inference
    probs = model.predict_proba(test_ds.sequences)
    preds = np.argmax(probs, axis=1)
    y_true = test_ds.labels

    # 4. Metrics
    acc = accuracy_score(y_true, preds)
    macro_p = precision_score(y_true, preds, average="macro", zero_division=0)
    macro_r = recall_score(y_true, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

    report_str = classification_report(y_true, preds, target_names=DYNAMIC_CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, preds)

    print(f"Test Accuracy:    {acc:.2%}")
    print(f"Macro Precision:  {macro_p:.4f}")
    print(f"Macro Recall:     {macro_r:.4f}")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"Weighted F1:      {weighted_f1:.4f}")
    print("\nDetailed Per-Class Breakdown:\n" + report_str)
    print("Confusion Matrix:\n", cm)

    results = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classes": DYNAMIC_CLASSES,
    }
    return results


def main():
    evaluate_dynamic_model()


if __name__ == "__main__":
    main()
