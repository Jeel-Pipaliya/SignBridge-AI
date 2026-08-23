"""
DAY 12 -- SignBridge AI  Week 2
Model Evaluation: confusion matrix, per-class accuracy, feature importance.

Run from the project root:
    python backend/models/evaluate_model.py

Requires:
    backend/data/processed/landmarks.csv
    backend/models/isl_model.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATASET_PATH = os.path.join("backend", "data", "processed", "landmarks.csv")
MODEL_PATH   = os.path.join("backend", "models", "isl_model.pkl")
OUTPUT_DIR   = os.path.join("backend", "models")


def main() -> None:
    print("=" * 55)
    print("  SignBridge AI -- Model Evaluation (Day 12)")
    print("=" * 55)

    # Load data
    df = pd.read_csv(DATASET_PATH)
    X  = df.drop("label", axis=1).values.astype(np.float32)
    y  = df["label"].values

    # Load model
    model = joblib.load(MODEL_PATH)
    classes = model.classes_

    # Full-dataset predictions
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\nFull-dataset accuracy  : {acc * 100:.2f}%")

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-fold CV accuracy     : {cv_scores.mean() * 100:.2f}%  "
          f"(+/- {cv_scores.std() * 100:.2f}%)")

    # Per-class report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("SignBridge AI -- Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    print(f"\nConfusion matrix saved : {cm_path}")

    # Top-10 feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop-10 most important features (landmark coordinates):")
    for rank, fi in enumerate(top_idx):
        lm   = fi // 3
        axis = ["x", "y", "z"][fi % 3]
        print(f"  {rank + 1:2d}. Landmark {lm:2d} ({axis})  importance={importances[fi]:.4f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
