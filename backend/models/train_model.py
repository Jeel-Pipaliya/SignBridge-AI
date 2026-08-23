"""
DAY 12 -- SignBridge AI  Week 2
Model Training: trains a Random Forest classifier on the ISL landmark CSV.

Run from the project root:
    python backend/models/train_model.py

Output:
    backend/models/isl_model.pkl      -- trained Random Forest
    backend/models/label_classes.npy  -- ordered class labels array
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ---- Configuration -----------------------------------------------------------
DATASET_PATH: str = os.path.join(
    "backend", "data", "processed", "landmarks.csv"
)
MODEL_DIR:    str = os.path.join("backend", "models")
MODEL_PATH:   str = os.path.join(MODEL_DIR, "isl_model.pkl")
LABELS_PATH:  str = os.path.join(MODEL_DIR, "label_classes.npy")

N_ESTIMATORS: int   = 200
TEST_SIZE:    float = 0.20
RANDOM_STATE: int   = 42


# ---- Helpers -----------------------------------------------------------------

def _load_dataset(path: str):
    """Load CSV, split into X (features) and y (labels)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run backend/data/collect_data.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset CSV is empty. Collect data first.")

    print(f"Dataset loaded: {len(df)} rows")
    print(f"  Signs found: {sorted(df['label'].unique())}")
    print(f"  Samples per sign:")
    for sign, count in df["label"].value_counts().items():
        print(f"    {sign:15s} {count}")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values
    return X, y


# ---- Training ----------------------------------------------------------------

def train(X, y):
    """Split data, train Random Forest, return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} ...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(y_test, y_pred))

    return model, accuracy


# ---- Entry point -------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  SignBridge AI -- Model Training (Day 12)")
    print("=" * 55)

    X, y = _load_dataset(DATASET_PATH)
    model, accuracy = train(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    np.save(LABELS_PATH, model.classes_)

    print(f"Model saved  : {MODEL_PATH}")
    print(f"Labels saved : {LABELS_PATH}")
    print(f"Classes      : {list(model.classes_)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
