"""
SignBridge AI - Model Training Module (Week 3 Day 1)
Trains and compares baseline classifiers:
1. Random Forest Classifier
2. Support Vector Machine (RBF Kernel)
3. Multi-Layer Perceptron (MLP)

Evaluates validation performance, selects the best model,
evaluates on the unseen test set, and serializes the model to models/isl_classifier.pkl.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dataset_loader import load_dataset_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Tuple[object, str, Dict[str, dict]]:
    """
    Trains Random Forest, SVM, and MLP classifiers.
    Compares validation accuracy and selects the top-performing model.
    """
    candidate_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            C=10.0,
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        ),
    }

    results: Dict[str, dict] = {}
    best_model_name = ""
    best_val_acc = -1.0
    best_model = None

    print("-" * 65)
    print(f"{'Model Name':<20} | {'Val Accuracy':<14} | {'Val Macro F1':<14} | {'Test Accuracy':<14}")
    print("-" * 65)

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        # Validation set evaluation
        val_preds = model.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_f1 = float(f1_score(y_val, val_preds, average="macro"))

        # Test set evaluation
        test_preds = model.predict(X_test)
        test_acc = float(accuracy_score(y_test, test_preds))
        test_f1 = float(f1_score(y_test, test_preds, average="macro"))

        results[name] = {
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "model_obj": model,
        }

        print(f"{name:<20} | {val_acc:>12.2%} | {val_f1:>12.4f} | {test_acc:>12.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = name
            best_model = model

    print("-" * 65)
    print(f"[Selected Model] Best validation performance: '{best_model_name}' ({best_val_acc:.2%})")

    return best_model, best_model_name, results


def save_trained_model(
    model: object,
    model_name: str,
    output_path: Path = config.MODEL_PATH,
) -> Path:
    """Save the selected trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Saved {model_name} model to {output_path}")
    return output_path


def run_training() -> Tuple[object, str, float]:
    """Execute complete training pipeline and export best classifier."""
    print("=" * 65)
    print("  SignBridge AI - Model Training & Architecture Selection (Week 3 Day 1)")
    print("=" * 65)

    # 1. Load stratified dataset partitions
    X_train, y_train, X_val, y_val, X_test, y_test, le = load_dataset_splits()
    print(f"Loaded {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples.")
    print(f"Number of classes: {len(le.classes_)} -> {list(le.classes_)}")

    # 2. Train and benchmark candidate algorithms
    best_model, best_name, metrics = train_and_compare_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 3. Export champion model
    saved_path = save_trained_model(best_model, best_name)
    test_acc = metrics[best_name]["test_accuracy"]

    print(f"\n[Training Complete]")
    print(f"  • Champion Model:      {best_name}")
    print(f"  • Final Test Accuracy: {test_acc:.2%}")
    print(f"  • Serialized Artifact: {saved_path}")

    return best_model, best_name, test_acc


if __name__ == "__main__":
    run_training()
