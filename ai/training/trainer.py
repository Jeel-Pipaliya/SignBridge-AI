"""
SignBridge AI - Model Trainer Module (Week 4 Phase 4)
Orchestrates training, validation tracking, checkpointing,
and generating training history loss/accuracy curves.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, f1_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.models.isl_classifier import ISLClassifier
from ai.utils.logger import setup_logger
from ai.utils.config import load_config, get_resolved_path

logger = setup_logger("ModelTrainer")


class Trainer:
    """
    Manages end-to-end model training, validation evaluation,
    artifact versioning, and diagnostic visualization.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.cfg = config_dict or load_config()

        # Resolve directories
        self.checkpoints_dir = get_resolved_path(self.cfg.get("paths", {}).get("checkpoints_dir", "models/checkpoints"))
        self.best_model_dir = get_resolved_path(self.cfg.get("paths", {}).get("best_model_dir", "models/best"))
        self.metadata_dir = get_resolved_path(self.cfg.get("paths", {}).get("metadata_dir", "models/metadata"))
        self.reports_dir = get_resolved_path(self.cfg.get("paths", {}).get("reports_dir", "reports"))

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.classes_: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    def load_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads train, val, and test splits from processed CSVs and encodes labels dynamically.
        """
        train_path = get_resolved_path(self.cfg.get("paths", {}).get("train_data", "data/processed/train.csv"))
        val_path = get_resolved_path(self.cfg.get("paths", {}).get("val_data", "data/processed/val.csv"))
        test_path = get_resolved_path(self.cfg.get("paths", {}).get("test_data", "data/processed/test.csv"))

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # Dynamic class discovery
        all_labels = sorted(list(set(df_train["label"].astype(str).unique())))
        self.classes_ = all_labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes_)}

        feat_cols = [c for c in df_train.columns if c.startswith("lm")]

        X_train = df_train[feat_cols].values.astype(np.float32)
        y_train = np.array([self.class_to_idx[str(lbl)] for lbl in df_train["label"]], dtype=np.int64)

        X_val = df_val[feat_cols].values.astype(np.float32)
        y_val = np.array([self.class_to_idx[str(lbl)] for lbl in df_val["label"]], dtype=np.int64)

        X_test = df_test[feat_cols].values.astype(np.float32)
        y_test = np.array([self.class_to_idx[str(lbl)] for lbl in df_test["label"]], dtype=np.int64)

        logger.info(
            f"Loaded splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, Classes={len(self.classes_)}"
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self) -> Dict[str, Any]:
        """
        Trains model architectures, tracks epoch progression,
        saves best model checkpoints, and generates history curves.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_splits()

        model_cfg = self.cfg.get("model", {})
        train_cfg = self.cfg.get("training", {})

        num_classes = len(self.classes_)
        input_dim = X_train.shape[1]

        # 1. Candidate architectures: MLP, Random Forest, and SVM
        models_to_evaluate = {
            "MLP_NeuralNet": ISLClassifier(
                model_type="mlp",
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_units=tuple(model_cfg.get("hidden_units", [128, 64])),
                activation=model_cfg.get("activation", "relu"),
                alpha=model_cfg.get("alpha", 0.001),
                learning_rate_init=train_cfg.get("learning_rate_init", 0.001),
                learning_rate=train_cfg.get("learning_rate", "adaptive"),
                max_iter=train_cfg.get("max_epochs", 200),
                random_state=train_cfg.get("random_seed", 42),
                early_stopping=train_cfg.get("early_stopping", True),
                n_iter_no_change=train_cfg.get("early_stopping_patience", 10),
            ),
            "RandomForest": ISLClassifier(
                model_type="random_forest",
                input_dim=input_dim,
                num_classes=num_classes,
                random_state=train_cfg.get("random_seed", 42),
            ),
            "SVM_RBF": ISLClassifier(
                model_type="svm",
                input_dim=input_dim,
                num_classes=num_classes,
                random_state=train_cfg.get("random_seed", 42),
            ),
        }

        print("\n" + "=" * 75)
        print(f"{'Model Candidate':<18} | {'Train Acc':<11} | {'Val Acc':<11} | {'Val F1':<10} | {'Test Acc':<10}")
        print("=" * 75)

        benchmark_results = {}
        champion_name = ""
        champion_val_acc = -1.0
        champion_model = None

        for name, clf in models_to_evaluate.items():
            clf.fit(X_train, y_train)

            train_pred = clf.predict(X_train)
            train_acc = float(accuracy_score(y_train, train_pred))

            val_pred = clf.predict(X_val)
            val_acc = float(accuracy_score(y_val, val_pred))
            val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

            test_pred = clf.predict(X_test)
            test_acc = float(accuracy_score(y_test, test_pred))
            test_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))

            benchmark_results[name] = {
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "model": clf,
            }

            print(f"{name:<18} | {train_acc:>9.2%} | {val_acc:>9.2%} | {val_f1:>8.4f} | {test_acc:>8.2%}")

            if val_acc > champion_val_acc:
                champion_val_acc = val_acc
                champion_name = name
                champion_model = clf

        print("=" * 75)
        print(f"[Selected Champion] Best Validation Performance: '{champion_name}' ({champion_val_acc:.2%})")

        # 2. Plot Training History for the MLP Neural Network
        mlp_model = models_to_evaluate["MLP_NeuralNet"]
        self._plot_training_history(mlp_model, X_train, y_train, X_val, y_val)

        # 3. Save Artifacts & Metadata
        self._save_artifacts(champion_model, champion_name, benchmark_results[champion_name])

        return {
            "champion_name": champion_name,
            "champion_model": champion_model,
            "benchmark_results": benchmark_results,
            "classes": self.classes_,
        }

    def _plot_training_history(
        self,
        mlp_classifier: ISLClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Plots training history with:
        Graph 1: Training vs Validation Accuracy
        Graph 2: Training Loss
        """
        loss_curve = getattr(mlp_classifier.model, "loss_curve_", [])
        val_scores = getattr(mlp_classifier.model, "validation_scores_", [])

        epochs = range(1, len(loss_curve) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Graph 1: Accuracy (if val_scores available or final scores)
        if len(val_scores) > 0:
            val_epochs = range(1, len(val_scores) + 1)
            ax1.plot(val_epochs, val_scores, label="Validation Accuracy", color="#2ca02c", linewidth=2)
        else:
            # Show final accuracy benchmark
            final_train_acc = accuracy_score(y_train, mlp_classifier.predict(X_train))
            final_val_acc = accuracy_score(y_val, mlp_classifier.predict(X_val))
            ax1.axhline(final_train_acc, color="#1f77b4", linestyle="--", label=f"Train Acc: {final_train_acc:.2%}")
            ax1.axhline(final_val_acc, color="#2ca02c", linestyle="-", label=f"Val Acc: {final_val_acc:.2%}")

        ax1.set_title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch / Iteration", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right")

        # Graph 2: Training Loss
        if len(loss_curve) > 0:
            ax2.plot(epochs, loss_curve, label="Training Loss (Cross Entropy)", color="#d62728", linewidth=2)
        ax2.set_title("Training Loss Curve", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch / Iteration", fontsize=10)
        ax2.set_ylabel("Cross Entropy Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plot_path = self.reports_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved training history plot to: {plot_path}")

    def _save_artifacts(
        self,
        champion_model: ISLClassifier,
        champion_name: str,
        champion_metrics: dict,
    ) -> None:
        """
        Serializes model checkpoints, best model, metadata, and class mapping.
        """
        # Save checkpoints
        checkpoint_file = self.checkpoints_dir / f"isl_classifier_{champion_name.lower()}_best.pkl"
        joblib.dump(champion_model, checkpoint_file)

        # Save to best/ and standard root models/
        best_file = self.best_model_dir / "isl_classifier.pkl"
        joblib.dump(champion_model, best_file)
        joblib.dump(champion_model, config.MODEL_PATH)

        # Save LabelEncoder compatibility artifact
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(self.classes_)
        joblib.dump(le, config.LABEL_ENCODER_PATH)

        # Save class names JSON
        class_names_path = self.metadata_dir / "class_names.json"
        with open(class_names_path, "w", encoding="utf-8") as f:
            json.dump({str(i): cls for i, cls in enumerate(self.classes_)}, f, indent=2)

        # Save model metadata
        metadata = {
            "model_name": f"SignBridge_ISL_{champion_name}",
            "version": "1.0",
            "model_type": champion_model.model_type,
            "input_dimension": champion_model.input_dim,
            "num_classes": len(self.classes_),
            "classes": self.classes_,
            "confidence_threshold": float(self.cfg.get("inference", {}).get("confidence_threshold", 0.70)),
            "train_accuracy": float(champion_metrics["train_acc"]),
            "validation_accuracy": float(champion_metrics["val_acc"]),
            "test_accuracy": float(champion_metrics["test_acc"]),
            "validation_macro_f1": float(champion_metrics["val_f1"]),
            "test_macro_f1": float(champion_metrics["test_f1"]),
        }
        metadata_path = self.metadata_dir / "model_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model artifacts and metadata saved to: {self.metadata_dir}")
