"""
SignBridge AI - Dynamic Gesture Model Training Pipeline (Week 5)
Trains BiLSTMClassifier and DynamicBaselineClassifier on gesture sequences (HELLO, J, Z).
Produces:
  - Checkpoint: models/dynamic/bilstm_dynamic.pt
  - Metadata: models/metadata/dynamic_model_metadata.json, dynamic_class_names.json
  - Plots: reports/dynamic_training_history.png, reports/dynamic_confusion_matrix.png
  - Report: reports/dynamic_classification_report.csv
  - Experiment log: experiments/experiment_log.csv

Usage:
  python -m ai.dynamic.train
  python -m ai.dynamic.train --synthetic --epochs 40
"""

import sys
import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.dynamic.dataset import load_dynamic_dataset, split_dataset, DYNAMIC_CLASSES
from ai.dynamic.model import BiLSTMClassifier, DynamicBaselineClassifier
from ai.utils.logger import setup_logger
from ai.utils.config import load_config

logger = setup_logger("DynamicTraining")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#0066cc", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff6600", lw=2, linestyle="--")
    ax1.set_title("Bi-LSTM Dynamic Loss Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#009933", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#cc0066", lw=2, linestyle="--")
    ax2.set_title("Bi-LSTM Dynamic Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved dynamic training history plot to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    output_path: Path,
) -> None:
    """Render high-resolution confusion matrix heatmap with pure matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Dynamic Gesture Bi-LSTM Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved dynamic confusion matrix plot to {output_path}")


def log_experiment_entry(
    experiment_id: str,
    model_name: str,
    val_acc: float,
    test_acc: float,
    macro_f1: float,
    epochs: int,
    lr: float,
    notes: str,
) -> None:
    """Append row to experiments/experiment_log.csv."""
    exp_file = Path(config.PROJECT_ROOT / "experiments" / "experiment_log.csv")
    exp_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = exp_file.exists()
    with open(exp_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Experiment ID", "Model", "Dataset version", "Features",
                "Epochs", "Learning rate", "Validation accuracy", "Test accuracy", "F1", "Notes"
            ])
        writer.writerow([
            experiment_id,
            model_name,
            "Dynamic-ISL-3-Classes-v1",
            "Landmarks-30x63-BiLSTM",
            epochs,
            lr,
            f"{val_acc:.4f}",
            f"{test_acc:.4f}",
            f"{macro_f1:.4f}",
            notes,
        ])
    logger.info(f"Logged experiment {experiment_id} ({model_name}) to {exp_file}")


def run_dynamic_training(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    force_synthetic: bool = False,
) -> Tuple[BiLSTMClassifier, Dict[str, float]]:
    """
    Main training routine for Week 5 dynamic sequence recognizer.
    """
    cfg = load_config()
    dyn_cfg = cfg.get("dynamic", {})

    epochs = epochs or int(dyn_cfg.get("epochs", 40))
    batch_size = batch_size or int(dyn_cfg.get("batch_size", 32))
    learning_rate = lr or float(dyn_cfg.get("learning_rate", 0.001))
    seq_len = int(dyn_cfg.get("sequence_length", 30))
    feat_dim = int(dyn_cfg.get("feature_dimension", 63))
    hidden_size = int(dyn_cfg.get("hidden_size", 128))
    dropout = float(dyn_cfg.get("dropout", 0.3))

    reports_dir = Path(config.PROJECT_ROOT / "reports")
    models_dir = Path(config.DYNAMIC_MODELS_DIR)
    metadata_dir = Path(config.DYNAMIC_METADATA_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  SIGNBRIDGE AI — Dynamic Gesture Model Training (Week 5)")
    print(f"  Vocabulary: {', '.join(DYNAMIC_CLASSES)}")
    print(f"  Sequence Shape: ({seq_len}, {feat_dim}) | Epochs: {epochs} | LR: {learning_rate}")
    print("=" * 68)

    # 1. Load Dataset
    dataset = load_dynamic_dataset(
        data_dir=config.DYNAMIC_SEQUENCES_DIR,
        target_length=seq_len,
        fallback_synthetic=True,
    )

    train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # 2. PyTorch DataLoaders
    train_x = torch.from_numpy(train_ds.sequences).float()
    train_y = torch.from_numpy(train_ds.labels).long()
    val_x = torch.from_numpy(val_ds.sequences).float()
    val_y = torch.from_numpy(val_ds.labels).long()
    test_x = torch.from_numpy(test_ds.sequences).float()
    test_y = torch.from_numpy(test_ds.labels).long()

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    # 3. Instantiate Bi-LSTM Model
    model = BiLSTMClassifier(
        sequence_length=seq_len,
        feature_dim=feat_dim,
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=len(DYNAMIC_CLASSES),
        dropout=dropout,
        classes=DYNAMIC_CLASSES,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # 4. Training Loop with Checkpointing
    best_val_acc = 0.0
    best_model_path = models_dir / "bilstm_dynamic.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        epoch_train_loss = total_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += bx.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model.save_checkpoint(best_model_path, optimizer=optimizer, epoch=epoch, val_acc=best_val_acc)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.1%} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1%}")

    train_time = time.time() - t0
    print(f"\nBi-LSTM Training completed in {train_time:.2f}s. Best Val Accuracy: {best_val_acc:.1%}")

    # 5. Load Best Model & Evaluate on Test Set
    best_model = BiLSTMClassifier.load_checkpoint(best_model_path)
    test_probs = best_model.predict_proba(test_ds.sequences)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = accuracy_score(test_ds.labels, test_preds)
    macro_f1 = f1_score(test_ds.labels, test_preds, average="macro", zero_division=0)

    print(f"\nUnseen Test Set Evaluation (Bi-LSTM):")
    print(f"  • Test Accuracy: {test_acc:.2%}")
    print(f"  • Macro F1-Score: {macro_f1:.4f}")

    # 6. Comparative Baseline Model (Random Forest over temporal statistics)
    print("\nTraining Comparative Temporal Baseline (Random Forest)...")
    baseline = DynamicBaselineClassifier(model_type="random_forest", classes=DYNAMIC_CLASSES)
    baseline.fit(train_ds.sequences, train_ds.labels)
    base_val_preds = baseline.predict(val_ds.sequences)
    base_val_acc = accuracy_score(val_ds.labels, base_val_preds)
    base_test_preds = baseline.predict(test_ds.sequences)
    base_test_acc = accuracy_score(test_ds.labels, base_test_preds)
    base_macro_f1 = f1_score(test_ds.labels, base_test_preds, average="macro", zero_division=0)
    print(f"  • Baseline Val Acc: {base_val_acc:.2%} | Test Acc: {base_test_acc:.2%} | Macro F1: {base_macro_f1:.4f}")

    # 7. Generate Reports & Plots
    plot_training_history(history, reports_dir / "dynamic_training_history.png")
    plot_confusion_matrix(test_ds.labels, test_preds, DYNAMIC_CLASSES, reports_dir / "dynamic_confusion_matrix.png")

    report_str = classification_report(test_ds.labels, test_preds, target_names=DYNAMIC_CLASSES, zero_division=0)
    report_dict = classification_report(test_ds.labels, test_preds, target_names=DYNAMIC_CLASSES, output_dict=True, zero_division=0)

    # Save classification report CSV
    report_csv = reports_dir / "dynamic_classification_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for cls_name in DYNAMIC_CLASSES:
            row = report_dict.get(cls_name, {})
            writer.writerow([
                cls_name,
                f"{row.get('precision', 0.0):.4f}",
                f"{row.get('recall', 0.0):.4f}",
                f"{row.get('f1-score', 0.0):.4f}",
                int(row.get('support', 0)),
            ])
        writer.writerow([])
        macro = report_dict.get("macro avg", {})
        writer.writerow(["Macro Avg", f"{macro.get('precision', 0.0):.4f}", f"{macro.get('recall', 0.0):.4f}", f"{macro.get('f1-score', 0.0):.4f}", int(macro.get('support', 0))])
        weighted = report_dict.get("weighted avg", {})
        writer.writerow(["Weighted Avg", f"{weighted.get('precision', 0.0):.4f}", f"{weighted.get('recall', 0.0):.4f}", f"{weighted.get('f1-score', 0.0):.4f}", int(weighted.get('support', 0))])

    # Save Metadata JSONs
    with open(metadata_dir / "dynamic_class_names.json", "w", encoding="utf-8") as f:
        json.dump({"classes": DYNAMIC_CLASSES, "num_classes": len(DYNAMIC_CLASSES)}, f, indent=2)

    metadata = {
        "model_architecture": "BiLSTM",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_length": seq_len,
        "feature_dim": feat_dim,
        "classes": DYNAMIC_CLASSES,
        "epochs_trained": epochs,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "baseline_comparison": {
            "model": "RandomForest_TemporalSummary",
            "val_accuracy": float(base_val_acc),
            "test_accuracy": float(base_test_acc),
            "macro_f1": float(base_macro_f1),
        },
    }
    with open(metadata_dir / "dynamic_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Log Experiments
    log_experiment_entry(
        experiment_id="EXP-002",
        model_name="BiLSTMClassifier",
        val_acc=best_val_acc,
        test_acc=test_acc,
        macro_f1=macro_f1,
        epochs=epochs,
        lr=learning_rate,
        notes="Week 5 Dynamic Gesture Bi-LSTM (HELLO, J, Z)",
    )
    log_experiment_entry(
        experiment_id="EXP-003",
        model_name="RandomForest_TemporalSummary",
        val_acc=base_val_acc,
        test_acc=base_test_acc,
        macro_f1=base_macro_f1,
        epochs=100,
        lr=0.0,
        notes="Week 5 Baseline Temporal Classifier comparison",
    )

    print("\nClassification Report:\n" + report_str)
    return best_model, metadata


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Dynamic Model Trainer")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation")
    args = parser.parse_args()

    run_dynamic_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
