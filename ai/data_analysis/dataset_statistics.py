"""
SignBridge AI - Dataset Statistics Module (Week 4 Phase 1)
Analyzes dataset composition across raw images and processed feature splits.
Generates tabular summary and class distribution visualizations saved under reports/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.utils.logger import setup_logger

logger = setup_logger("DatasetStatistics")


def compute_dataset_statistics() -> Dict[str, Any]:
    """
    Computes rigorous empirical metrics across raw dataset and processed splits.
    """
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw Dataset Inspection
    raw_dir = config.RAW_DATA_DIR
    raw_class_counts: Dict[str, int] = {}
    if raw_dir.exists():
        for d in sorted(os.listdir(raw_dir)):
            class_path = raw_dir / d
            if class_path.is_dir():
                valid_images = [
                    f for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                raw_class_counts[d] = len(valid_images)

    # 2. Processed Splits Inspection
    train_csv = config.PROCESSED_DATA_DIR / "train.csv"
    val_csv = config.PROCESSED_DATA_DIR / "val.csv"
    test_csv = config.PROCESSED_DATA_DIR / "test.csv"
    all_csv = config.LANDMARKS_CSV_PATH

    df_train = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    df_val = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()
    df_test = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()
    df_all = pd.read_csv(all_csv) if all_csv.exists() else pd.DataFrame()

    total_samples = len(df_all) if not df_all.empty else sum(raw_class_counts.values())
    total_classes = df_all["label"].nunique() if not df_all.empty and "label" in df_all else len(raw_class_counts)
    class_names = sorted(list(df_all["label"].unique())) if not df_all.empty and "label" in df_all else sorted(list(raw_class_counts.keys()))

    train_samples = len(df_train)
    val_samples = len(df_val)
    test_samples = len(df_test)

    # Samples per class in processed benchmark
    per_class_counts = df_all["label"].value_counts() if not df_all.empty and "label" in df_all else pd.Series(raw_class_counts)

    min_samples = int(per_class_counts.min()) if not per_class_counts.empty else 0
    max_samples = int(per_class_counts.max()) if not per_class_counts.empty else 0
    avg_samples = float(per_class_counts.mean()) if not per_class_counts.empty else 0.0

    print("=" * 60)
    print("SignBridge AI — Dataset Statistics Report")
    print("=" * 60)
    print(f"Total samples:          {total_samples}")
    print(f"Total classes:          {total_classes}")
    print(f"Training samples:       {train_samples}")
    print(f"Validation samples:     {val_samples}")
    print(f"Testing samples:        {test_samples}")
    print(f"Minimum samples/class:  {min_samples}")
    print(f"Maximum samples/class:  {max_samples}")
    print(f"Average samples/class:  {avg_samples:.2f}")
    print("=" * 60)

    # Save to reports/dataset_statistics.csv
    stats_df = pd.DataFrame([
        {"Metric": "Total samples", "Value": str(total_samples)},
        {"Metric": "Total classes", "Value": str(total_classes)},
        {"Metric": "Training samples", "Value": str(train_samples)},
        {"Metric": "Validation samples", "Value": str(val_samples)},
        {"Metric": "Testing samples", "Value": str(test_samples)},
        {"Metric": "Minimum samples/class", "Value": str(min_samples)},
        {"Metric": "Maximum samples/class", "Value": str(max_samples)},
        {"Metric": "Average samples/class", "Value": f"{avg_samples:.2f}"},
    ])
    csv_out = reports_dir / "dataset_statistics.csv"
    stats_df.to_csv(csv_out, index=False)
    logger.info(f"Saved dataset statistics CSV to: {csv_out}")

    # Generate Class Distribution Plot
    if not per_class_counts.empty:
        plt.figure(figsize=(14, 6))
        sorted_series = per_class_counts.sort_index()
        bars = plt.bar(sorted_series.index.astype(str), sorted_series.values, color="#2b5c8f", edgecolor="black", alpha=0.85)
        plt.title("SignBridge AI — ISL Dataset Class Distribution", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("ISL Sign Class", fontsize=12, labelpad=8)
        plt.ylabel("Sample Count", fontsize=12, labelpad=8)
        plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, str(int(h)), ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plot_out = reports_dir / "class_distribution.png"
        plt.savefig(plot_out, dpi=300)
        plt.close()
        logger.info(f"Saved class distribution plot to: {plot_out}")

    return {
        "total_samples": total_samples,
        "total_classes": total_classes,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "min_samples": min_samples,
        "max_samples": max_samples,
        "avg_samples": avg_samples,
        "class_names": class_names,
    }


if __name__ == "__main__":
    compute_dataset_statistics()
