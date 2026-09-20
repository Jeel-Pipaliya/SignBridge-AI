"""
SignBridge AI - Dataset Exploration & EDA Module (Week 2 Day 4)
Performs exploratory data analysis on the raw ISL dataset:
- Evaluates class distribution and sample counts
- Checks for missing or unreadable images
- Verifies image dimensions and channels
- Generates publication-ready plots:
    * outputs/plots/class_distribution.png
    * outputs/plots/sample_grid.png
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.ingest_dataset import scan_raw_dataset, verify_image_integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure plot output directory exists
config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_dataset_statistics(class_images: Dict[str, List[Path]]) -> dict:
    """Compute numerical summaries and integrity metrics for the dataset."""
    total_samples = sum(len(imgs) for imgs in class_images.values())
    total_classes = len(class_images)

    if total_classes == 0:
        return {
            "total_classes": 0,
            "total_samples": 0,
            "counts": {},
            "corrupt_count": 0,
            "resolutions": set(),
        }

    counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    resolutions = set()
    corrupted = []

    # Sample images to check dimensions and decodability
    for cls, paths in class_images.items():
        for p in paths[:5]:  # sample 5 images per class for speed
            try:
                img = cv2.imread(str(p))
                if img is None:
                    corrupted.append(str(p))
                else:
                    resolutions.add((img.shape[1], img.shape[0]))  # (width, height)
            except Exception:
                corrupted.append(str(p))

    max_count = max(counts.values()) if counts else 0
    min_count = min(counts.values()) if counts else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else 0.0

    return {
        "total_classes": total_classes,
        "total_samples": total_samples,
        "counts": counts,
        "imbalance_ratio": imbalance_ratio,
        "corrupted_count": len(corrupted),
        "resolutions": sorted(list(resolutions)),
    }


def plot_class_distribution(counts: Dict[str, int], output_path: Path = config.PLOTS_DIR / "class_distribution.png") -> Path:
    """Generate and save class frequency bar chart."""
    plt.figure(figsize=(12, 6))
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")
    else:
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    classes = list(counts.keys())
    values = list(counts.values())

    if HAS_SEABORN:
        palette = sns.color_palette("viridis", len(classes))
    else:
        palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(classes)))

    bars = plt.bar(classes, values, color=palette, edgecolor="black", alpha=0.85)

    # Average line
    mean_val = np.mean(values) if values else 0
    plt.axhline(mean_val, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")

    # Add numeric labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(1, height * 0.015),
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("SignBridge AI — ISL Prototype Dataset Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sign Class Label", fontsize=12, labelpad=10)
    plt.ylabel("Number of Samples", fontsize=12, labelpad=10)
    plt.xticks(rotation=0, fontsize=11, fontweight="bold")
    plt.ylim(0, max(values) * 1.15 if values else 10)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Class distribution plot saved to: {output_path}")
    return output_path


def plot_sample_grid(class_images: Dict[str, List[Path]], output_path: Path = config.PLOTS_DIR / "sample_grid.png") -> Path:
    """Create a sample grid showing one representative image per class."""
    classes = sorted(list(class_images.keys()))
    num_classes = len(classes)
    if num_classes == 0:
        logger.warning("No classes found to plot sample grid.")
        return output_path

    cols = 4
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for idx, cls in enumerate(classes):
        ax = axes[idx]
        imgs = class_images[cls]
        if imgs:
            # Load first image
            bgr = cv2.imread(str(imgs[0]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"Class: '{cls}'\n({len(imgs)} imgs)", fontsize=11, fontweight="bold")
        ax.axis("off")

    # Turn off any remaining unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("SignBridge AI — ISL Prototype Class Representative Grid", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Sample grid plot saved to: {output_path}")
    return output_path


def run_exploration() -> dict:
    """Run end-to-end dataset exploratory data analysis."""
    print("=" * 65)
    print("  SignBridge AI - Exploratory Data Analysis (EDA) (Week 2 Day 4)")
    print("=" * 65)

    class_images = scan_raw_dataset(config.RAW_DATA_DIR)
    stats = compute_dataset_statistics(class_images)

    print(f"\n[Dataset Statistics]")
    print(f"  • Total Classes:        {stats['total_classes']}")
    print(f"  • Total Samples:        {stats['total_samples']}")
    print(f"  • Imbalance Ratio:      {stats['imbalance_ratio']:.2f} (1.00 = perfectly balanced)")
    print(f"  • Corrupted Files:      {stats['corrupted_count']}")
    print(f"  • Image Resolutions:    {stats['resolutions']}")

    print(f"\n[Per-Class Sample Counts]")
    for cls, count in sorted(stats["counts"].items()):
        print(f"  - Class '{cls:6s}': {count} samples")

    # Generate Plots
    if stats["total_classes"] > 0:
        dist_plot = plot_class_distribution(stats["counts"])
        grid_plot = plot_sample_grid(class_images)
        print(f"\n[Generated Visualizations]")
        print(f"  1. {dist_plot}")
        print(f"  2. {grid_plot}")

    return stats


if __name__ == "__main__":
    run_exploration()
