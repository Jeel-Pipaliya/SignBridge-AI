"""
SignBridge AI - Master Training Entry Point (Week 4 Phase 4)
Usage:
    python -m ai.training.train
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.training.trainer import Trainer
from ai.utils.logger import setup_logger

logger = setup_logger("TrainMaster")


def run_training() -> dict:
    """Execute complete training pipeline."""
    print("=" * 75)
    print("  SIGNBRIDGE AI — ISL Classifier Model Training Pipeline (Week 4)")
    print("=" * 75)

    trainer = Trainer()
    results = trainer.train()

    champion_name = results["champion_name"]
    metrics = results["benchmark_results"][champion_name]

    print("\n" + "=" * 75)
    print("  Training Completed Successfully")
    print(f"  • Selected Model Architecture: {champion_name}")
    print(f"  • Training Accuracy:           {metrics['train_acc']:.2%}")
    print(f"  • Validation Accuracy:         {metrics['val_acc']:.2%}")
    print(f"  • Test Accuracy:               {metrics['test_acc']:.2%}")
    print(f"  • Test Macro F1:               {metrics['test_f1']:.4f}")
    print("=" * 75 + "\n")

    return results


if __name__ == "__main__":
    run_training()
