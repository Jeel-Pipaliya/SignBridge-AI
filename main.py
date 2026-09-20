"""
SignBridge AI - Master Application Entry Point
Intelligent Real-Time AI Communication Platform for Deaf & Hard-of-Hearing People

Usage:
    python main.py                    # Launch live webcam recognition demo (default)
    python main.py --mode demo        # Launch live recognition demo
    python main.py --mode collect     # Launch data collector
    python main.py --mode visualize   # Launch 21-joint landmark inspector
    python main.py --mode train       # Train & benchmark ML models
    python main.py --mode evaluate    # Evaluate model on test set
    python main.py --mode test        # Run automated test suite
"""

import sys
import argparse
from pathlib import Path

# Safe encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def print_banner() -> None:
    print("=" * 68)
    print("  [*] SIGNBRIDGE AI - Real-Time Indian Sign Language Platform")
    print("  Empowering Deaf & Hard-of-Hearing Communication with AI")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge AI Master Launcher")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=[
            "demo", "collect", "visualize", "train", "evaluate", "stats", "validate", "test",
            "dynamic-collect", "dynamic-train", "dynamic-evaluate", "realtime",
        ],
        help="Execution mode (default: 'demo')",
    )
    parser.add_argument("--label", type=str, default="HELLO", help="Label for data collection mode")
    parser.add_argument("--samples", type=int, default=30, help="Target samples for collection")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode for real-time engine")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi"], help="Language for speech synthesis (en or hi)")
    parser.add_argument("--auto-speak", action="store_true", help="Enable automatic speech on recognized sign commit")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for training")

    args = parser.parse_args()
    print_banner()

    if args.mode in ("demo", "realtime"):
        from ai.inference.realtime import ISLRealtimeEngine
        engine = ISLRealtimeEngine(language=args.lang, auto_speak=args.auto_speak)
        if args.benchmark:
            engine.run_benchmark()
        else:
            engine.run()

    elif args.mode == "dynamic-collect":
        from ai.data_collection.collect_dynamic import DynamicDataCollector
        collector = DynamicDataCollector(initial_class=args.label, target_sequences=args.samples)
        collector.run()

    elif args.mode == "dynamic-train":
        from ai.dynamic.train import run_dynamic_training
        run_dynamic_training(epochs=args.epochs)

    elif args.mode == "dynamic-evaluate":
        from ai.dynamic.evaluate import evaluate_dynamic_model
        evaluate_dynamic_model()

    elif args.mode == "collect":
        from data.collect_samples import run_collector
        run_collector(initial_label=args.label, target_samples=args.samples)

    elif args.mode == "visualize":
        from ai.visualize_landmarks import run_landmark_visualizer
        run_landmark_visualizer()

    elif args.mode == "train":
        from ai.training.train import run_training
        run_training()

    elif args.mode == "evaluate":
        from ai.evaluation.evaluate import evaluate_model
        evaluate_model()

    elif args.mode == "stats":
        from ai.data_analysis.dataset_statistics import compute_dataset_statistics
        compute_dataset_statistics()

    elif args.mode == "validate":
        from ai.data_analysis.validate_dataset import validate_dataset
        validate_dataset()

    elif args.mode == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(config.PROJECT_ROOT / "tests"), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)


if __name__ == "__main__":
    main()
