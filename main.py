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
        choices=["demo", "collect", "visualize", "train", "evaluate", "test"],
        help="Execution mode (default: 'demo')",
    )
    parser.add_argument("--label", type=str, default="HELLO", help="Label for data collection mode")
    parser.add_argument("--samples", type=int, default=30, help="Target samples for collection")

    args = parser.parse_args()
    print_banner()

    if args.mode == "demo":
        from realtime.realtime_demo import run_realtime_recognition
        run_realtime_recognition()

    elif args.mode == "collect":
        from data.collect_samples import run_collector
        run_collector(initial_label=args.label, target_samples=args.samples)

    elif args.mode == "visualize":
        from ai.visualize_landmarks import run_landmark_visualizer
        run_landmark_visualizer()

    elif args.mode == "train":
        from ai.train import run_training
        run_training()

    elif args.mode == "evaluate":
        from ai.evaluate import evaluate_trained_model
        evaluate_trained_model()

    elif args.mode == "test":
        import unittest
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(config.PROJECT_ROOT / "tests"), pattern="test_week*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)


if __name__ == "__main__":
    main()
