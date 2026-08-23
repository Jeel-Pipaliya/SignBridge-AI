# -*- coding: utf-8 -*-
"""
DAY 1 — SignBridge AI
Environment verification script.

Run:
    python tests/test_setup.py
"""

import sys
import os


def check_python_version() -> bool:
    version = sys.version_info
    print(f"Python: {sys.version}")
    ok = version.major == 3 and version.minor >= 9
    print("  [OK]  Python version" if ok else "  [!!] Python 3.9+ required")
    return ok


def check_packages() -> bool:
    packages = {
        "cv2":        "opencv-python",
        "mediapipe":  "mediapipe",
        "numpy":      "numpy",
        "pandas":     "pandas",
        "matplotlib": "matplotlib",
    }

    all_ok = True
    for module, pip_name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            print(f"  [OK]  {pip_name}: {version}")
        except ImportError:
            print(f"  [!!]  {pip_name} NOT FOUND -- run: pip install {pip_name}")
            all_ok = False

    return all_ok


def check_project_structure() -> None:
    folders = [
        "ai/datasets/raw",
        "ai/datasets/processed",
        "ai/datasets/collected",
        "ai/preprocessing",
        "ai/inference",
        "ai/evaluation",
        "tests",
    ]
    print("\nProject folders:")
    for folder in folders:
        exists = os.path.isdir(folder)
        status = "[OK]" if exists else "[--]"
        print(f"  {status}  {folder}")


def main() -> None:
    print("=" * 52)
    print("  SignBridge AI -- Week 1 Environment Check")
    print("=" * 52)

    python_ok = check_python_version()

    print("\nPackages:")
    packages_ok = check_packages()

    check_project_structure()

    print()
    print("=" * 52)
    if python_ok and packages_ok:
        print("  [READY]  SignBridge Week 1 environment is set up!")
    else:
        print("  [ERROR]  Fix the issues above, then re-run.")
    print("=" * 52)


if __name__ == "__main__":
    main()
