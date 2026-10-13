"""
DAY 17 -- SignBridge AI  Week 3
Data Preprocessing: load all sequence .npy files → X.npy, y.npy, labels.json

Input  : dataset/sequences/<sign>/sequence_NNNN.npy   shape (30, 63)
Output : dataset/processed/X.npy        shape (N, 30, 63)   float32
         dataset/processed/y.npy        shape (N,)           int32
         dataset/processed/labels.json  {"0": "hello", ...}

Run:
    python training/preprocess.py
"""

import os
import sys
import json
import numpy as np

SEQUENCE_DIR   = os.path.join("dataset", "sequences")
PROCESSED_DIR  = os.path.join("dataset", "processed")
X_PATH         = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH         = os.path.join(PROCESSED_DIR, "y.npy")
LABELS_PATH    = os.path.join(PROCESSED_DIR, "labels.json")

SEQUENCE_LENGTH = 30
NUM_FEATURES    = 63


def main() -> None:
    print("=" * 55)
    print("  SignBridge AI -- Data Preprocessing (Day 17)")
    print("=" * 55)

    if not os.path.isdir(SEQUENCE_DIR):
        print(f"ERROR: Sequence directory not found: {SEQUENCE_DIR}")
        print("Run training/collect_sequences.py first.")
        sys.exit(1)

    # Discover sign classes
    signs = sorted([
        d for d in os.listdir(SEQUENCE_DIR)
        if os.path.isdir(os.path.join(SEQUENCE_DIR, d))
    ])

    if not signs:
        print("ERROR: No sign directories found.")
        sys.exit(1)

    label_map = {str(i): s for i, s in enumerate(signs)}
    sign_to_idx = {s: i for i, s in enumerate(signs)}

    print(f"\nSigns found ({len(signs)}): {signs}")
    print(f"\nLoading sequences ...")

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []
    skipped = 0

    for sign in signs:
        sign_dir = os.path.join(SEQUENCE_DIR, sign)
        files    = sorted([
            f for f in os.listdir(sign_dir) if f.endswith(".npy")
        ])

        loaded = 0
        for fname in files:
            path = os.path.join(sign_dir, fname)
            seq  = np.load(path)

            # Validate shape
            if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                print(f"  SKIP  {path}  bad shape {seq.shape}")
                skipped += 1
                continue

            X_list.append(seq)
            y_list.append(sign_to_idx[sign])
            loaded += 1

        print(f"  {sign:15s}  {loaded:4d} sequences")

    if not X_list:
        print("\nERROR: No valid sequences found.")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)   # (N, 30, 63)
    y = np.array(y_list,  dtype=np.int32)     # (N,)

    # Shuffle
    rng   = np.random.default_rng(seed=42)
    idx   = rng.permutation(len(X))
    X, y  = X[idx], y[idx]

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print()
    print(f"Saved:")
    print(f"  X  shape : {X.shape}   -> {X_PATH}")
    print(f"  y  shape : {y.shape}   -> {Y_PATH}")
    print(f"  labels   :             -> {LABELS_PATH}")
    if skipped:
        print(f"  Skipped  : {skipped} sequences (bad shape)")
    print()
    print(f"Class distribution:")
    for i, sign in enumerate(signs):
        count = int(np.sum(y == i))
        print(f"  {sign:15s}  {count} samples")
    print("=" * 55)


if __name__ == "__main__":
    main()
