"""
DAY 11 -- SignBridge AI  Week 2
Dataset Collection Pipeline: collects ISL hand landmark samples into a CSV.

Each row in the CSV:
    label, feature_0, feature_1, ..., feature_62   (1 + 63 = 64 columns)

Run from the project root:
    python backend/data/collect_data.py

Controls:
    SPACE -- start/stop collecting for current sign
    N     -- move to next sign  (or finish current and loop)
    Q     -- quit

Workflow:
    1. Script shows sign name on screen.
    2. Press SPACE to toggle collection ON/OFF.
    3. Press N to move to the next sign in the list.
    4. Collect ~500-1000 samples per sign for a good baseline model.
"""

import cv2
import csv
import os
import sys
import time
import mediapipe as mp

# Allow running as: python backend/data/collect_data.py from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.preprocessing.landmark_extractor import extract_and_normalize


# ---- Configuration -----------------------------------------------------------
SIGNS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "help",
    "please",
    "sorry",
    "good",
    "bad",
    "stop",
]

TARGET_SAMPLES_PER_SIGN: int = 500          # aim for this many per sign
OUTPUT_FILE: str = os.path.join(
    "backend", "data", "processed", "landmarks.csv"
)


# ---- Helpers -----------------------------------------------------------------

def _count_existing(csv_path: str, sign: str) -> int:
    """Count how many rows for `sign` already exist in the CSV."""
    if not os.path.exists(csv_path):
        return 0
    count = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header
        for row in reader:
            if row and row[0] == sign:
                count += 1
    return count


def _ensure_header(csv_path: str) -> None:
    """Write the CSV header if the file does not exist yet."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"feature_{i}" for i in range(63)]
            writer.writerow(header)


def _draw_ui(
    frame,
    sign: str,
    sign_idx: int,
    total_signs: int,
    collected: int,
    target: int,
    collecting: bool,
) -> None:
    h, w = frame.shape[:2]
    state_color = (0, 60, 220) if collecting else (50, 50, 50)

    # Header bar
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, "SIGNBRIDGE AI -- Data Collection",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
    cv2.putText(frame, f"Sign [{sign_idx + 1}/{total_signs}]: {sign.upper()}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Progress bar
    ratio = min(collected / max(target, 1), 1.0)
    bar_w = int((w - 40) * ratio)
    cv2.rectangle(frame, (20, h - 55), (w - 20, h - 35), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, h - 55), (20 + bar_w, h - 35), (0, 200, 80), -1)

    cv2.putText(frame, f"Samples: {collected} / {target}",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Recording indicator
    status_text = "COLLECTING" if collecting else "PAUSED"
    cv2.putText(frame, status_text,
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

    # Controls
    cv2.putText(frame, "[SPACE] Start/Stop   [N] Next Sign   [Q] Quit",
                (w // 2 - 200, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (160, 160, 160), 1)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    _ensure_header(OUTPUT_FILE)

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        hands.close()
        return

    print(f"Dataset collection started.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Signs to collect: {SIGNS}")
    print(f"Target per sign : {TARGET_SAMPLES_PER_SIGN} samples")
    print()

    sign_idx = 0
    collecting = False

    with open(OUTPUT_FILE, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        while sign_idx < len(SIGNS):
            sign = SIGNS[sign_idx]
            collected = _count_existing(OUTPUT_FILE, sign)

            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

                # Collect sample if active
                if collecting:
                    feats = extract_and_normalize(results.multi_hand_landmarks[0])
                    writer.writerow([sign] + feats.tolist())
                    csv_file.flush()
                    collected += 1

                    if collected >= TARGET_SAMPLES_PER_SIGN:
                        collecting = False
                        print(f"  [{sign}] Target reached ({collected} samples).")

            _draw_ui(frame, sign, sign_idx, len(SIGNS),
                     collected, TARGET_SAMPLES_PER_SIGN, collecting)
            cv2.imshow("SignBridge AI -- Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quit.")
                break

            if key == ord(" "):
                collecting = not collecting
                state = "ON" if collecting else "OFF"
                print(f"  [{sign}] Collection {state}. Samples so far: {collected}")

            if key == ord("n"):
                collecting = False
                print(f"  [{sign}] Moving to next sign. Samples: {collected}")
                sign_idx += 1

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
