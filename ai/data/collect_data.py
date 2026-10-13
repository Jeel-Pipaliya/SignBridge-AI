"""
DAY 6 — SignBridge AI
Data Collection Pipeline: records sign-language sequences and saves them as .npy files.

Run from the project root:
    python -m ai.datasets.collect_data

What it does:
  1. Asks for a sign name (e.g. HELLO).
  2. Opens the webcam.
  3. Shows a "ready" screen — press SPACE to start recording.
  4. Records exactly SEQUENCE_LENGTH frames of MediaPipe Holistic landmarks.
  5. Saves the sequence as a .npy file in ai/datasets/collected/<SIGN>/.
  6. Repeats for NUM_SEQUENCES sequences.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# Allow running as `python ai/datasets/collect_data.py` from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai.preprocessing.landmark_extractor import extract_landmarks
from ai.preprocessing.normalize import normalize_landmarks


# ─── Configuration ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH: int = 30    # frames per sequence
NUM_SEQUENCES:   int = 5     # sequences to collect per run (can repeat the script)
COLLECTED_DIR:   str = os.path.join("ai", "datasets", "collected")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _next_seq_number(sign_dir: str) -> int:
    """Return the next available sequence number in sign_dir."""
    existing = [
        f for f in os.listdir(sign_dir)
        if f.startswith("sequence_") and f.endswith(".npy")
    ]
    return len(existing) + 1


def _draw_ready_screen(frame, sign_name: str, seq_num: int, total: int) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, "SIGNBRIDGE AI", (w // 2 - 160, h // 2 - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 120), 3)
    cv2.putText(frame, f"Sign: {sign_name}", (w // 2 - 120, h // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Sequence {seq_num} / {total}", (w // 2 - 110, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, "Press SPACE to start", (w // 2 - 150, h // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    cv2.putText(frame, "Press Q to quit", (w // 2 - 100, h // 2 + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)


def _draw_recording_screen(frame, sign_name: str, collected: int, total: int) -> None:
    h, w = frame.shape[:2]
    ratio = collected / total
    bar_w = int((w - 40) * ratio)

    # Progress bar
    cv2.rectangle(frame, (20, h - 50), (w - 20, h - 25), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 25), (0, 220, 80), -1)

    cv2.putText(frame, f"RECORDING  {sign_name}  {collected}/{total}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
    cv2.putText(frame, "[Q] Cancel", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ─── Core collection function ─────────────────────────────────────────────────

def collect_sign(sign_name: str, sequence_number: int, holistic, cap) -> bool:
    """
    Record one sequence for `sign_name`.

    Returns:
        True  — sequence saved successfully.
        False — user cancelled (Q pressed).
    """
    sign_dir = os.path.join(COLLECTED_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    # ── Wait for SPACE ────────────────────────────────────────────────────────
    while True:
        success, frame = cap.read()
        if not success:
            return False

        frame = cv2.flip(frame, 1)
        _draw_ready_screen(frame, sign_name, sequence_number, NUM_SEQUENCES)
        cv2.imshow("SignBridge — Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return False

    # ── Record frames ─────────────────────────────────────────────────────────
    sequence: list[np.ndarray] = []

    while len(sequence) < SEQUENCE_LENGTH:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        # Extract + normalize
        landmarks = extract_landmarks(results)
        landmarks = normalize_landmarks(landmarks)
        sequence.append(landmarks)

        # Draw holistic landmarks for live feedback
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)

        _draw_recording_screen(frame, sign_name, len(sequence), SEQUENCE_LENGTH)
        cv2.imshow("SignBridge — Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Recording cancelled.")
            return False

    # ── Save ──────────────────────────────────────────────────────────────────
    if len(sequence) == SEQUENCE_LENGTH:
        arr = np.array(sequence, dtype=np.float32)   # (30, 225)
        filename = os.path.join(sign_dir, f"sequence_{sequence_number:03d}.npy")
        np.save(filename, arr)
        print(f"  Saved: {filename}  shape={arr.shape}")
        return True

    print("  Incomplete sequence — not saved.")
    return False


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(COLLECTED_DIR, exist_ok=True)

    sign_name = input("Enter sign name (e.g. HELLO): ").strip().upper()
    if not sign_name:
        print("No sign name entered. Exiting.")
        return

    sign_dir = os.path.join(COLLECTED_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)
    start_index = _next_seq_number(sign_dir)

    print(f"\nCollecting {NUM_SEQUENCES} sequences for '{sign_name}'.")
    print(f"Starting from sequence #{start_index}.\n")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        holistic.close()
        return

    saved = 0
    for i in range(NUM_SEQUENCES):
        seq_num = start_index + i
        print(f"[{i + 1}/{NUM_SEQUENCES}] Preparing sequence {seq_num} …")
        ok = collect_sign(sign_name, seq_num, holistic, cap)
        if not ok:
            print("Aborted.")
            break
        saved += 1

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    print(f"\nDone. {saved} sequence(s) saved to: {sign_dir}")


if __name__ == "__main__":
    main()
