"""
DAY 15 & 16 -- SignBridge AI  Week 3
Sequence Collection: record 30-frame hand landmark sequences for LSTM training.

Sequence shape per sample : (30, 63)  →  30 frames × 63 hand features
Save location             : dataset/sequences/<sign>/sequence_<NNN>.npy

Controls
--------
SPACE   - start / stop recording the current sign
N       - advance to next sign
Q       - quit

Workflow per sign
-----------------
  1. Sign name shown on screen.
  2. Press SPACE to start a 30-frame recording (countdown shown).
  3. After 30 frames the sequence is auto-saved; repeat for next sample.
  4. Press N to advance; Q to quit.

Target: 50-100 sequences per sign.
"""

import cv2
import os
import sys
import time
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing.landmark_extractor import extract_and_normalize

# ---- Configuration -----------------------------------------------------------
SIGNS: list[str] = [
    "hello", "thank_you", "yes", "no", "help",
    "please", "sorry", "good", "bad", "stop",
    "eat", "drink", "water", "home", "school",
]

SEQUENCE_LENGTH: int = 30          # frames per sequence
TARGET_SEQUENCES: int = 75         # aim per sign
SEQUENCE_DIR:    str  = os.path.join("dataset", "sequences")

# Countdown frames before recording starts
COUNTDOWN_FRAMES: int = 45         # ~1.5 s at 30 fps


# ---- Helpers -----------------------------------------------------------------

def _next_seq_index(sign_dir: str) -> int:
    """Return the index for the next .npy file in sign_dir."""
    existing = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    return len(existing)


def _draw_hud(
    frame,
    sign: str,
    sign_idx: int,
    total: int,
    saved: int,
    target: int,
    state: str,          # "idle" | "countdown" | "recording"
    countdown: int = 0,
    frame_num: int = 0,
) -> None:
    h, w = frame.shape[:2]
    state_colors = {
        "idle":       (60, 60, 60),
        "countdown":  (0, 160, 255),
        "recording":  (0, 220, 60),
    }
    color = state_colors.get(state, (255, 255, 255))

    # Header
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, "SIGNBRIDGE AI -- Sequence Collector",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, f"Sign [{sign_idx + 1}/{total}]: {sign.upper()}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # State indicator
    if state == "countdown":
        msg = f"Get ready... {max(0, countdown // 15) + 1}"
        cv2.putText(frame, msg, (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    elif state == "recording":
        remaining = SEQUENCE_LENGTH - frame_num
        cv2.putText(frame, f"RECORDING  {frame_num:02d}/{SEQUENCE_LENGTH}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        # Frame bar
        bar_w = int((w - 40) * (frame_num / SEQUENCE_LENGTH))
        cv2.rectangle(frame, (20, 100), (w - 20, 112), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 100), (20 + bar_w, 112), color, -1)

    # Progress & controls
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Saved: {saved}/{target}",
                (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    cv2.putText(frame, "[SPACE] Record   [N] Next   [Q] Quit",
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    # Create directories
    for sign in SIGNS:
        os.makedirs(os.path.join(SEQUENCE_DIR, sign), exist_ok=True)

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
        return

    print("Sequence Collector started.")
    print(f"  Signs       : {SIGNS}")
    print(f"  Seq length  : {SEQUENCE_LENGTH} frames")
    print(f"  Target/sign : {TARGET_SEQUENCES}")
    print()

    sign_idx = 0
    state    = "idle"       # idle | countdown | recording
    countdown_ctr = 0
    frame_buffer: list[np.ndarray] = []

    while sign_idx < len(SIGNS):
        sign     = SIGNS[sign_idx]
        sign_dir = os.path.join(SEQUENCE_DIR, sign)
        saved    = _next_seq_index(sign_dir)

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        # Extract features (zeros if no hand)
        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
            )
            feats = extract_and_normalize(results.multi_hand_landmarks[0])
        else:
            feats = np.zeros(63, dtype=np.float32)

        # State machine
        if state == "idle":
            _draw_hud(frame, sign, sign_idx, len(SIGNS), saved, TARGET_SEQUENCES, "idle")

        elif state == "countdown":
            countdown_ctr -= 1
            _draw_hud(frame, sign, sign_idx, len(SIGNS), saved, TARGET_SEQUENCES,
                      "countdown", countdown=countdown_ctr)
            if countdown_ctr <= 0:
                state        = "recording"
                frame_buffer = []

        elif state == "recording":
            frame_buffer.append(feats)
            _draw_hud(frame, sign, sign_idx, len(SIGNS), saved, TARGET_SEQUENCES,
                      "recording", frame_num=len(frame_buffer))

            if len(frame_buffer) >= SEQUENCE_LENGTH:
                seq = np.array(frame_buffer, dtype=np.float32)   # (30, 63)
                idx = _next_seq_index(sign_dir)
                path = os.path.join(sign_dir, f"sequence_{idx:04d}.npy")
                np.save(path, seq)
                saved += 1
                print(f"  [{sign}] Saved sequence {idx:04d}  ({saved}/{TARGET_SEQUENCES})")
                state = "idle"

        cv2.imshow("SignBridge AI -- Sequence Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit.")
            break

        if key == ord(" ") and state == "idle":
            state         = "countdown"
            countdown_ctr = COUNTDOWN_FRAMES
            print(f"  [{sign}] Countdown started ...")

        if key == ord("n") and state == "idle":
            print(f"  [{sign}] Moving to next sign. Saved: {saved}")
            sign_idx += 1
            state = "idle"

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("\nCollection complete.")


if __name__ == "__main__":
    main()
