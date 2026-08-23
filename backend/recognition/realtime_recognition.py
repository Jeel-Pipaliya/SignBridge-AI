"""
DAY 13 + 14 -- SignBridge AI  Week 2
Real-Time ISL Recognition: webcam -> landmarks -> Random Forest -> sign + confidence.

Features (Day 14):
  - Confidence threshold (0.75) -- only shows prediction when confident enough
  - Prediction smoothing (majority vote over last 15 frames)
  - "Unknown" state for low-confidence or no-hand frames

Run from the project root:
    python backend/recognition/realtime_recognition.py

Controls:
    Q -- quit
"""

import cv2
import os
import sys
import time
import joblib
import numpy as np
import mediapipe as mp
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.preprocessing.landmark_extractor import extract_and_normalize


# ---- Configuration -----------------------------------------------------------
MODEL_PATH:          str   = os.path.join("backend", "models", "isl_model.pkl")
CONFIDENCE_THRESHOLD: float = 0.75   # Day 14: ignore low-confidence predictions
SMOOTHING_WINDOW:     int   = 15     # Day 14: majority vote over N frames


# ---- Prediction smoother (Day 14) -------------------------------------------

class PredictionSmoother:
    """
    Maintains a rolling window of predictions and returns the majority vote.
    Prevents flickering when the webcam produces noisy frames.
    """

    def __init__(self, window: int = 15) -> None:
        self.window = window
        self._history: list[str] = []

    def update(self, label: str) -> str:
        self._history.append(label)
        if len(self._history) > self.window:
            self._history.pop(0)
        return Counter(self._history).most_common(1)[0][0]

    def reset(self) -> None:
        self._history = []


# ---- UI helpers -------------------------------------------------------------

def _draw_hud(
    frame,
    fps: float,
    prediction: str,
    confidence: float,
    hand_detected: bool,
    threshold: float,
) -> None:
    h, w = frame.shape[:2]

    # Header
    cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.putText(frame, "SIGNBRIDGE AI -- Live Recognition",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Prediction box
    if hand_detected:
        box_color = (0, 180, 60) if confidence >= threshold else (80, 80, 80)
        cv2.rectangle(frame, (0, h - 110), (w, h), (0, 0, 0), -1)

        pred_text = prediction.upper()
        conf_pct  = confidence * 100

        cv2.putText(frame, pred_text,
                    (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, box_color, 3)
        cv2.putText(frame, f"Confidence: {conf_pct:.1f}%",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)

        # Confidence bar
        bar_w = int((w - 40) * min(confidence, 1.0))
        cv2.rectangle(frame, (20, h - 18), (w - 20, h - 6), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, h - 18), (20 + bar_w, h - 6), box_color, -1)
    else:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "No hand detected",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

    # Controls bar
    cv2.putText(frame, "[Q] Quit",
                (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ---- Main -------------------------------------------------------------------

def main() -> None:
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run backend/models/train_model.py first.")
        return

    model    = joblib.load(MODEL_PATH)
    smoother = PredictionSmoother(window=SMOOTHING_WINDOW)

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

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

    print("Real-time ISL Recognition started. Press Q to quit.")
    print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"  Smoothing window     : {SMOOTHING_WINDOW} frames")

    prev_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # FPS
        now  = time.time()
        fps  = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        prediction  = "Unknown"
        confidence  = 0.0
        hand_detected = bool(results.multi_hand_landmarks)

        if hand_detected:
            hand = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Extract features
            feats = extract_and_normalize(hand).reshape(1, -1)

            # Predict
            raw_pred     = model.predict(feats)[0]
            probabilities = model.predict_proba(feats)[0]
            confidence    = float(np.max(probabilities))

            # Day 14: threshold + smoothing
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = smoother.update(raw_pred)
            else:
                prediction = smoother.update("Unknown")
        else:
            smoother.reset()

        _draw_hud(frame, fps, prediction, confidence, hand_detected,
                  CONFIDENCE_THRESHOLD)

        cv2.imshow("SignBridge AI -- Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
