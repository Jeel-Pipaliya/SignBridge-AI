"""
SignBridge AI - Complete Real-Time Recognition Application (Week 3 Day 4 & 7)
The final capstone demonstration connecting:
1. Live Camera Stream (OpenCV)
2. Hand & 21 3D Landmark Detection (MediaPipe)
3. Invariant 63D Feature Normalizer (Translation & Scale)
4. Trained Machine Learning Classifier (SVM / Random Forest)
5. Temporal Sliding-Window Smoothing (Majority Voting)
6. Dynamic Sentence Text Accumulator with Interactive Controls

Controls:
    [SPACE]     : Insert space in recognized text
    [BACKSPACE] : Delete last token
    [C]         : Clear accumulated text buffer
    [Q] / [ESC] : Quit application
"""

import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Safe encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from realtime.camera import Camera, CameraError
from ai.hand_detection import HandDetector
from ai.feature_extraction import extract_features
from ai.predict import SignPredictor, PredictionResult
from realtime.recognizer import TemporalSmoother, TextAccumulator


def draw_ui_overlay(
    frame: np.ndarray,
    smoothed_sign: str,
    confidence: float,
    fps: float,
    accumulated_text: str,
    hand_count: int,
    candidate_sign: Optional[str] = None,
    candidate_conf: float = 0.0,
) -> np.ndarray:
    """
    Draws a modern HUD overlay with:
    - Header banner with FPS & hand tracking status
    - Real-time detected sign and confidence meter
    - Accumulated text banner at bottom
    - Keyboard controls legend
    """
    h, w = frame.shape[:2]

    # 1. Top HUD Bar (Translucent)
    top_bar = frame.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 85), (20, 24, 30), -1)
    cv2.addWeighted(top_bar, 0.75, frame, 0.25, 0, frame)

    # Title & FPS
    cv2.putText(
        frame,
        "SIGNBRIDGE AI - ISL Recognition",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 130, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Active Sign & Confidence
    is_detecting = hand_count > 0 and smoothed_sign != "..."
    sign_color = (0, 255, 0) if is_detecting else (160, 160, 160)

    if is_detecting:
        display_sign = smoothed_sign
    elif hand_count == 0:
        display_sign = "NO HAND"
    elif candidate_sign and candidate_sign != "Unknown":
        display_sign = f"DETECTING... ({candidate_sign} {candidate_conf:.0%})"
        sign_color = (0, 200, 255)  # Amber while accumulating/voting
    else:
        display_sign = "IDENTIFYING..."

    cv2.putText(
        frame,
        f"Sign: {display_sign}",
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        sign_color,
        2,
        cv2.LINE_AA,
    )

    if is_detecting:
        conf_text = f"Confidence: {confidence:.0%}"
        cv2.putText(
            frame,
            conf_text,
            (w - 230, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Confidence Bar
        bar_x, bar_y, bar_w, bar_h = w - 230, 74, 210, 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        filled_w = int(bar_w * min(1.0, confidence))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)

    # 2. Bottom Text Accumulation Bar
    bot_bar = frame.copy()
    cv2.rectangle(bot_bar, (0, h - 80), (w, h), (15, 18, 22), -1)
    cv2.addWeighted(bot_bar, 0.85, frame, 0.15, 0, frame)

    # Text Display
    cv2.putText(
        frame,
        "Recognized Text:",
        (15, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        accumulated_text,
        (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Keyboard Controls Legend
    controls_text = "[SPACE] Space | [BACKSPACE] Delete | [C] Clear | [Q] Quit"
    cv2.putText(
        frame,
        controls_text,
        (w - 430, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_realtime_recognition() -> None:
    """Main execution loop for SignBridge AI real-time recognition system."""
    print("=" * 65)
    print("  SIGNBRIDGE AI - Real-Time Indian Sign Language Recognition")
    print("  Week 3 Milestone Prototype Demonstration")
    print("=" * 65)

    # Verify model presence
    if not config.MODEL_PATH.exists():
        print(f"\n[ERROR] Model checkpoint not found at: {config.MODEL_PATH}")
        print("Please train the model first by running: python ai/train.py\n")
        return

    try:
        predictor = SignPredictor()
        smoother = TemporalSmoother()
        accumulator = TextAccumulator()
    except Exception as exc:
        print(f"[ERROR] Failed to initialize AI inference engine: {exc}")
        return

    print("AI Models and Smoothing Engine Loaded Successfully.")
    print("Opening camera interface... Press 'q' or 'ESC' to exit.")

    window_name = "SignBridge AI - Live ISL Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        with Camera() as cam, HandDetector() as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    print("[Warning] Failed to read from webcam.")
                    break

                # 1. Hand Landmark Detection
                annotated_frame, hands = detector.process(frame, draw=True)

                raw_prediction = None
                if hands:
                    primary_hand = hands[0]
                    # 2. Invariant 63D Feature Extraction & Prediction
                    raw_prediction = predictor.predict_landmarks(primary_hand.landmark_list)

                # 3. Temporal Smoothing (Sliding-Window Majority Voting)
                if raw_prediction is not None:
                    smoothed_sign, smoothed_conf = smoother.update(raw_prediction)
                else:
                    smoothed_sign, smoothed_conf = smoother.update(
                        PredictionResult("...", "...", 0.0, False, {})
                    )

                # 4. Text Formation & Accumulation
                accumulator.update(smoothed_sign)

                cand_lbl = raw_prediction.raw_label if raw_prediction else None
                cand_cnf = raw_prediction.confidence if raw_prediction else 0.0

                display_frame = draw_ui_overlay(
                    frame=annotated_frame,
                    smoothed_sign=smoothed_sign,
                    confidence=smoothed_conf,
                    fps=cam.fps,
                    accumulated_text=accumulator.get_text(),
                    hand_count=len(hands),
                    candidate_sign=cand_lbl,
                    candidate_conf=cand_cnf,
                )

                cv2.imshow(window_name, display_frame)

                # 6. Interactive Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (config.KEY_EXIT, 27):  # 'q' or ESC
                    break
                elif key == config.KEY_SPACE:
                    accumulator.add_space()
                elif key in (config.KEY_BACKSPACE, 127):
                    accumulator.backspace()
                elif key == config.KEY_CLEAR:
                    accumulator.clear()

    except CameraError as err:
        print(f"\n[CAMERA HARDWARE ERROR] {err}")
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 65)
        print("  SignBridge AI Session Closed.")
        print("=" * 65)


if __name__ == "__main__":
    run_realtime_recognition()
