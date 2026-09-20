"""
SignBridge AI - Master Real-Time ISL Recognition Engine (Week 4 Phase 7, 8, 9)
Usage:
    python -m ai.inference.realtime
    python -m ai.inference.realtime --camera 0 --threshold 0.70
    python -m ai.inference.realtime --benchmark
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import deque, Counter
import cv2
import numpy as np

# Safe encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from ai.predict import SignPredictor, PredictionResult
from ai.utils.logger import setup_logger
from ai.utils.config import load_config
from realtime.camera import Camera, CameraError
from realtime.recognizer import TemporalSmoother, TextAccumulator

logger = setup_logger("RealtimeInference")


class ISLRealtimeEngine:
    """
    Complete Real-Time Indian Sign Language Inference Engine.
    Coordinates camera acquisition, landmark detection, normalized feature extraction,
    model inference, temporal smoothing, and accessible UI rendering.
    """

    def __init__(
        self,
        camera_index: int = 0,
        confidence_threshold: Optional[float] = None,
        smoothing_window: Optional[int] = None,
    ):
        self.cfg = load_config()
        inf_cfg = self.cfg.get("inference", {})

        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(inf_cfg.get("confidence_threshold", 0.70))
        )
        self.smoothing_window = (
            smoothing_window
            if smoothing_window is not None
            else int(inf_cfg.get("smoothing_window", 5))
        )
        self.camera_index = camera_index

        logger.info(f"Initializing ISL Real-Time Engine (Threshold: {self.confidence_threshold:.0%}, Window: {self.smoothing_window})")

        self.predictor = SignPredictor(confidence_threshold=self.confidence_threshold)
        self.smoother = TemporalSmoother(window_size=self.smoothing_window)
        self.accumulator = TextAccumulator(
            confirmation_frames=int(inf_cfg.get("consecutive_confirm_frames", 6)),
            debounce_seconds=float(inf_cfg.get("debounce_seconds", 0.8)),
        )

        self.classes = self.predictor.classes

    def process_frame(
        self,
        frame: np.ndarray,
        detector: HandDetector,
    ) -> Tuple[np.ndarray, Optional[str], float, float, bool]:
        """
        Process single video frame through the full AI pipeline.
        Returns:
            annotated_frame: np.ndarray
            smoothed_sign: Optional[str]
            confidence: float
            latency_ms: float
            hand_detected: bool
        """
        t_start = time.perf_counter()

        # 1. MediaPipe Hand Landmark Detection
        annotated_frame, hands = detector.process(frame, draw=True)
        hand_detected = len(hands) > 0

        raw_prediction = None
        if hand_detected:
            # 2. Extract 63D invariant normalized features and infer
            primary_hand = hands[0]
            raw_prediction = self.predictor.predict_landmarks(primary_hand.landmark_list)

        # 3. Temporal Stabilization & Confidence Filtering
        if raw_prediction is not None and raw_prediction.is_valid:
            smoothed_sign, conf = self.smoother.update(raw_prediction)
        else:
            smoothed_sign, conf = self.smoother.update(
                PredictionResult("...", "...", 0.0, False, {})
            )

        # 4. Text Accumulation & Debounce
        if smoothed_sign != "...":
            self.accumulator.update(smoothed_sign)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return annotated_frame, smoothed_sign, conf, latency_ms, hand_detected

    def draw_hud(
        self,
        frame: np.ndarray,
        sign: str,
        confidence: float,
        fps: float,
        latency_ms: float,
        hand_detected: bool,
    ) -> np.ndarray:
        """
        Renders high-contrast, accessible HUD overlay.
        """
        h, w = frame.shape[:2]

        # Top Header Bar (Translucent Dark Charcoal)
        header_bar = frame.copy()
        cv2.rectangle(header_bar, (0, 0), (w, 85), (20, 24, 30), -1)
        cv2.addWeighted(header_bar, 0.80, frame, 0.20, 0, frame)

        # Title
        cv2.putText(
            frame,
            "SIGNBRIDGE AI — Real-Time ISL Recognition",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Diagnostics: FPS & Latency
        diag_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms"
        cv2.putText(
            frame,
            diag_text,
            (w - 290, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

        # Hand Tracking Status Indicator (Accessible text + color dot)
        if hand_detected:
            status_color = (0, 255, 0)
            status_str = "[HAND DETECTED]"
        else:
            status_color = (120, 120, 120)
            status_str = "[NO HAND DETECTED]"

        cv2.putText(
            frame,
            status_str,
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Main Sign Display & Confidence
        if hand_detected and sign != "...":
            display_sign = sign
            sign_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 200, 255)
            conf_str = f"Conf: {confidence:.0%}"
        elif hand_detected:
            display_sign = "IDENTIFYING..."
            sign_color = (0, 200, 255)
            conf_str = f"Threshold: {self.confidence_threshold:.0%}"
        else:
            display_sign = "WAITING FOR HAND..."
            sign_color = (160, 160, 160)
            conf_str = "No Signal"

        cv2.putText(
            frame,
            f"Sign: {display_sign}",
            (250, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            sign_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            conf_str,
            (w - 200, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Bottom Sentence Text Accumulator Bar
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 80), (w, h), (16, 20, 26), -1)
        cv2.addWeighted(bot_bar, 0.85, frame, 0.15, 0, frame)

        accum_text = self.accumulator.get_text()
        cv2.putText(
            frame,
            "Accumulated Sentence:",
            (15, h - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            accum_text if accum_text else "...",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Controls Hint
        controls = "[SPACE] Add Space | [BACKSPACE] Delete | [C] Clear | [Q] Quit"
        cv2.putText(
            frame,
            controls,
            (w - 430, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        return frame

    def run(self) -> None:
        """Launch interactive webcam recognition loop."""
        print("=" * 65)
        print("  SIGNBRIDGE AI — Real-Time Indian Sign Language Recognition")
        print("  Webcam Pipeline: Hands -> Normalization -> Classifier -> HUD")
        print("=" * 65)
        print("Loading AI models...")

        window_name = "SignBridge AI — Live ISL Recognition (Week 4)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            with Camera(camera_index=self.camera_index) as cam, HandDetector() as detector:
                print("Camera initialized successfully. Press [Q] to quit.")
                while True:
                    success, frame = cam.read()
                    if not success:
                        print("[Warning] Could not read frame from camera.")
                        break

                    annotated, sign, conf, latency, hand_detected = self.process_frame(frame, detector)

                    hud_frame = self.draw_hud(
                        frame=annotated,
                        sign=sign or "...",
                        confidence=conf,
                        fps=cam.fps,
                        latency_ms=latency,
                        hand_detected=hand_detected,
                    )

                    cv2.imshow(window_name, hud_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("Exiting real-time session...")
                        break
                    elif key == 32:  # SPACE
                        self.accumulator.add_space()
                    elif key in (8, 127):  # BACKSPACE
                        self.accumulator.backspace()
                    elif key in (ord("c"), ord("C")):
                        self.accumulator.clear()

        except CameraError as err:
            print(f"\n[CAMERA ERROR] {err}")
            print("Please check camera permissions, connection, or camera index.\n")
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
        finally:
            cv2.destroyAllWindows()
            print("SignBridge AI session terminated cleanly.")

    def run_benchmark(self, num_frames: int = 50) -> Dict[str, float]:
        """
        Headless benchmark verifying pipeline latency and throughput.
        """
        print(f"\nRunning headless benchmark on {num_frames} synthetic frames...")
        latencies = []
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with HandDetector() as detector:
            for _ in range(num_frames):
                t0 = time.perf_counter()
                _, _, _, lat, _ = self.process_frame(dummy_frame, detector)
                latencies.append(lat)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        effective_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        print(f"Benchmark Results:")
        print(f"  • Average Latency: {avg_latency:.2f} ms")
        print(f"  • 95th Percentile: {p95_latency:.2f} ms")
        print(f"  • Throughput:      {effective_fps:.1f} FPS")

        return {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "effective_fps": effective_fps,
        }


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Real-Time Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--window", type=int, default=None, help="Smoothing window size")
    parser.add_argument("--benchmark", action="store_true", help="Run latency/FPS benchmark without opening camera")

    args = parser.parse_args()

    engine = ISLRealtimeEngine(
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        smoothing_window=args.window,
    )

    if args.benchmark:
        engine.run_benchmark()
    else:
        engine.run()


if __name__ == "__main__":
    main()
