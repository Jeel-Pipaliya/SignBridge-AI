"""
SignBridge AI - Master Real-Time Recognition & Temporal Speech Engine (Week 5)
Integrates:
  1. MediaPipe Hand Landmark Detection
  2. Multi-Modal Static + Dynamic Fusion Engine (MLP + Bi-LSTM)
  3. Consecutive & Cooldown Token Accumulation
  4. ISL Grammar Reconstruction (Natural English)
  5. Multilingual Translation (English <-> Hindi)
  6. Offline Text-to-Speech (Non-Blocking Audio Output)
  7. High-Contrast Accessible Real-Time HUD

Usage:
  python -m ai.inference.realtime
  python -m ai.inference.realtime --camera 0 --threshold 0.70
  python -m ai.inference.realtime --benchmark
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
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
from ai.predict import SignPredictor
from ai.dynamic.predict import DynamicSignPredictor
from ai.fusion.recognizer_fusion import StaticDynamicFusionEngine, RecognitionResult
from ai.fusion.token_buffer import TokenBuffer
from translation.isl_grammar import ISLGrammarEngine, GrammarResult
from translation.translator import translate_sentence
from speech.tts import TextToSpeechEngine
from ai.utils.logger import setup_logger
from ai.utils.config import load_config
from realtime.camera import Camera, CameraError

logger = setup_logger("RealtimeInference")


class ISLRealtimeEngine:
    """
    Complete Real-Time Indian Sign Language Inference and Communication Platform.
    Coordinates camera acquisition, landmark detection, normalized feature extraction,
    static/dynamic model fusion, grammar reconstruction, and speech synthesis.
    """

    def __init__(
        self,
        camera_index: int = 0,
        confidence_threshold: Optional[float] = None,
        smoothing_window: Optional[int] = None,
        language: str = "en",
        auto_speak: bool = False,
    ):
        self.cfg = load_config()
        inf_cfg = self.cfg.get("inference", {})
        dyn_cfg = self.cfg.get("dynamic", {})
        tts_cfg = self.cfg.get("tts", {})

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
        self.current_language = language
        self.auto_speak = auto_speak or bool(tts_cfg.get("auto_speak", False))

        logger.info(
            f"Initializing ISL Real-Time Engine (Threshold: {self.confidence_threshold:.0%}, "
            f"Window: {self.smoothing_window}, Lang: {self.current_language})"
        )

        # Predictors & Fusion Engine
        self.static_predictor = SignPredictor(confidence_threshold=self.confidence_threshold)
        self.dynamic_predictor = DynamicSignPredictor(
            confidence_threshold=float(dyn_cfg.get("confidence_threshold", 0.70)),
            min_motion_energy=float(dyn_cfg.get("movement_threshold", 0.02)),
        )

        self.fusion_engine = StaticDynamicFusionEngine(
            static_predictor=self.static_predictor,
            dynamic_predictor=self.dynamic_predictor,
            sequence_length=int(dyn_cfg.get("sequence_length", 30)),
            inference_interval=int(dyn_cfg.get("inference_interval", 2)),
            dynamic_cooldown_seconds=float(dyn_cfg.get("cooldown_seconds", 1.5)),
            static_threshold=self.confidence_threshold,
            dynamic_threshold=float(dyn_cfg.get("confidence_threshold", 0.70)),
        )

        # Sentence Buffer, Grammar & Speech Modules
        self.token_buffer = TokenBuffer(
            confirmation_frames=int(inf_cfg.get("consecutive_confirm_frames", 6)),
            debounce_seconds=float(inf_cfg.get("debounce_seconds", 0.8)),
        )
        self.grammar_engine = ISLGrammarEngine()
        self.tts_engine = TextToSpeechEngine(
            default_language=self.current_language,
            auto_speak=self.auto_speak,
        )

        self.classes = self.static_predictor.classes
        self.last_grammar_result: Optional[GrammarResult] = None
        self.last_translated_text: str = ""

    def process_frame(
        self,
        frame: np.ndarray,
        detector: HandDetector,
    ) -> Tuple[np.ndarray, RecognitionResult, float, bool]:
        """
        Process single video frame through the multi-modal fusion AI pipeline.

        Returns:
            annotated_frame: np.ndarray
            rec_result: RecognitionResult (type, label, confidence, etc.)
            latency_ms: float
            hand_detected: bool
        """
        t_start = time.perf_counter()

        # 1. MediaPipe Hand Landmark Detection
        annotated_frame, hands = detector.process(frame, draw=True)
        hand_detected = len(hands) > 0

        # 2. Static + Dynamic Fusion Processing
        if hand_detected:
            primary_hand = hands[0]
            rec_result = self.fusion_engine.process_landmarks(primary_hand.landmark_list)
        else:
            rec_result = self.fusion_engine.process_landmarks(None)

        # 3. Token Formation & Sentence Accumulation
        if rec_result.is_stable and rec_result.label != "...":
            committed = self.token_buffer.update(
                token=rec_result.label,
                confidence=rec_result.confidence,
                source=rec_result.type,
            )
            if committed is not None:
                # Update grammar reconstruction on new token
                self.last_grammar_result = self.grammar_engine.process(self.token_buffer.get_tokens())
                if self.current_language == "hi":
                    self.last_translated_text = translate_sentence(self.last_grammar_result.corrected_text, target_language="hi")
                else:
                    self.last_translated_text = self.last_grammar_result.corrected_text

                # Auto-speak if enabled
                if self.auto_speak:
                    self.tts_engine.speak(self.last_translated_text, language=self.current_language)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return annotated_frame, rec_result, latency_ms, hand_detected

    def finalize_and_speak(self) -> None:
        """Finalize the current sentence, reconstruct natural grammar, and speak."""
        tokens = self.token_buffer.get_tokens()
        if not tokens:
            return

        self.last_grammar_result = self.grammar_engine.process(tokens)
        english_text = self.last_grammar_result.corrected_text
        if self.current_language == "hi":
            speak_text = translate_sentence(english_text, target_language="hi")
            self.last_translated_text = speak_text
        else:
            speak_text = english_text
            self.last_translated_text = english_text

        print(f"\n[FINALIZED SENTENCE] {speak_text}")
        self.tts_engine.speak(speak_text, language=self.current_language)

    def draw_hud(
        self,
        frame: np.ndarray,
        result: RecognitionResult,
        fps: float,
        latency_ms: float,
        hand_detected: bool,
    ) -> np.ndarray:
        """
        Renders rich, high-contrast, accessible HUD overlay displaying multimodal diagnostics.
        """
        h, w = frame.shape[:2]

        # Top Header Bar (Translucent Dark Charcoal)
        header_bar = frame.copy()
        cv2.rectangle(header_bar, (0, 0), (w, 105), (18, 22, 28), -1)
        cv2.addWeighted(header_bar, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(
            frame,
            "SIGNBRIDGE AI — Real-Time Multimodal ISL Platform (Week 5)",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Diagnostics: FPS, Latency & Language
        lang_str = f"Lang: {self.current_language.upper()}"
        diag_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms | {lang_str}"
        cv2.putText(
            frame,
            diag_text,
            (w - 380, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

        # Hand Detection & TTS Status
        if hand_detected:
            status_color = (0, 255, 0)
            status_str = "[HAND READY]"
        else:
            status_color = (120, 120, 120)
            status_str = "[NO HAND DETECTED]"

        tts_str = "[TTS ON]" if self.auto_speak else "[TTS MANUAL]"
        cv2.putText(frame, f"{status_str}  {tts_str}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA)

        # Sign Display & Modality
        if hand_detected and result.label != "...":
            display_sign = result.label
            modality_str = f"[{result.type}]"
            sign_color = (0, 255, 0) if result.type == "DYNAMIC" else (0, 230, 255)
            conf_str = f"Conf: {result.confidence:.0%}"
        elif hand_detected:
            display_sign = "IDENTIFYING..."
            modality_str = "[SCANNING]"
            sign_color = (0, 200, 255)
            conf_str = f"Threshold: {self.confidence_threshold:.0%}"
        else:
            display_sign = "WAITING FOR SIGN..."
            modality_str = "[IDLE]"
            sign_color = (150, 150, 150)
            conf_str = "No Signal"

        cv2.putText(frame, f"Sign: {display_sign} {modality_str}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, sign_color, 2, cv2.LINE_AA)
        cv2.putText(frame, conf_str, (w - 220, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom Sentence & Grammar Panel
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 110), (w, h), (14, 18, 24), -1)
        cv2.addWeighted(bot_bar, 0.90, frame, 0.10, 0, frame)

        # Raw Token Stream
        raw_tokens = self.token_buffer.get_text()
        cv2.putText(frame, f"ISL Tokens: {raw_tokens if raw_tokens else '...'}", (15, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        # Corrected Natural Sentence
        if self.last_grammar_result and self.last_grammar_result.corrected_text:
            corrected_display = self.last_grammar_result.corrected_text
        else:
            corrected_display = "..."
        cv2.putText(frame, f"English: \"{corrected_display}\"", (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Hindi Translation (if available)
        if self.last_translated_text and self.current_language == "hi":
            # Note: OpenCV putText has limited UTF-8 devanagari glyph support; show label or romanized status
            cv2.putText(frame, f"Hindi: [Voice Output Active]", (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Controls: [ENTER] Speak | [SPACE] Space | [BACKSPACE] Del | [C] Clear | [T] TTS | [L] Lang | [Q] Quit", (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        return frame

    def run(self) -> None:
        """Launch interactive webcam recognition loop."""
        print("=" * 68)
        print("  SIGNBRIDGE AI — Real-Time Multimodal Recognition & Speech (Week 5)")
        print("  Pipeline: Camera -> MediaPipe -> Static/Dynamic Fusion -> Grammar -> TTS")
        print("=" * 68)
        print("Loading AI models...")

        window_name = "SignBridge AI — Live Multimodal Recognition (Week 5)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            with Camera(camera_index=self.camera_index) as cam, HandDetector() as detector:
                print("Camera initialized successfully. Press [Q] to quit.")
                while True:
                    success, frame = cam.read()
                    if not success:
                        print("[Warning] Could not read frame from camera.")
                        break

                    annotated, result, latency, hand_detected = self.process_frame(frame, detector)

                    hud_frame = self.draw_hud(
                        frame=annotated,
                        result=result,
                        fps=cam.fps,
                        latency_ms=latency,
                        hand_detected=hand_detected,
                    )

                    cv2.imshow(window_name, hud_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("Exiting real-time session...")
                        break
                    elif key in (13, 10):  # ENTER
                        self.finalize_and_speak()
                    elif key == 32:  # SPACE
                        self.token_buffer.add_space()
                    elif key in (8, 127):  # BACKSPACE
                        self.token_buffer.backspace()
                        if self.token_buffer.get_tokens():
                            self.last_grammar_result = self.grammar_engine.process(self.token_buffer.get_tokens())
                        else:
                            self.last_grammar_result = None
                            self.last_translated_text = ""
                    elif key in (ord("c"), ord("C")):
                        self.token_buffer.clear()
                        self.last_grammar_result = None
                        self.last_translated_text = ""
                    elif key in (ord("t"), ord("T")):
                        self.auto_speak = not self.auto_speak
                        print(f"[TTS TOGGLE] Auto-Speak set to: {self.auto_speak}")
                    elif key in (ord("l"), ord("L")):
                        self.current_language = "hi" if self.current_language == "en" else "en"
                        print(f"[LANG TOGGLE] Active Language set to: {self.current_language.upper()}")

        except CameraError as err:
            print(f"\n[CAMERA ERROR] {err}")
            print("Please check camera permissions, connection, or camera index.\n")
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
        finally:
            self.tts_engine.shutdown()
            cv2.destroyAllWindows()
            print("SignBridge AI session terminated cleanly.")

    def run_benchmark(self, num_frames: int = 50) -> Dict[str, float]:
        """
        Headless benchmark verifying pipeline latency, static/dynamic inference time,
        and throughput.
        """
        print(f"\nRunning Week 5 headless multimodal benchmark on {num_frames} frames...")
        latencies = []
        static_latencies = []
        dynamic_latencies = []

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_seq = np.random.randn(30, 63).astype(np.float32)
        dummy_feat = np.random.randn(63).astype(np.float32)

        # Measure static model latency
        for _ in range(num_frames):
            t0 = time.perf_counter()
            _ = self.static_predictor.predict_features(dummy_feat)
            static_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Measure dynamic model latency
        if self.dynamic_predictor.is_ready:
            for _ in range(num_frames):
                t0 = time.perf_counter()
                _ = self.dynamic_predictor.predict_sequence(dummy_seq)
                dynamic_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Measure full camera + detection + fusion pipeline latency
        with HandDetector() as detector:
            for _ in range(num_frames):
                t0 = time.perf_counter()
                _, _, lat, _ = self.process_frame(dummy_frame, detector)
                latencies.append(lat)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        effective_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        avg_static_ms = float(np.mean(static_latencies))
        avg_dynamic_ms = float(np.mean(dynamic_latencies)) if dynamic_latencies else 0.0

        print(f"\nWeek 5 Benchmark Results:")
        print(f"  • Static Model Latency:  {avg_static_ms:.2f} ms")
        print(f"  • Dynamic Model Latency: {avg_dynamic_ms:.2f} ms")
        print(f"  • Total Pipeline Latency:{avg_latency:.2f} ms")
        print(f"  • 95th Percentile:       {p95_latency:.2f} ms")
        print(f"  • Effective Throughput:  {effective_fps:.1f} FPS")

        return {
            "avg_static_ms": avg_static_ms,
            "avg_dynamic_ms": avg_dynamic_ms,
            "avg_pipeline_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "effective_fps": effective_fps,
        }


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Real-Time Recognition (Week 5)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--window", type=int, default=None, help="Smoothing window size")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi"], help="Default speech language (en or hi)")
    parser.add_argument("--auto-speak", action="store_true", help="Enable automatic speech on committed signs")
    parser.add_argument("--benchmark", action="store_true", help="Run latency/FPS benchmark without opening camera")

    args = parser.parse_args()

    engine = ISLRealtimeEngine(
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        smoothing_window=args.window,
        language=args.lang,
        auto_speak=args.auto_speak,
    )

    if args.benchmark:
        engine.run_benchmark()
    else:
        engine.run()


if __name__ == "__main__":
    main()
