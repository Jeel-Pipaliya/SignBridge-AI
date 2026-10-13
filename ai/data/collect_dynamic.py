"""
SignBridge AI - Dynamic Gesture Sequence Data Collection CLI (Week 5)
Interactive recording tool for capturing temporal gesture sequences (HELLO, J, Z).

Controls:
  [R] - Start recording sequence
  [S] - Stop and save recording sequence
  [N] - Switch to next gesture class
  [C] - Cancel current in-progress recording
  [Q] - Save and exit

Usage:
  python -m ai.data_collection.collect_dynamic
  python -m ai.data_collection.collect_dynamic --class HELLO --target 50
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

# Safe console encoding for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from ai.preprocessing.normalization import normalize_landmarks
from ai.utils.logger import setup_logger

logger = setup_logger("DynamicCollector")

CLASSES = ["HELLO", "J", "Z"]


class DynamicDataCollector:
    """
    Interactive collector for multi-frame dynamic gesture sequences.
    """

    def __init__(
        self,
        camera_index: int = 0,
        initial_class: str = "HELLO",
        target_sequences: int = 50,
        output_dir: Optional[Path] = None,
        min_frames: int = 15,
        max_frames: int = 60,
    ):
        self.camera_index = camera_index
        self.classes = CLASSES
        self.current_class_idx = self.classes.index(initial_class) if initial_class in self.classes else 0
        self.target_sequences = target_sequences
        self.output_dir = Path(output_dir) if output_dir is not None else config.DYNAMIC_SEQUENCES_DIR
        self.raw_dir = config.DYNAMIC_RAW_DIR
        self.min_frames = min_frames
        self.max_frames = max_frames

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for cls in self.classes:
            (self.raw_dir / cls).mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self.current_sequence_frames: List[np.ndarray] = []
        self.current_timestamps: List[float] = []
        self.sequence_counts = {cls: self._count_existing_sequences(cls) for cls in self.classes}

    def _count_existing_sequences(self, class_name: str) -> int:
        """Count existing recorded sequence files for a class."""
        class_folder = self.raw_dir / class_name
        if not class_folder.exists():
            return 0
        return len(list(class_folder.glob("*.npz")))

    @property
    def current_class(self) -> str:
        return self.classes[self.current_class_idx]

    def next_class(self) -> None:
        """Switch to the next dynamic sign."""
        if self.is_recording:
            self.cancel_recording()
        self.current_class_idx = (self.current_class_idx + 1) % len(self.classes)
        print(f"\n[CLASS CHANGED] Now collecting for: {self.current_class}")

    def start_recording(self) -> None:
        """Begin accumulating frames."""
        self.is_recording = True
        self.current_sequence_frames = []
        self.current_timestamps = []
        print(f"\n[*] Recording sequence for class '{self.current_class}' started...")

    def stop_recording(self) -> bool:
        """Stop and persist sequence if it meets minimum frame length."""
        if not self.is_recording:
            return False

        self.is_recording = False
        frame_count = len(self.current_sequence_frames)

        if frame_count < self.min_frames:
            print(f"[Warning] Sequence discarded: only {frame_count} frames (minimum is {self.min_frames}).")
            self.current_sequence_frames = []
            self.current_timestamps = []
            return False

        # Save sequence as NPZ
        seq_idx = self.sequence_counts[self.current_class] + 1
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        sequence_id = f"{self.current_class}_{seq_idx:04d}_{timestamp_str}"

        # Raw file path
        raw_path = self.raw_dir / self.current_class / f"{sequence_id}.npz"
        # Sequences folder path
        seq_path = self.output_dir / f"{sequence_id}.npz"

        seq_data = np.array(self.current_sequence_frames, dtype=np.float32)
        ts_data = np.array(self.current_timestamps, dtype=np.float64)

        payload = {
            "sequence_id": sequence_id,
            "label": self.current_class,
            "sequence": seq_data,
            "timestamps": ts_data,
            "frame_count": frame_count,
        }

        np.savez_compressed(raw_path, **payload)
        np.savez_compressed(seq_path, **payload)

        self.sequence_counts[self.current_class] += 1
        print(f"[SAVED] {sequence_id} ({frame_count} frames) -> {self.sequence_counts[self.current_class]}/{self.target_sequences}")

        self.current_sequence_frames = []
        self.current_timestamps = []
        return True

    def cancel_recording(self) -> None:
        """Discard current frames without saving."""
        if self.is_recording:
            print("[CANCELLED] Discarded current recording.")
            self.is_recording = False
            self.current_sequence_frames = []
            self.current_timestamps = []

    def draw_hud(
        self,
        frame: np.ndarray,
        hand_detected: bool,
        fps: float,
    ) -> np.ndarray:
        """Render HUD overlay with instructions and status."""
        h, w = frame.shape[:2]

        # Top Bar (Dark Charcoal)
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 100), (20, 24, 30), -1)
        cv2.addWeighted(top_bar, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(
            frame,
            "SignBridge AI — Dynamic Gesture Collector",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Class and progress
        cls = self.current_class
        count = self.sequence_counts[cls]
        cls_info = f"Class: {cls} [{count}/{self.target_sequences}]"
        cv2.putText(
            frame,
            cls_info,
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Recording state badge
        if self.is_recording:
            rec_str = f"● RECORDING ({len(self.current_sequence_frames)} frames)"
            rec_color = (0, 0, 255)
        else:
            rec_str = "STANDBY"
            rec_color = (180, 180, 180)

        cv2.putText(
            frame,
            rec_str,
            (w - 360, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            rec_color,
            2,
            cv2.LINE_AA,
        )

        # Hand Detection Status
        hand_str = "[HAND READY]" if hand_detected else "[NO HAND]"
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(
            frame,
            hand_str,
            (w - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            hand_color,
            2,
            cv2.LINE_AA,
        )

        # Bottom Controls Hint Bar
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 45), (w, h), (15, 20, 25), -1)
        cv2.addWeighted(bot_bar, 0.85, frame, 0.15, 0, frame)

        controls = "[R] Record  |  [S] Stop & Save  |  [N] Next Class  |  [C] Cancel  |  [Q] Quit"
        cv2.putText(
            frame,
            controls,
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return frame

    def run(self) -> None:
        """Main collection loop."""
        print("=" * 68)
        print("  SIGNBRIDGE AI — Dynamic Gesture Sequence Collector (Week 5)")
        print(f"  Target Classes: {', '.join(self.classes)}")
        print(f"  Saving to: {self.output_dir}")
        print("=" * 68)

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera device at index {self.camera_index}.")
            return

        detector = HandDetector(static_image_mode=False, max_num_hands=1)
        window_name = "SignBridge AI — Dynamic Data Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        prev_time = time.perf_counter()
        fps = 30.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Warning] Failed to read frame from camera.")
                    break

                curr_time = time.perf_counter()
                dt = curr_time - prev_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                prev_time = curr_time

                annotated_frame, hands = detector.process(frame, draw=True)
                hand_detected = len(hands) > 0

                if self.is_recording and hand_detected:
                    primary_hand = hands[0]
                    # Extract raw coordinates and normalized features
                    features = normalize_landmarks(primary_hand.landmark_list)
                    self.current_sequence_frames.append(features)
                    self.current_timestamps.append(time.time())

                    # Auto stop if reached max frames
                    if len(self.current_sequence_frames) >= self.max_frames:
                        print(f"Max frame limit reached ({self.max_frames}). Auto-saving...")
                        self.stop_recording()

                hud_frame = self.draw_hud(annotated_frame, hand_detected, fps)
                cv2.imshow(window_name, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("r"), ord("R")):
                    if not self.is_recording:
                        self.start_recording()
                elif key in (ord("s"), ord("S")):
                    if self.is_recording:
                        self.stop_recording()
                elif key in (ord("n"), ord("N")):
                    self.next_class()
                elif key in (ord("c"), ord("C")):
                    self.cancel_recording()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nDynamic sequence data collection session ended.")
            for cls in self.classes:
                print(f"  • Class {cls}: {self.sequence_counts[cls]} sequences recorded.")


def main():
    parser = argparse.ArgumentParser(description="SignBridge AI Dynamic Gesture Collector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--class", dest="cls", type=str, default="HELLO", help="Initial class (HELLO, J, Z)")
    parser.add_argument("--target", type=int, default=50, help="Target sequences per class")
    parser.add_argument("--outdir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    collector = DynamicDataCollector(
        camera_index=args.camera,
        initial_class=args.cls,
        target_sequences=args.target,
        output_dir=Path(args.outdir) if args.outdir else None,
    )
    collector.run()


if __name__ == "__main__":
    main()
