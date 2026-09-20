"""
SignBridge AI - Guided Data Collection Prototype (Week 1 Day 6)
Captures live hand landmark features from the webcam and appends them
to a standardized CSV file for machine learning training.

Usage:
    python data/collect_samples.py
    python data/collect_samples.py --label HELLO --samples 50

Controls:
    [SPACE] or [S] : Capture single landmark sample
    [B]            : Burst capture mode (continuous capture with short delay)
    [N]            : Change current sign label
    [Q] or [ESC]   : Exit and save session
"""

import sys
import os
import time
import argparse
import csv
from pathlib import Path
from typing import Optional
import cv2
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from ai.feature_extraction import extract_features, get_feature_names
from realtime.camera import Camera, CameraError


class DataCollector:
    """
    Manages live data collection sessions, saving normalized 63D features to CSV.
    """

    def __init__(
        self,
        output_csv: Path = config.RAW_COLLECTED_CSV,
        initial_label: str = "SAMPLE",
    ):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.current_label = initial_label.strip().upper()
        self.session_count = 0
        self.burst_mode = False
        self.last_burst_time = 0.0
        self.burst_delay = 0.15  # seconds between burst captures

        # Initialize CSV header if file doesn't exist
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Create the CSV with standard 63-feature headers + label if not already present."""
        if not self.output_csv.exists() or self.output_csv.stat().st_size == 0:
            headers = get_feature_names() + ["label"]
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"[Info] Created new dataset file at: {self.output_csv}")
        else:
            print(f"[Info] Appending to existing dataset file at: {self.output_csv}")

    def get_class_sample_count(self, label: str) -> int:
        """Count how many samples exist for the specified label in the CSV."""
        if not self.output_csv.exists():
            return 0
        try:
            df = pd.read_csv(self.output_csv)
            if "label" in df.columns:
                return int((df["label"] == label).sum())
        except Exception:
            return 0
        return 0

    def save_sample(self, features: list) -> bool:
        """Append a single 63-feature vector + current label to the CSV."""
        if features is None or len(features) != config.NUM_FEATURES:
            return False

        row = list(features) + [self.current_label]
        with open(self.output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.session_count += 1
        return True

    def set_label(self, new_label: str) -> None:
        """Update current active label."""
        cleaned = new_label.strip().upper()
        if cleaned:
            self.current_label = cleaned
            print(f"[Info] Active label changed to: '{self.current_label}'")


def run_collector(initial_label: str = "HELLO", target_samples: Optional[int] = None) -> None:
    """Run the interactive data collection interface."""
    collector = DataCollector(initial_label=initial_label)

    print("=" * 65)
    print("  SignBridge AI - Guided Data Collection Prototype (Week 1 Day 6)")
    print(f"  Target File:  {collector.output_csv}")
    print(f"  Active Label: {collector.current_label}")
    print("=" * 65)
    print("  [SPACE] / [S] : Capture single sample")
    print("  [B]           : Toggle burst capture mode")
    print("  [N]           : Enter new label")
    print("  [Q] / [ESC]   : Exit")
    print("=" * 65)

    try:
        with Camera() as cam, HandDetector() as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    print("[Error] Failed to read from camera.")
                    break

                # Process hand detection
                annotated, hands = detector.process(frame, draw=True)
                cam.draw_fps(annotated)

                hand_present = len(hands) > 0
                features = None
                if hand_present:
                    primary_hand = hands[0]
                    features = extract_features(primary_hand.landmark_list)

                # Burst capture logic
                current_time = time.time()
                captured_this_frame = False
                if collector.burst_mode and hand_present and features is not None:
                    if current_time - collector.last_burst_time >= collector.burst_delay:
                        if collector.save_sample(features):
                            collector.last_burst_time = current_time
                            captured_this_frame = True

                # Check target limit
                total_for_label = collector.get_class_sample_count(collector.current_label)
                if target_samples and total_for_label >= target_samples:
                    collector.burst_mode = False

                # ── Draw HUD ───────────────────────────────────────────────
                h, w = annotated.shape[:2]
                hud_bg = annotated.copy()
                cv2.rectangle(hud_bg, (0, 0), (w, 110), (20, 20, 20), -1)
                cv2.addWeighted(hud_bg, 0.75, annotated, 0.25, 0, annotated)

                # Active label and counts
                cv2.putText(
                    annotated,
                    f"LABEL: {collector.current_label}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                count_str = f"Samples: {total_for_label}"
                if target_samples:
                    count_str += f" / {target_samples}"
                cv2.putText(
                    annotated,
                    count_str,
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Hand status badge
                status_color = (0, 255, 0) if hand_present else (0, 0, 255)
                status_text = "HAND READY" if hand_present else "NO HAND DETECTED"
                cv2.putText(
                    annotated,
                    status_text,
                    (w - 240, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                # Burst indicator
                if collector.burst_mode:
                    cv2.putText(
                        annotated,
                        "BURST RECORDING...",
                        (w - 240, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Flash indicator if captured
                if captured_this_frame:
                    cv2.rectangle(annotated, (0, 0), (w, h), (0, 255, 0), 4)

                # Controls legend at bottom
                cv2.rectangle(annotated, (0, h - 30), (w, h), (10, 10, 10), -1)
                cv2.putText(
                    annotated,
                    "[SPACE/S] Capture | [B] Burst Mode | [N] New Label | [Q] Quit",
                    (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("SignBridge AI - Data Collector", annotated)

                key = cv2.waitKey(1) & 0xFF

                if key in (config.KEY_EXIT, 27):  # 'q' or ESC
                    break
                elif key in (config.KEY_SAVE_SAMPLE, config.KEY_SPACE):  # 's' or SPACE
                    if hand_present and features is not None:
                        if collector.save_sample(features):
                            print(f"Captured 1 sample for '{collector.current_label}'. Total: {total_for_label + 1}")
                    else:
                        print("[Warning] Cannot capture: No hand detected.")
                elif key == ord('b'):
                    collector.burst_mode = not collector.burst_mode
                    print(f"[Info] Burst mode: {'ENABLED' if collector.burst_mode else 'DISABLED'}")
                elif key == ord('n'):
                    # Prompt in console for new label
                    collector.burst_mode = False
                    cv2.destroyAllWindows()
                    try:
                        new_lbl = input("\nEnter new ISL Sign label (e.g., HELLO, YES, NO, 1): ")
                        collector.set_label(new_lbl)
                    except EOFError:
                        pass

    except CameraError as err:
        print(f"\n[CAMERA ERROR] {err}")
    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 65)
        print(f" Session Finished. Collected {collector.session_count} total samples.")
        print(f" Dataset saved to: {collector.output_csv}")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI Data Collector")
    parser.add_argument("--label", type=str, default="HELLO", help="Initial sign label")
    parser.add_argument("--samples", type=int, default=None, help="Target number of samples")
    args = parser.parse_args()

    run_collector(initial_label=args.label, target_samples=args.samples)
