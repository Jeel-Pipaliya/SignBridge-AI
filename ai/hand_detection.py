"""
SignBridge AI - Hand Detection Module
Integrates MediaPipe Hands for modular hand detection, landmark extraction, and visualization.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple
import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


class HandData(NamedTuple):
    """Encapsulates extracted landmark information for a single detected hand."""
    hand_index: int
    handedness: str  # "Left" or "Right"
    confidence: float
    raw_landmarks: any  # MediaPipe NormalizedLandmarkList
    landmark_list: List[Tuple[float, float, float]]  # [(x, y, z), ...] 21 points


class HandDetector:
    """
    Reusable MediaPipe Hand Detection & Landmark Tracking Wrapper.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def process(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> Tuple[np.ndarray, List[HandData]]:
        """
        Process a BGR frame, detect hands, optionally draw landmarks, and return structured HandData.

        Args:
            frame: BGR image from OpenCV (H, W, C)
            draw: Whether to draw landmarks on the frame

        Returns:
            annotated_frame: Frame with drawn landmarks (copy if draw=True, original if draw=False)
            detected_hands: List of HandData objects for all detected hands
        """
        output_frame = frame.copy() if draw else frame
        h, w = frame.shape[:2]

        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        detected_hands: List[HandData] = []

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Retrieve handedness (Left/Right) and confidence score
                handedness = "Unknown"
                score = 1.0
                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    handedness = classification.label
                    score = classification.score

                # Extract 21 (x, y, z) coordinates
                coords: List[Tuple[float, float, float]] = []
                for lm in hand_lms.landmark:
                    coords.append((lm.x, lm.y, lm.z))

                detected_hands.append(
                    HandData(
                        hand_index=idx,
                        handedness=handedness,
                        confidence=score,
                        raw_landmarks=hand_lms,
                        landmark_list=coords,
                    )
                )

                if draw:
                    self.mp_drawing.draw_landmarks(
                        output_frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    # Draw handedness label near wrist
                    wrist = hand_lms.landmark[0]
                    px, py = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        output_frame,
                        f"{handedness} ({score:.0%})",
                        (max(px - 30, 10), max(py - 15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        return output_frame, detected_hands

    def get_first_hand_landmarks(
        self,
        frame: np.ndarray,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Convenience method: returns the 21 (x, y, z) landmarks of the primary hand, or None.
        """
        _, hands = self.process(frame, draw=False)
        if hands:
            return hands[0].landmark_list
        return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def run_hand_detection_demo() -> None:
    """Run interactive live camera hand detection demo (Week 1 Day 3)."""
    from realtime.camera import Camera, CameraError

    print("=" * 60)
    print(" SignBridge AI - Hand Detection & Landmark Tracking (Week 1 Day 3)")
    print(" Press 'q' or 'ESC' to exit")
    print("=" * 60)

    try:
        with Camera() as cam, HandDetector() as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    print("Camera frame read failed. Exiting.")
                    break

                # Detect hands & draw landmarks
                annotated_frame, hands = detector.process(frame, draw=True)

                # Draw FPS and Status
                cam.draw_fps(annotated_frame)
                hand_count = len(hands)
                status_color = (0, 255, 0) if hand_count > 0 else (0, 0, 255)
                status_text = f"Hands Detected: {hand_count}"

                cv2.putText(
                    annotated_frame,
                    status_text,
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("SignBridge AI - Day 3 Hand Detection", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (config.KEY_EXIT, 27):
                    break

    except CameraError as err:
        print(f"\n[CAMERA ERROR] {err}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_detection_demo()
