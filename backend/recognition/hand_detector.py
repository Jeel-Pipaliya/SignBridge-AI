"""
DAY 9 -- SignBridge AI  Week 2
Hand Detector: wraps MediaPipe Hands into a clean, reusable class.

Run standalone:
    python backend/recognition/hand_detector.py

Controls: Q to quit.
"""

import cv2
import mediapipe as mp
import time


class HandDetector:
    """
    Detects hand landmarks via MediaPipe Hands.

    Args:
        max_num_hands:         Maximum number of hands to detect (1 or 2).
        detection_confidence:  Min confidence to consider a detection valid.
        tracking_confidence:   Min confidence to continue tracking.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ---- Public API ----------------------------------------------------------

    def detect(self, bgr_frame):
        """
        Run MediaPipe Hands on `bgr_frame` (OpenCV BGR image).

        Returns the raw MediaPipe results object.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        return results

    def draw_landmarks(self, frame, results):
        """
        Draw hand connections and landmark dots on `frame` (in-place).

        Returns the annotated frame.
        """
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return frame

    def hand_count(self, results) -> int:
        """Return how many hands were detected."""
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


# ---- Standalone demo ---------------------------------------------------------

def main() -> None:
    detector = HandDetector(max_num_hands=2)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Hand Detector running. Press Q to quit.")

    prev_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = detector.detect(frame)
        frame   = detector.draw_landmarks(frame, results)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now

        h, w = frame.shape[:2]
        count = detector.hand_count(results)

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands: {count}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "[Q] Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("SignBridge AI -- Day 9: Hand Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
