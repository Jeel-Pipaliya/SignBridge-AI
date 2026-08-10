from __future__ import annotations

from dataclasses import dataclass

import cv2

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    mp = None


@dataclass(slots=True)
class DetectionResult:
    hands: object | None = None
    face: object | None = None
    pose: object | None = None


class MediaPipeHolisticDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        if mp is None:
            raise ImportError("mediapipe is not installed. Install dependencies from requirements.txt first.")

        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return result

    def draw(self, frame_bgr, result) -> None:
        if result.pose_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
            )

        if result.face_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                result.face_landmarks,
                self.mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.styles.get_default_face_mesh_tesselation_style(),
            )

        if result.left_hand_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                result.left_hand_landmarks,
                self.mp.solutions.hands.HAND_CONNECTIONS,
            )

        if result.right_hand_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                result.right_hand_landmarks,
                self.mp.solutions.hands.HAND_CONNECTIONS,
            )

