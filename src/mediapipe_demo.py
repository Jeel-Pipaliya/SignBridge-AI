from __future__ import annotations

import cv2

from src.camera import CameraConfig, CameraStream
from src.mediapipe_detector import MediaPipeHolisticDetector
from src.vision import draw_text, draw_timestamp, flip_frame


def run_mediapipe_demo(source: int | str = 0) -> None:
    with CameraStream(CameraConfig(source=source)) as camera:
        if not camera.is_open():
            raise RuntimeError("Unable to open the camera source.")

        detector = MediaPipeHolisticDetector()

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = flip_frame(frame)
            result = detector.process(frame)
            detector.draw(frame, result)

            hands_visible = bool(result.left_hand_landmarks or result.right_hand_landmarks)
            face_visible = bool(result.face_landmarks)
            pose_visible = bool(result.pose_landmarks)

            draw_text(frame, f"Hands: {'YES' if hands_visible else 'NO'}")
            draw_text(frame, f"Face: {'YES' if face_visible else 'NO'}", origin=(20, 70))
            draw_text(frame, f"Pose: {'YES' if pose_visible else 'NO'}", origin=(20, 100))
            draw_timestamp(frame)

            cv2.imshow("SignBridge MediaPipe Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_mediapipe_demo()
