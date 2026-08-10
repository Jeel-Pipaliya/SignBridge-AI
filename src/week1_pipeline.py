from __future__ import annotations

import time

import cv2

from src.data_collector import CollectionSample, SignDataCollector
from src.mediapipe_detector import MediaPipeHolisticDetector
from src.vision import draw_text, draw_timestamp, flip_frame


def run_week1_pipeline(label: str = "hello", source: int | str = 0) -> None:
    collector = SignDataCollector()
    samples: list[CollectionSample] = []
    recording = False
    last_frame_time = time.time()

    with collector.build_camera(source=source) as camera:
        if not camera.is_open():
            raise RuntimeError("Unable to open the camera source.")

        detector = MediaPipeHolisticDetector()

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            current_time = time.time()
            elapsed = max(current_time - last_frame_time, 1e-6)
            fps = 1.0 / elapsed
            last_frame_time = current_time

            frame = flip_frame(frame)
            result = detector.process(frame)
            detector.draw(frame, result)

            hands_visible = bool(result.left_hand_landmarks or result.right_hand_landmarks)
            face_visible = bool(result.face_landmarks)
            pose_visible = bool(result.pose_landmarks)

            if recording:
                samples.append(collector.collect_result(result, label=label))

            draw_text(frame, f"FPS: {fps:.1f}")
            draw_text(frame, f"Recording: {'YES' if recording else 'NO'}", origin=(20, 70))
            draw_text(frame, f"Frames: {len(samples)}", origin=(20, 100))
            draw_text(frame, f"Hands: {'YES' if hands_visible else 'NO'}", origin=(20, 130))
            draw_text(frame, f"Face: {'YES' if face_visible else 'NO'}", origin=(20, 160))
            draw_text(frame, f"Pose: {'YES' if pose_visible else 'NO'}", origin=(20, 190))
            draw_timestamp(frame)

            cv2.imshow("SignBridge Week 1 Pipeline", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                recording = not recording
                if recording:
                    samples = []
            if key == ord("s") and samples:
                collector.save_labeled_sequence(label, samples)
                samples = []
                recording = False

        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_week1_pipeline()
