from __future__ import annotations

import cv2

from src.data_collector import CollectionSample, SignDataCollector
from src.mediapipe_detector import MediaPipeHolisticDetector
from src.vision import draw_text, draw_timestamp, flip_frame


def run_sign_data_collector(label: str = "hello", source: int | str = 0) -> None:
    collector = SignDataCollector()
    samples: list[CollectionSample] = []
    recording = False

    with collector.build_camera(source=source) as camera:
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

            if recording:
                samples.append(collector.collect_result(result, label=label))

            draw_text(frame, f"Label: {label}")
            draw_text(frame, f"Recording: {'YES' if recording else 'NO'}", origin=(20, 70))
            draw_text(frame, f"Frames: {len(samples)}", origin=(20, 100))
            draw_timestamp(frame)

            cv2.imshow("SignBridge Data Collector", frame)
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
    run_sign_data_collector()
