from __future__ import annotations

from pathlib import Path
import time

import cv2

from src.camera import CameraConfig, CameraStream
from src.vision import blur_frame, canny_edges, draw_text, draw_timestamp, flip_frame, to_gray, to_hsv


def run_smart_camera(source: int | str = 0) -> None:
    with CameraStream(CameraConfig(source=source)) as camera:
        if not camera.is_open():
            raise RuntimeError("Unable to open the camera source.")

        mode = "color"
        last_frame_time = time.time()
        screenshots_dir = Path("data/raw/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            current_time = time.time()
            elapsed = max(current_time - last_frame_time, 1e-6)
            fps = 1.0 / elapsed
            last_frame_time = current_time

            display_frame = flip_frame(frame)

            if mode == "gray":
                display_frame = cv2.cvtColor(to_gray(display_frame), cv2.COLOR_GRAY2BGR)
            elif mode == "hsv":
                display_frame = cv2.cvtColor(to_hsv(display_frame), cv2.COLOR_HSV2BGR)
            elif mode == "blur":
                display_frame = blur_frame(display_frame)
            elif mode == "edges":
                display_frame = cv2.cvtColor(canny_edges(display_frame), cv2.COLOR_GRAY2BGR)

            draw_text(display_frame, f"Mode: {mode}")
            draw_text(display_frame, f"FPS: {fps:.1f}", origin=(20, 70))
            draw_timestamp(display_frame)

            cv2.imshow("SignBridge Smart Camera", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("m"):
                display_frame = flip_frame(display_frame)
            elif key == ord("g"):
                mode = "gray"
            elif key == ord("h"):
                mode = "hsv"
            elif key == ord("b"):
                mode = "blur"
            elif key == ord("e"):
                mode = "edges"
            elif key == ord("c"):
                mode = "color"
            elif key == ord("s"):
                screenshot_path = screenshots_dir / f"frame_{int(current_time)}.png"
                cv2.imwrite(str(screenshot_path), display_frame)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_smart_camera()
