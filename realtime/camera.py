"""
SignBridge AI - Realtime Camera Module
Provides a robust, modular OpenCV webcam capture interface with real-time FPS calculation.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Custom exception raised when camera fails to initialize or capture."""
    pass


class Camera:
    """
    Modular OpenCV webcam wrapper supporting:
    - Custom camera indices and resolutions
    - Graceful camera failure handling
    - Real-time rolling FPS calculation
    - Context manager protocol (with Camera() as cam:)
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        fps_smoothing: float = 0.9,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps_smoothing = fps_smoothing

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

        # FPS Tracking
        self._prev_time = 0.0
        self._fps = 0.0

    def start(self) -> "Camera":
        """Initialize and open the video capture device with multi-backend fallback."""
        logger.info(f"Opening camera at index {self.camera_index}...")

        backends = []
        if sys.platform.startswith("win"):
            # Try CAP_ANY (uses preferred system backend like MSMF), then DSHOW, then MSMF
            backends = [
                ("ANY", cv2.CAP_ANY),
                ("DSHOW", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
            ]
        else:
            backends = [("DEFAULT", cv2.CAP_ANY)]

        opened_cap = None
        for b_name, backend in backends:
            try:
                test_cap = cv2.VideoCapture(self.camera_index, backend)
                if test_cap.isOpened():
                    # Set requested resolution
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                    # Perform a test read to ensure camera actually streams frames
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        logger.info(f"Connected to camera {self.camera_index} via backend {b_name}.")
                        opened_cap = test_cap
                        if test_frame.mean() < 2.0:
                            logger.warning(
                                "Camera frame is nearly pitch black! "
                                "If your laptop has a physical privacy slider or camera shutter, make sure it is OPEN."
                            )
                        break
                    else:
                        test_cap.release()
            except Exception as e:
                logger.debug(f"Backend {b_name} failed: {e}")

        if opened_cap is None:
            # Check if another camera index is available to give helpful guidance
            working_indices = []
            for test_idx in range(4):
                if test_idx == self.camera_index:
                    continue
                try:
                    c = cv2.VideoCapture(test_idx)
                    if c.isOpened():
                        r, f = c.read()
                        if r and f is not None:
                            working_indices.append(test_idx)
                        c.release()
                except Exception:
                    pass

            alt_msg = ""
            if working_indices:
                alt_msg = f" Working camera found at index: {working_indices}. Update config.CAMERA_INDEX."
            raise CameraError(
                f"Unable to access camera at index {self.camera_index}. "
                f"Please ensure no other application (e.g. background demo, Teams, Zoom) is using your webcam.{alt_msg}"
            )

        self.cap = opened_cap
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened successfully. Resolution: {actual_w}x{actual_h}")

        self.is_running = True
        self._prev_time = time.time()
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the camera.
        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        if not self.is_running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera.")
            return False, None

        # Resize if hardware resolution differs from configured target
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Horizontal mirror for intuitive sign interaction
        frame = cv2.flip(frame, 1)

        # Update FPS
        curr_time = time.time()
        delta = curr_time - self._prev_time
        self._prev_time = curr_time

        if delta > 0:
            instant_fps = 1.0 / delta
            if self._fps == 0.0:
                self._fps = instant_fps
            else:
                self._fps = (self.fps_smoothing * self._fps) + ((1.0 - self.fps_smoothing) * instant_fps)

        return True, frame

    @property
    def fps(self) -> float:
        """Return the current smoothed FPS."""
        return self._fps

    def draw_fps(
        self,
        frame: np.ndarray,
        position: Tuple[int, int] = (15, 30),
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw FPS counter on the given frame."""
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        """Release the camera hardware resources cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Camera resources released.")

    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def run_camera_demo() -> None:
    """Run an interactive standalone camera preview with FPS display."""
    print("=" * 60)
    print(" SignBridge AI - Realtime Camera System (Week 1 Day 2)")
    print(" Press 'q' or 'ESC' to exit")
    print("=" * 60)

    try:
        with Camera() as cam:
            while True:
                success, frame = cam.read()
                if not success:
                    print("Error: Could not read frame from camera. Exiting.")
                    break

                # Draw UI overlays
                cam.draw_fps(frame)
                cv2.putText(
                    frame,
                    "SignBridge AI - Camera Active | Press 'q' to Quit",
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("SignBridge AI - Camera Demo", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (config.KEY_EXIT, 27):  # 'q' or ESC
                    break

    except CameraError as err:
        print(f"\n[CAMERA ERROR] {err}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_demo()
