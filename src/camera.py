from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2


@dataclass(slots=True)
class CameraConfig:
    source: int | str = 0
    width: int | None = None
    height: int | None = None
    fps: int | None = None


class CameraStream:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.capture = cv2.VideoCapture(self.config.source)

        if self.config.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)

    def is_open(self) -> bool:
        return self.capture.isOpened()

    def read(self) -> tuple[bool, Any]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> CameraStream:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        self.release()
