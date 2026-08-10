from __future__ import annotations

from datetime import datetime

import cv2


def resize_frame(frame, width: int | None = None, height: int | None = None):
    if width is None and height is None:
        return frame

    original_height, original_width = frame.shape[:2]
    if width is None:
        scale = height / original_height
        width = int(original_width * scale)
    elif height is None:
        scale = width / original_width
        height = int(original_height * scale)

    return cv2.resize(frame, (width, height))


def crop_frame(frame, x: int, y: int, width: int, height: int):
    return frame[y : y + height, x : x + width]


def flip_frame(frame, horizontal: bool = True):
    flip_code = 1 if horizontal else 0
    return cv2.flip(frame, flip_code)


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def blur_frame(frame, kernel_size: int = 5):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def threshold_frame(frame, threshold_value: int = 127, max_value: int = 255):
    gray = frame if len(frame.shape) == 2 else to_gray(frame)
    _, thresholded = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    return thresholded


def canny_edges(frame, lower_threshold: int = 50, upper_threshold: int = 150):
    gray = frame if len(frame.shape) == 2 else to_gray(frame)
    return cv2.Canny(gray, lower_threshold, upper_threshold)


def draw_text(frame, text: str, origin: tuple[int, int] = (20, 40), color: tuple[int, int, int] = (0, 255, 0)):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame


def draw_timestamp(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return draw_text(frame, timestamp, origin=(20, 70), color=(255, 255, 255))

