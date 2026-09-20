"""
SignBridge AI - Landmark Understanding & Visualization Module (Week 1 Day 4)
Visualizes all 21 MediaPipe hand landmarks with anatomical index labels,
coordinates (x, y, z), and color-coded finger groups.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ai.hand_detection import HandDetector
from realtime.camera import Camera, CameraError

# ─── MediaPipe 21 Hand Landmarks Anatomy ─────────────────────────────────────
LANDMARK_NAMES: Dict[int, str] = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

# Color-coding by anatomical finger group (BGR format)
FINGER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WRIST": (0, 255, 255),    # Yellow
    "THUMB": (0, 0, 255),      # Red
    "INDEX": (0, 255, 0),      # Green
    "MIDDLE": (255, 255, 0),   # Cyan
    "RING": (255, 0, 0),       # Blue
    "PINKY": (255, 0, 255),    # Magenta
}


def get_landmark_group(idx: int) -> str:
    """Return the anatomical finger group for a given landmark index (0-20)."""
    if idx == 0:
        return "WRIST"
    elif 1 <= idx <= 4:
        return "THUMB"
    elif 5 <= idx <= 8:
        return "INDEX"
    elif 9 <= idx <= 12:
        return "MIDDLE"
    elif 13 <= idx <= 16:
        return "RING"
    elif 17 <= idx <= 20:
        return "PINKY"
    return "UNKNOWN"


def draw_indexed_landmarks(
    frame: np.ndarray,
    landmarks: list,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw annotated landmarks with indices (0-20) and finger-group colors.

    Args:
        frame: BGR image
        landmarks: List of 21 (x, y, z) tuples (normalized coordinates in [0, 1])
        show_labels: Whether to print numeric index near each joint
    """
    h, w = frame.shape[:2]
    points = []

    for idx, (x, y, z) in enumerate(landmarks):
        px, py = int(x * w), int(y * h)
        points.append((px, py))
        group = get_landmark_group(idx)
        color = FINGER_COLORS.get(group, (255, 255, 255))

        # Draw circle on landmark
        radius = 7 if idx in (0, 4, 8, 12, 16, 20) else 5
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), 1)

        if show_labels:
            # Place index label slightly offset
            offset_x = 8 if px < w - 30 else -20
            offset_y = -6 if py > 20 else 15
            cv2.putText(
                frame,
                str(idx),
                (px + offset_x, py + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Draw custom connections between landmarks
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm base
        (5, 9), (9, 13), (13, 17),
    ]

    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            p1 = points[start_idx]
            p2 = points[end_idx]
            group = get_landmark_group(end_idx)
            color = FINGER_COLORS.get(group, (200, 200, 200))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw anatomical color legend on top-left of the display."""
    legend_items = [
        ("Wrist (0)", FINGER_COLORS["WRIST"]),
        ("Thumb (1-4)", FINGER_COLORS["THUMB"]),
        ("Index (5-8)", FINGER_COLORS["INDEX"]),
        ("Middle (9-12)", FINGER_COLORS["MIDDLE"]),
        ("Ring (13-16)", FINGER_COLORS["RING"]),
        ("Pinky (17-20)", FINGER_COLORS["PINKY"]),
    ]

    x, y_start = 15, 80
    for i, (label, color) in enumerate(legend_items):
        y = y_start + (i * 22)
        cv2.circle(frame, (x + 6, y - 5), 6, color, -1)
        cv2.putText(
            frame,
            label,
            (x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def run_landmark_visualizer() -> None:
    """Interactive visualizer showing indexed landmarks and coordinate inspector."""
    print("=" * 60)
    print(" SignBridge AI - 21 Landmark Understanding & Inspector (Week 1 Day 4)")
    print(" Press 'q' or 'ESC' to exit")
    print("=" * 60)

    window_name = "SignBridge AI - Landmark Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        with Camera() as cam, HandDetector() as detector:
            while True:
                success, frame = cam.read()
                if not success:
                    break

                # Get raw frame and detections without default drawing
                _, hands = detector.process(frame, draw=False)

                annotated = frame.copy()
                cam.draw_fps(annotated)
                draw_legend(annotated)

                if hands:
                    primary_hand = hands[0]
                    # Draw our custom indexed landmarks
                    draw_indexed_landmarks(annotated, primary_hand.landmark_list, show_labels=True)

                    # Show wrist coordinate info
                    wrist = primary_hand.landmark_list[0]
                    cv2.putText(
                        annotated,
                        f"{primary_hand.handedness} Hand | Wrist: x={wrist[0]:.2f}, y={wrist[1]:.2f}, z={wrist[2]:.2f}",
                        (15, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        annotated,
                        "Hold your hand in front of the camera",
                        (15, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("SignBridge AI - Landmark Inspector", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (config.KEY_EXIT, 27):
                    break

    except CameraError as err:
        print(f"\n[CAMERA ERROR] {err}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_landmark_visualizer()
