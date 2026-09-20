"""
DAY 2 — SignBridge AI
OpenCV webcam capture with FPS display and image saving.

Run:
    python ai/inference/webcam.py

Controls:
    Q — Quit
    S — Save current frame as image
"""

import cv2
import time
import os


# ─── Constants ────────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join("ai", "datasets", "collected")
IMAGE_PREFIX = "image"


# ─── Helpers ────────────────────────────────────────────────────────

def get_next_image_index(save_dir: str) -> int:
    """Return the next available image index in save_dir."""
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith(IMAGE_PREFIX) and f.endswith(".jpg")
    ]
    return len(existing) + 1


def draw_ui(frame, fps: float, saved_count: int) -> None:
    """Overlay FPS counter and key-hint bar on the frame (in-place)."""
    h, w = frame.shape[:2]

    # FPS overlay
    cv2.putText(
        frame, f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
    )

    # How many images saved
    if saved_count > 0:
        cv2.putText(
            frame, f"Saved: {saved_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )

    # Bottom controls bar
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, "[S] Save Frame    [Q] Quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam opened. Press S to save a frame, Q to quit.")

    previous_time: float = 0.0
    fps: float = 0.0
    saved_count: int = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)          # mirror effect

        # ── FPS ──────────────────────────────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else fps
        previous_time = current_time

        draw_ui(frame, fps, saved_count)
        cv2.imshow("SignBridge AI — Day 2: Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit.")
            break

        if key == ord("s"):
            idx = get_next_image_index(SAVE_DIR)
            filename = os.path.join(SAVE_DIR, f"{IMAGE_PREFIX}_{idx:03d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
