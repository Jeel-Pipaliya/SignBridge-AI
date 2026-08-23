"""
DAY 7 — SignBridge AI
Week 1 Integration: Complete camera → landmark → sequence pipeline.

Run:
    python ai/inference/signbridge_camera.py

Controls:
    S — Save a snapshot image
    R — Toggle sequence recording (records 30 frames then saves .npy)
    Q — Quit
"""

import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai.preprocessing.landmark_extractor import extract_landmarks, TOTAL_FEATURES
from ai.preprocessing.normalize import normalize_landmarks
from ai.preprocessing.sequence_builder import SequenceBuilder


# ─── Configuration ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH: int = 30
SNAPSHOT_DIR: str    = os.path.join("ai", "datasets", "collected")
SEQUENCE_DIR: str    = os.path.join("ai", "datasets", "collected", "RECORDED")


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _spec(color, radius=3, thickness=1):
    return mp.solutions.drawing_utils.DrawingSpec(
        color=color, thickness=thickness, circle_radius=radius
    )


def draw_holistic(frame, results) -> None:
    """Draw all detected body-part landmarks with distinct colors."""
    mp_drawing  = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            _spec((80, 110, 10), radius=1),
            _spec((80, 256, 121), radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            _spec((245, 117, 66), radius=4),
            _spec((245, 66, 230), radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            _spec((121, 22, 76), radius=5),
            _spec((121, 44, 250), radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            _spec((80, 22, 10), radius=5),
            _spec((80, 44, 121), radius=2),
        )


def draw_hud(
    frame,
    fps: float,
    face_ok: bool,
    pose_ok: bool,
    left_ok: bool,
    right_ok: bool,
    landmarks_count: int,
    recording: bool,
    rec_progress: tuple[int, int],
    saved_seqs: int,
) -> None:
    """Draw the heads-up display panel on the frame."""
    h, w = frame.shape[:2]

    def _ok(flag):
        return (0, 220, 80) if flag else (80, 80, 80)

    # ── Top info bar ─────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 165), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 165), (30, 30, 30), 1)

    cv2.putText(frame, "SIGNBRIDGE AI", (w // 2 - 115, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Landmarks: {landmarks_count}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    cv2.putText(frame, f"Hand: {'YES' if (left_ok or right_ok) else 'NO'}", (220, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ok(left_ok or right_ok), 2)
    cv2.putText(frame, f"Pose: {'YES' if pose_ok else 'NO'}", (220, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ok(pose_ok), 2)
    cv2.putText(frame, f"Face: {'YES' if face_ok else 'NO'}", (220, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ok(face_ok), 2)

    # ── Recording indicator ───────────────────────────────────────────────────
    if recording:
        done, total = rec_progress
        ratio = done / total if total > 0 else 0
        bar_w = int((w - 40) * ratio)
        cv2.rectangle(frame, (20, 135), (w - 20, 155), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 135), (20 + bar_w, 155), (0, 60, 220), -1)
        cv2.putText(frame, f"REC {done}/{total}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)
    else:
        if saved_seqs > 0:
            cv2.putText(frame, f"Sequences saved: {saved_seqs}", (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)

    # ── Bottom controls bar ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "[S] Snapshot   [R] Record   [Q] Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def get_next_index(directory: str, prefix: str, ext: str) -> int:
    """Return the next sequential file index in a directory."""
    existing = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(ext)
    ]
    return len(existing) + 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(SEQUENCE_DIR, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        holistic.close()
        return

    print("SignBridge AI — Week 1 Demo")
    print("Controls: S = snapshot | R = record sequence | Q = quit")

    previous_time: float   = 0.0
    fps: float             = 0.0
    recording: bool        = False
    saved_seqs: int        = 0
    builder = SequenceBuilder(sequence_length=SEQUENCE_LENGTH)

    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # ── Holistic inference ────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        # ── FPS ───────────────────────────────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else fps
        previous_time = current_time

        # ── Landmark extraction ───────────────────────────────────────────────
        raw_landmarks = extract_landmarks(results)
        norm_landmarks = normalize_landmarks(raw_landmarks)

        # Count non-zero landmark values as "active landmarks"
        active = int(np.count_nonzero(raw_landmarks) // 3)

        # ── Recording ─────────────────────────────────────────────────────────
        if recording:
            builder.add_frame(norm_landmarks)
            if builder.is_ready():
                seq = builder.get_sequence()
                idx = get_next_index(SEQUENCE_DIR, "sequence_", ".npy")
                path = os.path.join(SEQUENCE_DIR, f"sequence_{idx:03d}.npy")
                np.save(path, seq)
                saved_seqs += 1
                print(f"Sequence saved: {path}  shape={seq.shape}")
                builder.reset()
                recording = False

        # ── Draw landmarks ────────────────────────────────────────────────────
        draw_holistic(frame, results)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(
            frame,
            fps=fps,
            face_ok=results.face_landmarks is not None,
            pose_ok=results.pose_landmarks is not None,
            left_ok=results.left_hand_landmarks is not None,
            right_ok=results.right_hand_landmarks is not None,
            landmarks_count=active,
            recording=recording,
            rec_progress=builder.progress(),
            saved_seqs=saved_seqs,
        )

        cv2.imshow("SignBridge AI — Week 1", frame)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit.")
            break

        if key == ord("s"):
            idx = get_next_index(SNAPSHOT_DIR, "snapshot_", ".jpg")
            path = os.path.join(SNAPSHOT_DIR, f"snapshot_{idx:03d}.jpg")
            cv2.imwrite(path, frame)
            print(f"Snapshot saved: {path}")

        if key == ord("r"):
            if not recording:
                recording = True
                builder.reset()
                print("Recording started …")
            else:
                print("Already recording.")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
