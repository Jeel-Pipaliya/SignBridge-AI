"""
DAY 4 — SignBridge AI
MediaPipe Holistic: Hands + Pose + Face in a single pipeline.

Run:
    python ai/inference/holistic_detection.py

Controls:
    Q — Quit
"""

import cv2
import mediapipe as mp
import time


# ─── Drawing spec helpers ─────────────────────────────────────────────────────

def _landmark_spec(color, radius=3):
    return mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=1, circle_radius=radius)


def _connection_spec(color, thickness=1):
    return mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=thickness)


# ─── UI overlay ───────────────────────────────────────────────────────────────

def draw_status(frame, fps, face_ok, pose_ok, left_ok, right_ok) -> None:
    h, w = frame.shape[:2]

    def colored(flag):
        return (0, 255, 80) if flag else (60, 60, 60)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Face:  {'YES' if face_ok  else 'NO'}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colored(face_ok), 2)
    cv2.putText(frame, f"Pose:  {'YES' if pose_ok  else 'NO'}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colored(pose_ok), 2)
    cv2.putText(frame, f"L Hand:{'YES' if left_ok  else 'NO'}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colored(left_ok), 2)
    cv2.putText(frame, f"R Hand:{'YES' if right_ok else 'NO'}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colored(right_ok), 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "[Q] Quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

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
        return

    print("MediaPipe Holistic started. Press Q to quit.")

    previous_time: float = 0.0
    fps: float = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # ── FPS ──────────────────────────────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else fps
        previous_time = current_time

        # ── Draw Face (contours only — less cluttered than all mesh) ─────────
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                _landmark_spec((80, 110, 10), radius=1),
                _connection_spec((80, 256, 121), thickness=1),
            )

        # ── Draw Pose ────────────────────────────────────────────────────────
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                _landmark_spec((245, 117, 66), radius=4),
                _connection_spec((245, 66, 230), thickness=2),
            )

        # ── Draw Left Hand ────────────────────────────────────────────────────
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                _landmark_spec((121, 22, 76), radius=4),
                _connection_spec((121, 44, 250), thickness=2),
            )

        # ── Draw Right Hand ───────────────────────────────────────────────────
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                _landmark_spec((80, 22, 10), radius=4),
                _connection_spec((80, 44, 121), thickness=2),
            )

        face_ok  = results.face_landmarks is not None
        pose_ok  = results.pose_landmarks is not None
        left_ok  = results.left_hand_landmarks is not None
        right_ok = results.right_hand_landmarks is not None

        draw_status(frame, fps, face_ok, pose_ok, left_ok, right_ok)
        cv2.imshow("SignBridge AI — Day 4: Holistic Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
