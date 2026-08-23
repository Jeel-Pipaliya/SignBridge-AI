"""
DAY 3 — SignBridge AI
MediaPipe Hands: detect up to 2 hands, draw landmarks, print coordinates.

Run:
    python ai/inference/hand_detection.py

Controls:
    Q — Quit
"""

import cv2
import mediapipe as mp
import time


# ─── Constants ────────────────────────────────────────────────────────────────
LANDMARK_NAMES = {
    0: "WRIST",
    4: "THUMB_TIP",
    8: "INDEX_TIP",
    12: "MIDDLE_TIP",
    16: "RING_TIP",
    20: "PINKY_TIP",
}


# ─── Draw helpers ─────────────────────────────────────────────────────────────

def draw_hand_info(frame, hand_label: str, hand_landmarks, pos_y: int) -> None:
    """Print hand label and wrist coordinate on screen."""
    wrist = hand_landmarks.landmark[0]
    cv2.putText(
        frame,
        f"{hand_label} | Wrist: ({wrist.x:.2f}, {wrist.y:.2f})",
        (20, pos_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2,
    )


def draw_ui(frame, fps: float, hand_count: int) -> None:
    h, w = frame.shape[:2]

    cv2.putText(
        frame, f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.putText(
        frame, f"Hands: {hand_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )

    # Controls
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, "[Q] Quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("MediaPipe Hands started. Press Q to quit.")

    previous_time: float = 0.0
    fps: float = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False          # performance hint
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # ── FPS ──────────────────────────────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else fps
        previous_time = current_time

        hand_count = 0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Which hand? (MediaPipe labels from its own perspective)
                handedness = results.multi_handedness[idx].classification[0].label

                # Draw connections + landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                draw_hand_info(frame, handedness, hand_landmarks, 100 + idx * 30)

                # Print landmark coordinates to terminal (useful for learning)
                print(f"\n── {handedness} Hand ──")
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    name = LANDMARK_NAMES.get(lm_idx, f"LM_{lm_idx:02d}")
                    print(f"  {name:12s}  x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}")

        draw_ui(frame, fps, hand_count)
        cv2.imshow("SignBridge AI — Day 3: Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit.")
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
