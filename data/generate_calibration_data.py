"""
SignBridge AI - Prototype Calibration Image Generator
Generates realistic prototype benchmark sign images for all 12 ISL classes
to ensure the pipeline is immediately runnable, testable, and verifiable
even before the full multi-gigabyte Mendeley dataset is downloaded.

Classes:
  Numerals: '1', '2', '3', '4', '5'
  Alphabets: 'A', 'B', 'C', 'L', 'V', 'Y'
  Word: 'HELLO'
"""

import sys
import os
import math
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.ingest_dataset import PROTOTYPE_CLASSES, ensure_directories


def draw_hand_pose(
    canvas: np.ndarray,
    wrist_pos: tuple,
    finger_states: dict,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    skin_color: tuple = (180, 200, 235),  # BGR skin tone
) -> np.ndarray:
    """
    Renders a realistic stylized hand with palm and 5 articulated fingers.

    finger_states: {
        'thumb':  'extended' | 'folded' | 'outward',
        'index':  'extended' | 'folded' | 'curved',
        'middle': 'extended' | 'folded' | 'curved',
        'ring':   'extended' | 'folded' | 'curved',
        'pinky':  'extended' | 'folded' | 'curved'
    }
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(dx, dy):
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        return int(wrist_pos[0] + rx * scale), int(wrist_pos[1] + ry * scale)

    # 1. Draw Palm Base
    palm_pts = np.array([
        rot(-30, 0), rot(30, 0), rot(40, -70), rot(-35, -70)
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [palm_pts], skin_color)
    cv2.polylines(canvas, [palm_pts], True, (140, 160, 200), 2)

    # 2. Finger Base & Direction Configs
    finger_bases = {
        'thumb':  (-28, -25),
        'index':  (-22, -70),
        'middle': (-4, -72),
        'ring':   (14, -70),
        'pinky':  (32, -65),
    }
    finger_lengths = {
        'thumb': 50,
        'index': 70,
        'middle': 78,
        'ring': 72,
        'pinky': 55,
    }

    # Draw each digit
    for digit, state in finger_states.items():
        base_x, base_y = finger_bases[digit]
        length = finger_lengths[digit]

        if digit == 'thumb':
            if state == 'outward':
                tip = (base_x - length * 0.9, base_y - length * 0.4)
            elif state == 'folded':
                tip = (base_x + 25, base_y - 10)
            else:  # extended
                tip = (base_x - length * 0.6, base_y - length * 0.8)
        else:
            if state == 'extended':
                tip = (base_x, base_y - length)
            elif state == 'curved':
                tip = (base_x + 15, base_y - length * 0.4)
            else:  # folded
                tip = (base_x, base_y + 15)

        p_base = rot(base_x, base_y)
        p_tip = rot(tip[0], tip[1])

        thickness = int(14 * scale)
        cv2.line(canvas, p_base, p_tip, skin_color, thickness, cv2.LINE_AA)
        cv2.line(canvas, p_base, p_tip, (140, 160, 200), max(1, int(2 * scale)), cv2.LINE_AA)
        cv2.circle(canvas, p_tip, int(7 * scale), skin_color, -1)

    return canvas


def get_pose_for_class(cls_label: str) -> dict:
    """Map prototype ISL class label to finger articulation states."""
    if cls_label == "1":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'folded', 'ring': 'folded', 'pinky': 'folded'}
    elif cls_label == "2":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'extended', 'ring': 'folded', 'pinky': 'folded'}
    elif cls_label == "3":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'folded'}
    elif cls_label == "4":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}
    elif cls_label == "5" or cls_label == "HELLO":
        return {'thumb': 'outward', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}
    elif cls_label == "A":
        return {'thumb': 'extended', 'index': 'folded', 'middle': 'folded', 'ring': 'folded', 'pinky': 'folded'}
    elif cls_label == "B":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}
    elif cls_label == "C":
        return {'thumb': 'curved', 'index': 'curved', 'middle': 'curved', 'ring': 'curved', 'pinky': 'curved'}
    elif cls_label == "L":
        return {'thumb': 'outward', 'index': 'extended', 'middle': 'folded', 'ring': 'folded', 'pinky': 'folded'}
    elif cls_label == "V":
        return {'thumb': 'folded', 'index': 'extended', 'middle': 'extended', 'ring': 'folded', 'pinky': 'folded'}
    elif cls_label == "Y":
        return {'thumb': 'outward', 'index': 'folded', 'middle': 'folded', 'ring': 'folded', 'pinky': 'outward'}
    else:
        return {'thumb': 'extended', 'index': 'extended', 'middle': 'extended', 'ring': 'extended', 'pinky': 'extended'}


def generate_benchmark_images(samples_per_class: int = 25) -> int:
    """
    Generate calibration benchmark images across all 12 classes with natural variations:
    varied scales, subtle rotation angles, skin tones, noise, and backgrounds.
    """
    ensure_directories()
    total_generated = 0
    np.random.seed(42)

    for cls in PROTOTYPE_CLASSES:
        cls_dir = config.RAW_DATA_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        pose = get_pose_for_class(cls)

        for i in range(samples_per_class):
            img_path = cls_dir / f"{cls}_{i+1:03d}.jpg"

            # Vary canvas background (dark, neutral, office-like)
            bg_color = np.random.randint(40, 90, size=3, dtype=np.uint8)
            canvas = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), bg_color, dtype=np.uint8)

            # Add subtle background gradient
            grad_1d = np.linspace(0, 30, config.CAMERA_HEIGHT, dtype=np.uint8)
            grad_3ch = np.dstack([np.tile(grad_1d[:, None], (1, config.CAMERA_WIDTH))] * 3)
            canvas = cv2.add(canvas, grad_3ch)

            # Natural variations
            scale = np.random.uniform(0.85, 1.25)
            angle = np.random.uniform(-12.0, 12.0)
            offset_x = int(np.random.uniform(-30, 30))
            offset_y = int(np.random.uniform(-25, 25))
            wrist_pos = (config.CAMERA_WIDTH // 2 + offset_x, int(config.CAMERA_HEIGHT * 0.72) + offset_y)

            # Vary skin tone slightly
            tone_adj = int(np.random.uniform(-15, 15))
            skin_color = (
                max(0, min(255, 180 + tone_adj)),
                max(0, min(255, 200 + tone_adj)),
                max(0, min(255, 235 + tone_adj)),
            )

            draw_hand_pose(canvas, wrist_pos, pose, scale=scale, angle_deg=angle, skin_color=skin_color)

            # Add gentle Gaussian noise and slight blur to simulate real camera sensors
            noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
            noisy_canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(noisy_canvas, (3, 3), 0.5)

            cv2.imwrite(str(img_path), blurred)
            total_generated += 1

    print(f"[Success] Generated {total_generated} prototype calibration images across {len(PROTOTYPE_CLASSES)} classes.")
    return total_generated


if __name__ == "__main__":
    generate_benchmark_images(samples_per_class=25)
