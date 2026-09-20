"""
SignBridge AI - Central Configuration
Defines paths, camera settings, model hyper-parameters, and feature extraction settings.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CONFUSION_MATRIX_DIR = OUTPUTS_DIR / "confusion_matrix"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Model & Feature Settings
MODEL_PATH = MODELS_DIR / "isl_classifier.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LANDMARKS_CSV_PATH = PROCESSED_DATA_DIR / "processed_landmarks.csv"
RAW_COLLECTED_CSV = RAW_DATA_DIR / "collected_landmarks.csv"

NUM_LANDMARKS = 21
NUM_COORDINATES = 3  # (x, y, z)
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDINATES  # 63 features

# MediaPipe Hands Settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.70
MIN_TRACKING_CONFIDENCE = 0.50

# Camera Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Real-Time Recognition & Temporal Smoothing Settings
CONFIDENCE_THRESHOLD = 0.50  # Balanced confidence threshold for 32-class inference
SMOOTHING_WINDOW = 5         # Number of recent frames for majority vote
CONSECUTIVE_FRAMES_TO_ADD = 8  # Stable frames needed before adding to sentence
DEBOUNCE_SECONDS = 1.0       # Cooldown between adding the same sign twice

# Keybindings
KEY_EXIT = ord('q')
KEY_SAVE_SAMPLE = ord('s')
KEY_SPACE = 32
KEY_BACKSPACE = 8
KEY_CLEAR = ord('c')
