# SignBridge AI

**An AI-powered Sign Language Recognition System.**

---

## Week 1 — Camera → Landmark → Dataset Pipeline

The Week 1 goal is to build a reliable data pipeline:

```
Webcam → OpenCV → MediaPipe Holistic → Landmark Extraction → .npy Dataset
```

---

## Project Structure

```
signbridge-ai/
├── ai/
│   ├── datasets/
│   │   ├── raw/           # raw video (gitignored)
│   │   ├── processed/     # intermediate files
│   │   └── collected/     # recorded .npy sequences
│   │
│   ├── preprocessing/
│   │   ├── landmark_extractor.py   # Day 5 — 225-feature vectors
│   │   ├── normalize.py            # Day 5 — body-relative normalization
│   │   └── sequence_builder.py    # Day 6 — (30, 225) sequence builder
│   │
│   ├── inference/
│   │   ├── webcam.py               # Day 2 — OpenCV + FPS + save
│   │   ├── hand_detection.py       # Day 3 — MediaPipe Hands
│   │   ├── holistic_detection.py   # Day 4 — Holistic pipeline
│   │   └── signbridge_camera.py    # Day 7 — Week 1 integration
│   │
│   └── evaluation/
│
├── tests/
│   ├── test_setup.py       # Day 1 — environment check
│   ├── test_landmarks.py   # Day 5 — extractor + normalize tests
│   └── test_sequence.py    # Day 6 — SequenceBuilder tests
│
├── backend/
├── frontend/
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Create & activate virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe numpy pandas matplotlib
pip freeze > requirements.txt
```

### 3. Verify environment

```bash
python tests/test_setup.py
```

---

## Day-by-Day Scripts

| Day | Script | Run command |
|-----|--------|-------------|
| Day 1 | `tests/test_setup.py` | `python tests/test_setup.py` |
| Day 2 | `ai/inference/webcam.py` | `python ai/inference/webcam.py` |
| Day 3 | `ai/inference/hand_detection.py` | `python ai/inference/hand_detection.py` |
| Day 4 | `ai/inference/holistic_detection.py` | `python ai/inference/holistic_detection.py` |
| Day 5 | `tests/test_landmarks.py` | `python tests/test_landmarks.py` |
| Day 6 | `ai/datasets/collect_data.py` | `python -m ai.datasets.collect_data` |
| Day 7 | `ai/inference/signbridge_camera.py` | `python ai/inference/signbridge_camera.py` |

---

## Keyboard Controls

| Script | Key | Action |
|--------|-----|--------|
| `webcam.py` | `S` | Save snapshot |
| `webcam.py` | `Q` | Quit |
| `signbridge_camera.py` | `S` | Save snapshot |
| `signbridge_camera.py` | `R` | Record 30-frame sequence |
| `signbridge_camera.py` | `Q` | Quit |
| `collect_data.py` | `SPACE` | Start recording |
| `collect_data.py` | `Q` | Cancel / Quit |

---

## Feature Vector Layout

```
Left  Hand → 21 landmarks × 3 = 63  values   [0  : 63 )
Right Hand → 21 landmarks × 3 = 63  values   [63 : 126)
Pose       → 33 landmarks × 3 = 99  values   [126: 225)
─────────────────────────────────────────────
Total                           225  values
```

Each sequence: shape `(30, 225)` — 30 frames × 225 features.

---

## Collecting Your First Signs

```bash
python -m ai.datasets.collect_data
```

Suggested starter signs: `HELLO`, `YES`, `NO`, `THANK_YOU`, `HELP`

Collect **20–30 sequences per sign** (5 minimum for testing).

---

## Running Tests

```bash
python tests/test_landmarks.py
python tests/test_sequence.py
```

---

## Week 1 Checklist

- [x] Python 3.9+ installed
- [x] Virtual environment created
- [x] All packages installed
- [x] Project folders structured
- [x] Webcam script works
- [x] MediaPipe Hands detection works
- [x] MediaPipe Holistic works (Hands + Pose + Face)
- [x] Landmark extractor produces (225,) vectors
- [x] Normalization implemented
- [x] Sequence builder produces (30, 225) arrays
- [x] Data collection pipeline saves `.npy` files
- [x] Week 1 integration demo runs
- [ ] 5 test signs collected (20-30 sequences each)

---

---

## Week 2 — ISL Single-Sign Recognition (Random Forest)

### Goal
Camera → Hand Landmarks → Random Forest → Sign + Confidence

### Week 2 File Map

| Day | File | Purpose |
|-----|------|---------|
| Day 8 | _(structure + deps)_ | `scikit-learn`, `joblib` installed |
| Day 9 | [`backend/recognition/hand_detector.py`](backend/recognition/hand_detector.py) | HandDetector class |
| Day 10 | [`backend/preprocessing/landmark_extractor.py`](backend/preprocessing/landmark_extractor.py) | 63-feature wrist-relative vector |
| Day 11 | [`backend/data/collect_data.py`](backend/data/collect_data.py) | CSV dataset builder (SPACE/N/Q) |
| Day 12 | [`backend/models/train_model.py`](backend/models/train_model.py) | Train Random Forest, save .pkl |
| Day 12 | [`backend/models/evaluate_model.py`](backend/models/evaluate_model.py) | Confusion matrix + CV score |
| Day 13-14 | [`backend/recognition/realtime_recognition.py`](backend/recognition/realtime_recognition.py) | Live recognition + smoothing |

### Week 2 Quick Start

```powershell
# Day 8: install deps
.venv\Scripts\pip install scikit-learn joblib

# Day 9: test hand detector
.venv\Scripts\python backend/recognition/hand_detector.py

# Day 11: collect dataset (500 samples per sign)
.venv\Scripts\python backend/data/collect_data.py

# Day 12: train model
.venv\Scripts\python backend/models/train_model.py

# Day 12: evaluate
.venv\Scripts\python backend/models/evaluate_model.py

# Day 13-14: live recognition
.venv\Scripts\python backend/recognition/realtime_recognition.py

# Week 2 unit tests
.venv\Scripts\python tests/test_week2.py
```

### ISL Signs (Starter Set)
`hello`, `thank_you`, `yes`, `no`, `help`, `please`, `sorry`, `good`, `bad`, `stop`

Collect **500+ samples per sign** for reliable accuracy.

---

*Week 3: LSTM/GRU temporal sequence model for continuous sign recognition.*
