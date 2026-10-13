# SignBridge AI — Week 1 Milestone Report

**Project Title:** SignBridge AI — Intelligent Real-Time AI Communication Platform for Deaf & Hard-of-Hearing People  
**Milestone:** Week 1 — Project Foundation & Computer Vision Pipeline  
**Date:** September 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

During Week 1, the foundational computer-vision and feature engineering pipeline for SignBridge AI was constructed from scratch. We established a clean modular architecture separating hardware capture, hand tracking, geometric normalization, live data collection, and central configuration. All components have been verified with automated unit tests and mathematical invariance proofs.

---

## 2. Completed Deliverables

| Day | Module / Component | File Location | Status | Key Deliverable |
|---|---|---|---|---|
| **Day 1** | Project Setup & Config | [`config.py`](file:///d:/Project/signbridge-ai/config.py), [`.gitignore`](file:///d:/Project/signbridge-ai/.gitignore), [`requirements.txt`](file:///d:/Project/signbridge-ai/requirements.txt), [`LICENSE`](file:///d:/Project/signbridge-ai/LICENSE) | ✅ Complete | Modular configuration, clean virtual environment, standard repo layout. |
| **Day 2** | Camera Capture Engine | [`realtime/camera.py`](file:///d:/Project/signbridge-ai/realtime/camera.py) | ✅ Complete | Robust OpenCV webcam wrapper with FPS smoothing and graceful error handling. |
| **Day 3** | Hand & Landmark Detector | [`ai/hand_detection.py`](file:///d:/Project/signbridge-ai/ai/hand_detection.py) | ✅ Complete | MediaPipe Hands integration returning structured `HandData`, handedness, and landmarks. |
| **Day 4** | Landmark Understanding | [`ai/visualize_landmarks.py`](file:///d:/Project/signbridge-ai/ai/visualize_landmarks.py) | ✅ Complete | Anatomical visualization of 21 landmarks (0–20) with finger-group color coding. |
| **Day 5** | Feature Normalization | [`ai/feature_extraction.py`](file:///d:/Project/signbridge-ai/ai/feature_extraction.py) | ✅ Complete | 63D feature vector normalized for translation (wrist at origin) and scale invariance. |
| **Day 6** | Guided Data Collector | [`data/collect_samples.py`](file:///d:/Project/signbridge-ai/data/collect_samples.py) | ✅ Complete | Interactive prototype recording live signs into standard CSV format (`data/raw/`). |
| **Day 7** | Testing & Documentation | [`tests/test_week1.py`](file:///d:/Project/signbridge-ai/tests/test_week1.py), `docs/week1_report.md` | ✅ Complete | 9 automated integration tests, complete milestone documentation. |

---

## 3. Computer Vision & Feature Engineering Pipeline

The end-to-end Week 1 pipeline functions as follows:

```text
 Webcam Frame (BGR, 640x480)
            │
            ▼
    OpenCV (Flip & Color Convert to RGB)
            │
            ▼
  MediaPipe Hands (21 3D Landmarks)
            │
            ▼
 Feature Normalizer (Translation & Scale Invariance)
   1. Subtract Wrist (lm0) -> Wrist becomes (0, 0, 0)
   2. Divide by Scale Distance (Wrist to Middle MCP base)
            │
            ▼
   Flattened 63-Dimensional Feature Vector
            │
            ▼
  Standardized CSV Dataset / Classifier Ready
```

---

## 4. Mathematical Verification of Feature Invariance

To ensure the classifier does not overfit to where the user's hand is located in the frame or how close they stand to the webcam, `ai/feature_extraction.py` enforces mathematical invariance:

1. **Translation Invariance:** Shifting landmark coordinates by an arbitrary displacement vector $(\Delta x, \Delta y, \Delta z)$ produces identical features (Max numerical difference: $3.58 \times 10^{-7}$).
2. **Scale Invariance:** Scaling the hand by arbitrary factors relative to the wrist produces identical features (Max numerical difference: $1.19 \times 10^{-7}$).
3. **Combined Invariance:** Simultaneously shifting and scaling yields identical features (Max numerical difference: $6.22 \times 10^{-7}$).

---

## 5. Automated Test Results

Running the automated test suite ([`tests/test_week1.py`](file:///d:/Project/signbridge-ai/tests/test_week1.py)):

```text
Ran 9 tests in 0.178s
OK

[✓] test_required_directories_exist: PASSED
[✓] test_config_constants: PASSED
[✓] test_detector_initialization: PASSED
[✓] test_detector_blank_frame: PASSED
[✓] test_feature_vector_dimension (63D): PASSED
[✓] test_feature_names_count: PASSED
[✓] test_wrist_at_origin_after_normalization: PASSED
[✓] test_invariance_verification: PASSED
[✓] test_csv_creation_and_append: PASSED
```

---

## 6. Problems Encountered & Engineering Solutions

1. **Camera Resource Locking on Windows:**
   - *Problem:* Direct `cv2.VideoCapture(0)` can sometimes freeze or hang when attempting default backends on Windows.
   - *Solution:* Added `cv2.CAP_DSHOW` as preferred backend with automatic fallback, and implemented the context manager protocol (`__enter__` / `__exit__`) to guarantee hardware release on exit or exceptions.
2. **Division-by-Zero in Scale Normalization:**
   - *Problem:* If hands are partially obscured or middle MCP joint is co-located with the wrist, norm calculation could yield zero.
   - *Solution:* Added defensive epsilon threshold (`1e-6`) with fallback to maximum landmark distance.
3. **Data Leakage in CSV Writing:**
   - *Problem:* Ad-hoc headers or varying column counts across runs break machine learning dataframes.
   - *Solution:* Implemented automated single-source header generation via `get_feature_names()` (63 column names + 1 label column).

---

## 7. Week 2 Objectives

With the computer-vision and feature extraction foundation fully operational, Week 2 will focus on:
1. Researching and selecting verified academic Indian Sign Language (ISL) datasets.
2. Establishing a 10–20 class prototype vocabulary (alphabets/numbers/core signs).
3. Batch preprocessing of dataset images into landmark features (`processed_landmarks.csv`).
4. Generating exploratory data analysis (EDA) distribution plots and class balance reports.
5. Creating stratified Train/Validation/Test splits ready for model training in Week 3.
