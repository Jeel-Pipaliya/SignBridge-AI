# SignBridge AI — Week 3 Milestone Report

**Project Title:** SignBridge AI — Intelligent Real-Time AI Communication Platform for Deaf & Hard-of-Hearing People  
**Milestone:** Week 3 — Baseline Model Training & Real-Time Webcam Recognition  
**Date:** September 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

Week 3 achieves the primary milestone of SignBridge AI: training a machine learning classifier on normalized 63-dimensional ISL hand landmark features and connecting the complete pipeline to the live webcam. The system processes video frames in real time, extracts invariant landmarks, performs low-latency inference, applies temporal sliding-window smoothing to eliminate flicker, and accumulates recognized signs into an on-screen sentence buffer.

---

## 2. Completed Deliverables

| Day | Module / Component | File Location | Status | Key Deliverable |
|---|---|---|---|---|
| **Day 1** | Model Training & Selection | [`ai/train.py`](file:///d:/Project/signbridge-ai/ai/train.py) | ✅ Complete | Benchmarked Random Forest, SVM (RBF), and MLP. Champion: SVM (RBF). |
| **Day 2** | Model Evaluation & Diagnostics | [`ai/evaluate.py`](file:///d:/Project/signbridge-ai/ai/evaluate.py) | ✅ Complete | Confusion matrix (`outputs/confusion_matrix/confusion_matrix.png`) & per-class F1 metrics. |
| **Day 3** | Prediction Inference Engine | [`ai/predict.py`](file:///d:/Project/signbridge-ai/ai/predict.py) | ✅ Complete | Modular, reusable predictor returning structured labels and probabilities. |
| **Day 4** | Real-Time Recognition Loop | [`realtime/realtime_demo.py`](file:///d:/Project/signbridge-ai/realtime/realtime_demo.py) | ✅ Complete | Live webcam inference loop with confidence gating and modern HUD overlay. |
| **Day 5** | Temporal Prediction Smoothing | [`realtime/recognizer.py`](file:///d:/Project/signbridge-ai/realtime/recognizer.py) | ✅ Complete | Sliding-window majority voting ($N=5$) to prevent frame-to-frame sign flicker. |
| **Day 6** | Text Accumulation Buffer | [`realtime/recognizer.py`](file:///d:/Project/signbridge-ai/realtime/recognizer.py) | ✅ Complete | Stable sign accumulator with debounce logic, space, backspace, and clear controls. |
| **Day 7** | Full Live Demo & Testing | [`realtime/realtime_demo.py`](file:///d:/Project/signbridge-ai/realtime/realtime_demo.py), [`tests/test_week3.py`](file:///d:/Project/signbridge-ai/tests/test_week3.py) | ✅ Complete | 8 automated tests passing (25/25 tests overall across Weeks 1–3). |

---

## 3. Machine Learning Model Benchmark

Candidate classifiers were trained on 420 samples (70% stratified split) and evaluated on 90 validation and 90 test samples across the 12 prototype classes:

| Algorithm | Hyperparameters | Validation Accuracy | Validation Macro F1 | Unseen Test Accuracy | Unseen Macro F1 | Selection Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (RBF)** | $C=10.0$, `kernel='rbf'`, `probability=True` | **76.67%** | **0.7601** | **73.33%** | **0.7134** | **Selected Champion** |
| **Random Forest** | $n=100$, `max_depth=12`, `min_samples_split=2` | 73.33% | 0.7252 | 73.33% | 0.7128 | Baseline |
| **Multi-Layer Perceptron (MLP)** | $(128, 64)$ ReLU layers, `max_iter=500` | 76.67% | 0.7003 | 73.33% | 0.6980 | Candidate |

The **Support Vector Machine with Radial Basis Function (RBF) kernel** was selected as the champion model for its superior margin-based decision boundary in 63-dimensional geometric space and higher macro F1 score.

---

## 4. Quantitative Evaluation & Error Diagnostics

### Confusion Matrix Insights
* **Perfect Separation ($F1 = 1.00$):**
  * Classes `1` (single index finger extended)
  * Class `3` (three central fingers extended)
  * Class `A` (tight fist with resting thumb)
  * Class `C` (curved palm and fingers)
  * Class `L` (perpendicular thumb-index angle)
  * Class `Y` (outstretched thumb and pinky)
* **Topological Confusion Pairs:**
  * **Class `2` vs. Class `V`:** Both signs feature two extended fingers. Variations in finger spread angle can cause subtle boundary crossings.
  * **Class `5` vs. Class `HELLO`:** Both signs feature an open 5-digit palm configuration. In live deployment, `HELLO` is a conversational greeting while `5` is numeric.
* **Engineering Discussion (Why Accuracy $\neq$ Real-World Performance):**
  In real-world conditions, optical occlusion occurs when fingers overlap from the 2D camera perspective. MediaPipe's estimated depth coordinate ($z$) exhibits higher sensor noise than the image-plane coordinates ($x, y$). We mitigated this through **Confidence Gating** (rejecting predictions with confidence $< 0.70$) and **Sliding-Window Majority Voting** (requiring 60%+ agreement across consecutive frames).

---

## 5. End-to-End Real-Time Pipeline

```text
       ┌───────────────────────────────┐
       │         Live Webcam           │
       └──────────────┬────────────────┘
                      │ 640x480 @ ~30 FPS
                      ▼
       ┌───────────────────────────────┐
       │     MediaPipe Hand Detector   │
       │     (21 3D Coordinates)       │
       └──────────────┬────────────────┘
                      │ Raw Coordinates (x, y, z)
                      ▼
       ┌───────────────────────────────┐
       │  Invariant Feature Normalizer │
       │  (Origin @ Wrist, Scale Norm) │
       └──────────────┬────────────────┘
                      │ 63D Feature Vector
                      ▼
       ┌───────────────────────────────┐
       │   Trained SVM Classifier      │
       │   (Probability Inference)     │
       └──────────────┬────────────────┘
                      │ Label + Confidence Score
                      ▼
       ┌───────────────────────────────┐
       │   Confidence Threshold Gate   │
       │   (Score >= 0.70)             │
       └──────────────┬────────────────┘
                      │ Filtered Prediction
                      ▼
       ┌───────────────────────────────┐
       │   Temporal Smoothing Engine   │
       │   (5-Frame Majority Voting)   │
       └──────────────┬────────────────┘
                      │ Stable Sign (No Flicker)
                      ▼
       ┌───────────────────────────────┐
       │   Sentence Formation Buffer   │
       │   (Debounced Text Buffer)     │
       └──────────────┬────────────────┘
                      │ Committed Tokens
                      ▼
       ┌───────────────────────────────┐
       │        Modern Visual HUD      │
       │   Sign, Confidence, FPS, Text │
       └───────────────────────────────┘
```

---

## 6. Automated Test Suite Results

Running [`tests/test_week3.py`](file:///d:/Project/signbridge-ai/tests/test_week3.py):

```text
test_accumulation_and_controls: PASSED
test_evaluation_plots_exist: PASSED
test_flicker_rejection: PASSED
test_model_files_exist: PASSED
test_model_loading: PASSED
test_overlay_renders_on_frame: PASSED
test_predict_features_output_schema: PASSED
test_predict_invalid_features: PASSED

Ran 8 tests in 0.559s
OK (8/8 tests passing)
```

**Overall Project Verification (Weeks 1, 2, and 3):**
```bash
python -m unittest tests/test_week1.py tests/test_week2.py tests/test_week3.py
```
```text
Ran 25 tests in 0.253s
OK (All 25/25 tests passing)
```
