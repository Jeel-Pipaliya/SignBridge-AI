# SignBridge AI — Week 4 Milestone Completion Report

**Project Title:** SignBridge AI — Intelligent Real-Time AI Communication Platform for Deaf & Hard-of-Hearing People  
**Milestone:** Week 4 — 32-Class ISL Recognition Model & Real-Time Webcam Pipeline  
**Date:** September 2026  
**Status:** Completed, Evaluated & Fully Verified  

---

## 1. Executive Summary

Week 4 completes the core real-time recognition milestone of SignBridge AI. The system scales from the preliminary 12-class prototype to **32 Indian Sign Language classes** (Digits `1`–`5`, Alphabets `A`–`Z`, and Conversational Sign `HELLO`). The complete pipeline has been modularized with automated dataset statistics and validation, invariant landmark extraction, YAML-driven training with early stopping, rigorous multi-class evaluation with confusion matrices and classification reports, and a low-latency real-time inference engine operating on standard webcam hardware at **75+ FPS** and **13.2 ms** latency.

---

## 2. Existing Code Audit

* **What already existed:**
  * Raw image data across 32 class folders in `data/raw/`.
  * Initial feature CSV (`processed_landmarks.csv`) and basic train/val/test splits.
  * Basic hand detector and preliminary 12-class Random Forest model from Week 3.
  * 25 legacy unit tests for Weeks 1–3.
* **What was reused:**
  * MediaPipe hand detection integration (`ai/hand_detection.py`).
  * Translation and scale normalization principles (wrist at origin, anatomical scale factor).
  * Temporal sliding-window majority voting and debouncing logic (`realtime/recognizer.py`).
  * OpenCV camera hardware abstraction (`realtime/camera.py`).
* **What was changed & upgraded:**
  * Unified landmark extraction supporting single hand (63D), dual hand (126D), and safe zero-handling.
  * Created configurable `ISLClassifier` supporting Multi-Layer Perceptron neural networks, SVM (RBF), and Random Forest.
  * Standardized YAML training configuration (`configs/training.yaml`).
  * Replaced manual scripts with standardized CLI entry points (`ai.training.train`, `ai.evaluation.evaluate`, `ai.inference.realtime`).
  * Created automated dataset auditing tools (`ai/data_analysis/`).
  * Expanded automated test suite to 48 comprehensive unit and integration tests.

---

## 3. Dataset Audit & Validation

* **Total Classes:** 32 (`1`, `2`, `3`, `4`, `5`, `A`–`Z`, `HELLO`)
* **Total Samples:** 1,120 balanced landmark feature vectors
* **Split Partitioning (Stratified 70 / 15 / 15):**
  * **Training Samples:** 784 (24–25 samples per class)
  * **Validation Samples:** 168 (5–6 samples per class)
  * **Testing Samples:** 168 (5–6 samples per class)
* **Class Balance:**
  * Minimum samples / class: 35
  * Maximum samples / class: 35
  * Average samples / class: 35.00
* **Data Quality Verification:**
  * Raw images checked: 2,750
  * Valid images: 2,750
  * Corrupted files: 0
  * Missing labels: 0
  * Small images (<32px): 0
  * Duplicates detected & flagged: 138
  * Dataset Status: **OK**
* **Artifacts Generated:**
  * `reports/dataset_statistics.csv`
  * `reports/class_distribution.png`
  * `data/landmarks/train/`, `val/`, `test/` binary cache

---

## 4. Model Architecture & Training

* **Architecture:** Multi-Layer Perceptron (`MLP_NeuralNet`)
  * **Input Layer:** 63 invariant features (21 landmarks × 3 spatial coordinates)
  * **Hidden Layer 1:** 128 units, ReLU activation, L2 regularization ($\alpha = 0.001$)
  * **Hidden Layer 2:** 64 units, ReLU activation, L2 regularization ($\alpha = 0.001$)
  * **Output Layer:** 32 units with Softmax probability distribution
  * **Optimizer:** Adam (Initial $\text{lr} = 0.001$, adaptive learning rate schedule)
  * **Loss Function:** Multi-Class Categorical Cross-Entropy
* **Training Benchmarks:**

| Candidate Model | Train Accuracy | Validation Accuracy | Validation Macro F1 | Unseen Test Accuracy | Selection Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **MLP (Neural Network)** | **73.09%** | **69.05%** | **0.6248** | **66.07%** | **Selected Champion** |
| **Random Forest** | 100.00% | 67.86% | 0.6606 | 66.67% | Candidate (Overfitting on train) |
| **SVM (RBF Kernel)** | 79.72% | 66.67% | 0.6333 | 66.07% | Candidate |

* **Training History Artifact:**
  * `reports/training_history.png` (Training vs Validation Accuracy curve and Loss curve)
* **Checkpoints & Metadata:**
  * `models/checkpoints/isl_classifier_mlp_neuralnet_best.pkl`
  * `models/best/isl_classifier.pkl`
  * `models/metadata/model_metadata.json`
  * `models/metadata/class_names.json`

---

## 5. Model Evaluation (Unseen Test Set)

Evaluated on 168 untouched test samples across all 32 classes:

* **Overall Test Accuracy:** **66.07%**
* **Macro Precision:** **0.6042**
* **Macro Recall:** **0.6583**
* **Macro F1-Score:** **0.6085**
* **Weighted F1-Score:** **0.6127**
* **Experiment Log Entry:** `EXP-001` recorded in `experiments/experiment_log.csv`

### Misclassification Error Analysis (Most Confused Class Pairs)

1. **Sign `B` → Predicted `4` (6 occurrences):** Both signs feature 4 extended fingers; the primary differentiator is thumb placement against the palm vs. outstretched thumb.
2. **Sign `HELLO` → Predicted `5` (5 occurrences):** Both share an open 5-digit palm configuration. In static 2D frames, they are geometrically identical without motion dynamics.
3. **Sign `K` → Predicted `P` (5 occurrences):** Inverted hand orientation with similar finger spread.
4. **Sign `N` → Predicted `M` (5 occurrences):** Two vs. three folded fingers over the thumb (fine inter-finger occlusion).
5. **Sign `1` → Predicted `Z` (4 occurrences):** Single index finger extended; gesture trajectory distinguishes `Z` in dynamic signing.
6. **Sign `G` → Predicted `L` (4 occurrences):** Thumb and index finger perpendicular / parallel orientation.
7. **Sign `S` → Predicted `M` (4 occurrences):** Fist gesture with thumb wrapped over fingers.

### Generated Artifacts
* High-Resolution Heatmap: `reports/confusion_matrix.png`
* Tabular Breakdown: `reports/classification_report.csv`

---

## 6. Real-Time Webcam Inference Performance

* **Average Frame Processing Latency:** **13.21 ms** (Target: $\le 100\text{ ms}$)
* **95th Percentile Latency:** **13.81 ms**
* **Throughput:** **75.7 FPS** (Target: $\ge 15\text{ FPS}$)
* **Confidence Gating Threshold:** $0.70$ (rejects noisy or indeterminate poses)
* **Temporal Stabilization:** 5-frame sliding-window deque with $60\%$ majority voting
* **Text Debouncing:** $0.8\text{ s}$ cooldown between identical sign commits
* **No-Hand State:** Graceful decay, buffer reset, and accessible visual alert
* **User Interface:** High-contrast HUD displaying sign, confidence meter, FPS, latency, and sentence buffer.

---

## 7. Verification & Automated Test Suite

Executed command:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Results:**
* `test_preprocessing.py`: 6/6 tests passing (shape, wrist origin, translation/scale invariance, zero-guard)
* `test_landmarks.py`: 4/4 tests passing (single hand, dual hand, zero hand, extraction + norm)
* `test_model.py`: 4/4 tests passing (MLP construction, dynamic classes, probability outputs, metadata)
* `test_prediction.py`: 4/4 tests passing (prediction schema, threshold gating, no-hand, invalid dimensions)
* `test_utils.py`: 5/5 tests passing (config, logger, compute_metrics, top misclassifications)
* `test_week1.py`: 9/9 tests passing (legacy environment, invariant math, camera detector)
* `test_week2.py`: 8/8 tests passing (legacy EDA, splits, preprocessor)
* `test_week3.py`: 8/8 tests passing (legacy models, predictor, temporal smoother, HUD)

**Total:** **48 / 48 Automated Tests Passing (100% Success Rate)**.

---

## 8. Files Created & Modified in Week 4

### Created Files:
* `configs/training.yaml`
* `ai/utils/__init__.py`
* `ai/utils/config.py`
* `ai/utils/logger.py`
* `ai/utils/metrics.py`
* `ai/data_analysis/__init__.py`
* `ai/data_analysis/dataset_statistics.py`
* `ai/data_analysis/validate_dataset.py`
* `ai/preprocessing/normalization.py`
* `ai/preprocessing/feature_pipeline.py`
* `ai/models/isl_classifier.py`
* `ai/training/__init__.py`
* `ai/training/trainer.py`
* `ai/evaluation/__init__.py`
* `ai/inference/realtime.py`
* `tests/test_preprocessing.py`
* `tests/test_landmarks.py`
* `tests/test_model.py`
* `tests/test_prediction.py`
* `tests/test_utils.py`
* `reports/dataset_statistics.csv`
* `reports/class_distribution.png`
* `reports/training_history.png`
* `reports/confusion_matrix.png`
* `reports/classification_report.csv`
* `experiments/experiment_log.csv`
* `models/metadata/model_metadata.json`
* `models/metadata/class_names.json`
* `WEEK4_REPORT.md`
* `docs/week4_report.md`

### Modified Files:
* `requirements.txt` (added `pyyaml>=6.0.0`)
* `ai/preprocessing/landmark_extractor.py` (added robust 1-hand, 2-hand, 0-hand extraction & backwards compatibility)
* `ai/training/train.py` (integrated Trainer, YAML config, checkpointing)
* `ai/evaluation/evaluate.py` (integrated multi-class metrics, confusion matrix, misclassifications)
* `main.py` (integrated stats, validate, benchmark, and Week 4 modules)
* `README.md` (added Week 4 quickstart and results)
* `.gitignore` (added checkpoints, best models, and landmarks cache)

---

## 9. Week 5 Readiness

With the static 32-class real-time recognition pipeline fully verified, SignBridge AI is ready for Week 5:
1. **Dynamic Gesture & Temporal Sequence Modeling:** Moving from isolated static frames to dynamic gestures (`HELLO` wave, `J`, `Z`) using Bi-LSTM / Temporal Convolutional Networks (TCN).
2. **Grammar Correction Engine:** Laying groundwork for Indian Sign Language (OSV syntax) to English/Hindi translation.
3. **Text-to-Speech (TTS) Integration:** Connecting recognized sentence buffer to offline vocalization.
