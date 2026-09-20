# 🚀 SIGNBRIDGE AI

### Intelligent Real-Time AI Communication Platform for Deaf & Hard-of-Hearing People

SignBridge AI is an AI-powered assistive communication platform that translates **Indian Sign Language (ISL)** into understandable text and speech in real time. Designed as a final-year B.Tech AIML capstone project, it bridges the communication barrier between the deaf & hard-of-hearing community and society through robust computer vision and machine learning.

---

## 📌 Problem Statement

Over 63 million people in India experience significant auditory impairment. Indian Sign Language (ISL) is their primary medium of expression, yet the vast majority of the general public lacks sign language literacy. This creates persistent friction in educational institutions, healthcare facilities, workplaces, and daily public interactions.

SignBridge AI addresses this divide by providing an open-source, affordable, real-time vision-based ISL recognition engine that functions on standard consumer webcams without requiring specialized sensory gloves.

---

## 🧠 System Architecture

```text
Webcam Frame (640x480 @ 30 FPS)
       │
       ▼
OpenCV Capture Engine (`realtime/camera.py`)
       │
       ▼
MediaPipe Hands Detector (`ai/hand_detection.py`)
       │
       ▼
21 3D Hand Landmarks (x, y, z)
       │
       ▼
Feature Extraction & Invariant Normalizer (`ai/feature_extraction.py`)
   ├── Translation Invariance: Centered relative to wrist (lm0 -> 0, 0, 0)
   └── Scale Invariance: Normalized by wrist-to-middle MCP distance
       │
       ▼
Invariant 63-Dimensional Feature Vector
       │
       ▼
Trained ML Classifier (`models/isl_classifier.pkl` - SVM RBF)
       │
       ▼
Confidence Threshold Gate (Score >= 0.70)
       │
       ▼
Temporal Smoothing Engine (`realtime/recognizer.py` - 5-Frame Majority Vote)
       │
       ▼
Text Accumulation Engine & Interactive HUD (`realtime/realtime_demo.py`)
```

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.10+
* **Computer Vision:** OpenCV (`opencv-python`), MediaPipe Hands
* **Data Processing:** NumPy, Pandas
* **Machine Learning:** Scikit-Learn (SVM, Random Forest, MLP), Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Testing:** Unittest (25 automated integration tests)

---

## 📁 Project Directory Structure

```text
SignBridge-AI/
│
├── README.md                           # Master project documentation
├── requirements.txt                    # Core Python dependencies
├── .gitignore                          # Git exclusions (caches, models, raw data)
├── LICENSE                             # MIT Open-Source License
├── config.py                           # Central configuration (paths, thresholds, camera)
│
├── data/
│   ├── raw/                            # Original images and collected raw CSVs
│   │   ├── collected_landmarks.csv     # Live data collection output
│   │   └── README.md
│   ├── processed/                      # Preprocessed feature datasets (63D)
│   │   ├── processed_landmarks.csv     # Compiled feature dataset (600 samples)
│   │   ├── train.csv                   # 70% Stratified training set
│   │   ├── val.csv                     # 15% Stratified validation set
│   │   └── test.csv                    # 15% Stratified unseen test set
│   └── class_mapping.json              # Deterministic class-to-index mapping
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Class distributions & sample visualization
│   └── 03_model_training.ipynb         # Model benchmarking (RF vs SVM vs MLP)
│
├── ai/
│   ├── __init__.py
│   ├── hand_detection.py               # MediaPipe Hands reusable wrapper
│   ├── feature_extraction.py           # 63D invariant feature normalizer
│   ├── visualize_landmarks.py          # 21-joint anatomical inspector
│   ├── preprocessing.py                # Batch image-to-feature preprocessor
│   ├── dataset_loader.py               # Stratified 70/15/15 train/val/test loader
│   ├── train.py                        # Model training and architecture comparison
│   ├── evaluate.py                     # Metrics, classification report & confusion matrix
│   ├── predict.py                      # Standalone real-time inference engine
│   └── model/
│       └── README.md
│
├── realtime/
│   ├── __init__.py
│   ├── camera.py                       # Modular OpenCV video capture with FPS tracking
│   ├── recognizer.py                   # Temporal smoothing & text accumulation
│   └── realtime_demo.py                # Main live webcam executable
│
├── models/
│   ├── .gitkeep
│   ├── isl_classifier.pkl              # Trained SVM (RBF) classifier checkpoint
│   └── label_encoder.pkl               # Serialized LabelEncoder
│
├── outputs/
│   ├── plots/
│   │   ├── class_distribution.png      # Class sample count bar chart
│   │   ├── sample_grid.png             # 12-class representative sample grid
│   │   └── per_class_f1.png            # Per-class F1-score performance
│   ├── confusion_matrix/
│   │   └── confusion_matrix.png        # High-resolution confusion matrix heatmap
│   └── predictions/
│
├── docs/
│   ├── isl_dataset_research.md         # Research report on 5 verified ISL datasets
│   ├── prototype_vocabulary.md         # 12-class prototype vocabulary specification
│   ├── week1_report.md                 # Week 1 milestone report
│   ├── week2_report.md                 # Week 2 milestone report
│   └── week3_report.md                 # Week 3 milestone report
│
├── tests/
│   ├── test_week1.py                   # Week 1 integration tests (9 tests)
│   ├── test_week2.py                   # Week 2 integration tests (8 tests)
│   └── test_week3.py                   # Week 3 integration tests (8 tests)
│
├── frontend/                           # Reserved for future Web UI
├── backend/                            # Reserved for future API backend
├── speech/                             # Reserved for future TTS/STT
├── translation/                        # Reserved for ISL Grammar translation
└── dashboard/                          # Reserved for analytics dashboard
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Jeel-Pipaliya/SignBridge-AI.git
cd SignBridge-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite (25 Tests)

```bash
python -m unittest tests/test_week1.py tests/test_week2.py tests/test_week3.py
```

---

## 🎮 How to Run Live Applications

| Module / Milestone | Command | Purpose & Controls |
|---|---|---|
| **Live Recognition Demo (Week 3)** | `python realtime/realtime_demo.py` | **Full End-to-End Live Sign Recognition**<br>`[SPACE]`: Space, `[BACKSPACE]`: Delete, `[C]`: Clear, `[Q]`: Quit |
| **Data Collection Tool (Week 1)** | `python data/collect_samples.py` | Collects live sign samples into CSV (`data/raw/`)<br>`[SPACE]`: Capture, `[B]`: Burst mode, `[N]`: New label |
| **Landmark Inspector (Week 1)** | `python ai/visualize_landmarks.py` | Anatomical 21-joint inspection with index labels (0–20) |
| **Camera Test (Week 1)** | `python realtime/camera.py` | Tests camera stream & FPS display (`[Q]` to exit) |
| **Model Training (Week 3)** | `python ai/train.py` | Trains & benchmarks RF, SVM, MLP; exports champion model |
| **Evaluation & Metrics (Week 3)** | `python ai/evaluate.py` | Generates Confusion Matrix and F1 performance plots |

---

## 📊 Evaluation Results (Measured on Unseen Test Set)

* **Dataset Size:** 600 samples across 12 prototype classes
* **Champion Model:** Support Vector Machine with RBF Kernel ($C=10.0$)
* **Overall Test Accuracy:** **73.33%**
* **Macro Precision:** **0.7142**
* **Macro Recall:** **0.7440**
* **Macro F1-Score:** **0.7134**

### Perfect Separation Classes ($F1 = 1.00$):
* `1` (Single index finger extended)
* `3` (Three central fingers extended)
* `A` (Fist with thumb resting alongside index)
* `C` (Curved fingers and palm forming an arc)
* `L` (Perpendicular index finger and thumb)
* `Y` (Extended thumb and pinky)

---

## ⚠️ Limitations & Real-World Challenges

1. **Topological Ambiguities:**
   * Sign `2` vs. `V`: Both signs share two extended digits; small perspective variations can cause boundary crossings.
   * Sign `5` vs. `HELLO`: Both feature an open 5-digit palm configuration.
2. **Optical Occlusion:**
   * When fingers overlap from the perspective of a single 2D camera, MediaPipe's estimated depth ($z$) coordinate is noisier than $(x, y)$.
3. **Lighting & Motion Blur:**
   * Rapid gestures cause temporal landmark jitter, which is actively stabilized using our sliding-window majority vote.

---

## 🎯 Week 4 Milestone: 32-Class ISL Recognition & Real-Time Pipeline

Week 4 scales the prototype from 12 to **32 full Indian Sign Language classes** (`1`–`5`, `A`–`Z`, `HELLO`), incorporating an end-to-end modular pipeline, automated dataset validation, YAML configuration, Multi-Layer Perceptron neural network architecture, and a low-latency real-time inference loop.

### Quick Start Commands

#### 1. Installation
```bash
pip install -r requirements.txt
```

#### 2. Dataset Verification & Quality Audit
```bash
# Compute comprehensive dataset statistics and class distribution
python -m ai.data_analysis.dataset_statistics

# Run automated data health and corruption check
python -m ai.data_analysis.validate_dataset
```

#### 3. Model Training
```bash
# Train candidate classifiers (MLP Neural Net, SVM, Random Forest) with YAML configuration
python -m ai.training.train
```

#### 4. Model Evaluation & Diagnostics
```bash
# Run test set evaluation, generate confusion matrix & classification report
python -m ai.evaluation.evaluate
```

#### 5. Real-Time Webcam Recognition
```bash
# Launch live webcam recognition with temporal stabilization & HUD
python -m ai.inference.realtime

# Run headless latency & FPS benchmark
python -m ai.inference.realtime --benchmark
```

#### 6. Automated Testing
```bash
# Run all 48 automated unit and integration tests
python main.py --mode test
```

### Empirical Week 4 Results

* **Vocabulary:** 32 ISL Classes (`1`–`5`, `A`–`Z`, `HELLO`)
* **Dataset:** 1,120 balanced landmark samples (784 train / 168 val / 168 test)
* **Champion Model:** Multi-Layer Perceptron (`MLP_NeuralNet` — Dense(128, ReLU) -> Dense(64, ReLU) -> Softmax(32))
* **Test Accuracy:** **66.07%** across 32 classes
* **Real-Time Latency:** **13.2 ms** per frame
* **Throughput:** **75+ FPS** on standard CPU
* **Confidence Gate:** 0.70 threshold with 5-frame sliding-window temporal smoothing

---

## 📄 License

This project is licensed under the [MIT License](file:///d:/Project/signbridge-ai/LICENSE).
