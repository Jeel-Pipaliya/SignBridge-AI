# 🚀 SIGNBRIDGE AI

### Intelligent Real-Time AI Communication & Learning Platform for Indian Sign Language (ISL)

SignBridge AI is an end-to-end assistive communication platform that translates **Indian Sign Language (ISL)** into understandable text and speech in real time. Designed as a production-minded B.Tech AIML capstone project, it bridges the barrier between the deaf & hard-of-hearing community and society through robust computer vision, neural sequence modeling, deterministic grammar reconstruction, and accessible full-stack engineering.

---

## 📌 Problem Statement

Over 63 million individuals in India experience significant auditory impairment. Indian Sign Language (ISL) is their primary medium of expression, yet the vast majority of the general public lacks sign language literacy. This creates persistent friction in educational institutions, healthcare facilities, workplaces, and daily public interactions.

SignBridge AI addresses this divide by providing an open-source, affordable, real-time vision-based ISL recognition engine that functions on standard consumer webcams without requiring specialized sensory gloves.

---

## 🧠 System Architecture

```text
                                CLIENT LAYER (React 18 + Vite + Tailwind CSS)
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ • Live Recognition Studio (/recognize)      • 32-Sign Educational Tutor (/tutor)       │
   │ • Two-Way Speech & Chat (/communication)     • Meeting & Lecture Captions (/captions)   │
   │ • High-Contrast Emergency Mode (/emergency) • Analytics & Telemetry (/analytics)       │
   └────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                 JSON / REST API / HTTPS
                                               ▼
                              BACKEND LAYER (FastAPI + Python 3.12)
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ • JWT Authentication & Bcrypt Hashing       • CORS & Input Schema Validation (Pydantic)│
   │ • Invariant Normalizer (Wrist + MCP)        • ISL Grammar Reconstruction & Translation │
   │ • Telemetry & Practice Analytics            • Async Non-Blocking TTS Worker (pyttsx3)  │
   └────────────────────────────────────────────────────────────────────────────────────────┘
                       │                                             │
                       ▼                                             ▼
             PERSISTENCE LAYER                             AI & INFERENCE ENGINES
   ┌───────────────────────────────────┐         ┌──────────────────────────────────────────┐
   │ • PostgreSQL 16 (Production)      │         │ • Static 32-Class MLP Classifier (66.07%)│
   │ • SQLite fallback (Local Dev)     │         │ • Dynamic 3-Class Bi-LSTM Sequence (100%)│
   │ • Alembic Database Migrations     │         │ • MediaPipe Hands (21 3D Joint Nodes)    │
   └───────────────────────────────────┘         └──────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS (WCAG 2.1 AA Compliant), Lucide React, TanStack Query, React Router v6.
* **Backend:** Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic, Bcrypt, PyJWT, Uvicorn.
* **Computer Vision & AI:** MediaPipe Hands, OpenCV, PyTorch, Scikit-Learn, NumPy, Pandas.
* **Speech & Language:** Web Speech API, Offline pyttsx3 SAPI5 with phonetic transliteration fallback, deterministic ISL Grammar Engine.
* **Database & DevOps:** PostgreSQL 16, SQLite (dev), Docker, Docker Compose, GitHub Actions CI.
* **Testing:** Unittest & Pytest (96 automated integration and invariant unit tests passing).

---

## 📁 Project Directory Structure

```text
SignBridge-AI/
├── README.md                           # Master project documentation
├── requirements.txt                    # Core Python dependencies
├── config.py                           # Central configuration
├── docker-compose.yml                  # Production Docker Compose stack
├── alembic.ini                         # Alembic migration configuration
├── .gitignore                          # Git exclusions
│
├── frontend/                           # React 18 + TypeScript + Vite + Tailwind Client
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── api/                        # Typed API client & interfaces
│   │   ├── context/                    # AuthContext & AccessibilityContext
│   │   ├── components/                 # Navbar, Footer, ProtectedRoute, StatCard
│   │   ├── hooks/                      # useWebcam hook with permission management
│   │   ├── pages/                      # 11 Full Accessible Pages
│   │   ├── App.tsx                     # Routing & providers
│   │   └── main.tsx
│   └── Dockerfile                      # Production Nginx multi-stage Dockerfile
│
├── backend/                            # FastAPI REST Backend Service
│   ├── Dockerfile
│   ├── .env.example
│   ├── migrations/                     # Alembic migration scripts
│   └── app/
│       ├── main.py                     # Master FastAPI application entrypoint
│       ├── core/                       # Settings & JWT security
│       ├── db/                         # SQLAlchemy session, base & seed script
│       ├── models/                     # Relational database models
│       ├── schemas/                    # Pydantic request/response schemas
│       ├── services/                   # ML inference, grammar, and TTS services
│       └── api/v1/                     # REST API route endpoints
│
├── ai/                                 # Computer Vision & Machine Learning Pipelines
│   ├── feature_extraction.py           # 63D invariant feature normalizer
│   ├── hand_detection.py               # MediaPipe Hands reusable wrapper
│   ├── predict.py                      # Sub-millisecond inference engine
│   ├── train.py                        # Model training and benchmarking
│   └── dynamic/                        # Bi-LSTM sequence model for temporal gestures
│
├── models/                             # Trained Checkpoints & Serialized Encoders
│   ├── best/isl_classifier.pkl         # Champion 32-class MLP model
│   ├── dynamic/bilstm_dynamic.pt       # 3-class dynamic gesture Bi-LSTM
│   ├── label_encoder.pkl               # Serialized LabelEncoder
│   └── metadata/                       # Evaluated metrics JSONs
│
├── tests/                              # Automated Test Suite (96 Tests)
│   ├── test_backend_api.py             # FastAPI REST endpoints & auth tests
│   ├── test_week1.py                   # Camera, landmark, and invariance tests
│   ├── test_week2.py                   # Dataset and feature extraction tests
│   ├── test_week3.py                   # Model training and prediction tests
│   ├── test_dynamic_model.py           # Bi-LSTM dynamic sequence tests
│   ├── test_grammar.py                 # ISL grammar reconstruction tests
│   └── test_tts.py                     # Text-to-speech non-blocking tests
│
└── docs/                               # Comprehensive Technical Documentation
    ├── ARCHITECTURE.md                 # Complete system design
    ├── API.md                          # REST API specification & schemas
    ├── DATABASE.md                     # Database ER diagram & table definitions
    ├── ML_PIPELINE.md                  # Invariant mathematics & inference pipeline
    ├── DATASET_GUIDE.md                # Dataset acquisition & collection guide
    ├── MODEL_EVALUATION.md             # Ground-truth test metrics & confusion analysis
    ├── SECURITY_AND_PRIVACY.md         # Cryptography & GDPR privacy erasure
    ├── DEPLOYMENT.md                   # Local startup & production deployment
    ├── TROUBLESHOOTING.md              # Troubleshooting guide & common fixes
    └── PROJECT_REPORT.md               # Final engineering capstone report
```

---

## ⚡ Quick Start & Installation

### 1. Set Up Python Backend

```bash
# Clone the repository
git clone https://github.com/Jeel-Pipaliya/SignBridge-AI.git
cd SignBridge-AI

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy alembic bcrypt pyjwt python-multipart httpx pytest email-validator

# Run database migrations and seed default data
alembic upgrade head
python -m backend.app.db.seed

# Start backend server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
* Backend API: `http://127.0.0.1:8000`
* Swagger Docs: `http://127.0.0.1:8000/docs`

### 2. Set Up React Frontend

```bash
cd frontend

# Install packages
npm install

# Start Vite dev server
npm run dev
```
* Frontend Web App: `http://localhost:5173`

### 3. Pre-Seeded Demo Account
* **Email:** `demo@signbridge.ai`
* **Password:** `SignBridge@2026`
*(A "Quick-Fill Demo Account" button is also provided on the sign-in page).*

---

## 🧪 Automated Testing Suite (96 Tests)

Run the entire test suite covering machine learning mathematical invariance, sequence models, text-to-speech queues, grammar engines, and backend API security:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Results:**
```text
Ran 96 tests in 2.859s
OK
Feature Extraction Mathematical Invariance:
  [1] Translation Invariance: PASSED (Max Diff: 3.58e-07)
  [2] Scale Invariance:       PASSED (Max Diff: 1.19e-07)
  [3] Combined Invariance:    PASSED (Max Diff: 6.22e-07)
```

---

## 🐳 Docker Compose Production Deployment

Launch the full production stack (PostgreSQL, FastAPI Backend, and Nginx-served React Frontend):

```bash
docker-compose up -d --build
```
* Web Application: `http://localhost`
* Backend API: `http://localhost:8000/docs`

---

## 📊 Measured Evaluation Results (Held-Out Test Set)

* **Static Classifier:** Multi-Layer Perceptron Neural Network (32 classes: `1`–`5`, `A`–`Z`, `HELLO`)
* **Held-Out Test Set Accuracy:** **66.07%**
* **Validation Accuracy:** **69.05%**
* **Test Macro F1-Score:** **0.6085**
* **Dynamic Classifier:** Bi-LSTM (`HELLO`, `J`, `Z`) Test Accuracy: **100.0%**
* **Inference Latency:** **< 15 ms** on standard consumer CPU

---

## 📜 License & Acknowledgements

Open-source under the **MIT License**. Built with dedication to support the Indian Deaf & Hard-of-Hearing Community.
