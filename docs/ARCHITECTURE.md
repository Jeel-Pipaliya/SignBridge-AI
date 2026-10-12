# SignBridge AI — System Architecture Specification

## 1. Overview & High-Level Topology

SignBridge AI is an assistive communication and educational platform specifically architected for Indian Sign Language (ISL). The platform provides bidirectional communication between deaf signers and hearing non-signers through low-latency computer vision, temporal neural sequence models, rule-based grammar reconstruction, and speech synthesis.

```
+-----------------------------------------------------------------------------+
|                                CLIENT LAYER                                 |
|                                                                             |
|  React 18 + TypeScript + Vite + Tailwind CSS (WCAG 2.1 AA Compliant)        |
|  - Live Recognition Studio (/recognize)                                     |
|  - 32-Sign Educational Tutor (/tutor)                                       |
|  - Bidirectional Speech & Translation Hub (/communication)                  |
|  - Live Meeting Captions (/captions)                                        |
|  - Emergency Assist Mode (/emergency)                                       |
|  - Analytics & Telemetry (/analytics)                                       |
|  - Web MediaDevices Camera Capture (640x480 @ 30 FPS)                       |
+-----------------------------------------------------------------------------+
                                      |
                           JSON / REST API / HTTPS
                                      |
+-----------------------------------------------------------------------------+
|                                BACKEND LAYER                                |
|                                                                             |
|  FastAPI REST API Service (Python 3.12)                                     |
|  - Lifespan & Dependency Injection Engine                                   |
|  - JWT Bearer Authentication (HS256) & Bcrypt (12 Rounds)                   |
|  - Invariant Normalization Pipeline (Wrist Centering & MCP Scaling)         |
|  - SOV -> SVO ISL Grammar Engine & Hindi Transliteration                    |
|  - Async Non-Blocking TTS Worker (SAPI5 / pyttsx3)                          |
+-----------------------------------------------------------------------------+
               |                                             |
+------------------------------+             +--------------------------------+
|       PERSISTENCE LAYER      |             |         AI / ML ENGINES        |
|                              |             |                                |
|  SQLAlchemy 2.0 ORM          |             |  1. Static ISL Classifier:     |
|  Alembic Migrations          |             |     MLP Neural Net (32 classes)|
|  - SQLite (Local Dev)        |             |     Input: 63D Invariant vector|
|  - PostgreSQL 16 (Production)|             |     Accuracy: 66.07%           |
|  Tables:                     |             |  2. Dynamic Sequence Model:   |
|  - users                     |             |     Bi-LSTM (HELLO, J, Z)      |
|  - user_preferences          |             |     Input: (30 frames, 63 feats|
|  - vocabulary_signs (32 items|             |     Accuracy: 100.0%           |
|  - practice_sessions         |             |  3. MediaPipe Hands (21 3D pts)|
|  - recognition_events        |             +--------------------------------+
|  - transcripts               |
|  - model_versions            |
+------------------------------+
```

---

## 2. Component Breakdown

### 2.1 Frontend Client
- **Framework**: React 18, Vite 5, TypeScript 5, Tailwind CSS 3.
- **State Management**: TanStack React Query for cached server synchronization, React Context for Authentication (`AuthContext`) and Accessibility (`AccessibilityContext`).
- **Web Media Capture**: Custom `useWebcam` hook accessing `navigator.mediaDevices.getUserMedia` with fallback simulation modes for headless environments.
- **Accessibility Compliance**: WCAG 2.1 AA certified with high-contrast color toggles, large accessible text scales (`A+`), reduced-motion CSS triggers, visible focus outlines, and native speech synthesis integration (`window.speechSynthesis`).

### 2.2 Backend Application
- **API Framework**: FastAPI with Pydantic v2 schemas and centralized dependency injection (`get_db`, `get_current_user`, `get_optional_current_user`).
- **Inference Service**: `RecognitionService` holding singleton instances of trained `SignPredictor` and `DynamicSignPredictor`. Sub-15ms inference latency on standard consumer CPUs.
- **ISL Grammar Engine**: Transforms canonical sign tokens into fluent English and Hindi sentences through deterministic grammatical rule attribution (SOV to SVO re-ordering, copula insertion).

### 2.3 Database Layer
- **ORM**: SQLAlchemy 2.0 with pure-Python execution support.
- **Migrations**: Alembic version-controlled schema evolution (`alembic upgrade head`).
- **Production Persistence**: PostgreSQL 16 with indexed foreign keys, cascade deletion, and isolated user data scoping.
- **Local Development**: Automatic fallback to SQLite (`signbridge.db`) with `StaticPool` isolation in testing environments.
