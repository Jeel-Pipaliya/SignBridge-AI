# SignBridge AI — Comprehensive Engineering Project Report
**Major Capstone Project in Artificial Intelligence & Machine Learning (B.Tech AIML)**  
**Project Lead:** Jeel Pipaliya  
**Platform:** SignBridge AI (Indian Sign Language Assistive Communication & Education Platform)

---

## 1. Abstract & Problem Statement

Over 63 million individuals in India suffer from significant auditory and speech impairments. While Indian Sign Language (ISL) serves as the primary mode of expression, a pervasive literacy deficit exists within the hearing public. Specialized hardware solutions such as sensory gloves impose cost, maintenance, and comfort burdens that hinder widespread adoption.

SignBridge AI resolves this communication divide by providing a vision-based, real-time ISL recognition and educational suite operating on standard consumer webcams without dedicated sensors. The system achieves sub-15ms inference latency, guarantees translation and scale invariance via mathematical wrist centering and MCP normalization, and integrates a dual AI architecture (static 32-class MLP + dynamic 3-class Bi-LSTM) with deterministic grammar reconstruction and bidirectional speech synthesis.

---

## 2. Mathematical Invariance Formulation

Given 21 raw joints $\mathbf{L} = \{(x_i, y_i, z_i)\}_{i=0}^{20}$:
1. **Translation Normalization**: $\mathbf{L}'_i = \mathbf{L}_i - \mathbf{L}_0$ ensures origin invariance $\mathbf{L}'_0 = (0, 0, 0)$.
2. **Scale Normalization**: $\mathbf{L}''_i = \mathbf{L}'_i / \|\mathbf{L}'_9\|_2$ eliminates distance sensitivity.
3. **Invariance Verification**: Unit tests verify max coordinate drift $< 6.22 \times 10^{-7}$ across arbitrary spatial translations and scale expansions.

---

## 3. Measured Empirical Results

### Static 32-Class ISL MLP:
- **Architecture**: Dense(128) -> BatchNorm -> Dropout(0.3) -> Dense(64) -> Dense(32)
- **Held-Out Test Set Accuracy**: **66.07%**
- **Validation Accuracy**: **69.05%**
- **Test Macro F1-Score**: **0.6085**
- **Inference Speed**: **3.8 ms** per frame

### Dynamic Bi-LSTM:
- **Temporal Classes**: `HELLO`, `J`, `Z`
- **Sequence Length**: 30 frames (1 second at 30 FPS)
- **Test Accuracy**: **100.0%**

---

## 4. Software Architecture & Deliverables

- **Frontend**: Full-featured React 18 / TypeScript / Tailwind CSS / Vite client meeting WCAG 2.1 AA accessibility standards.
- **Backend**: Modular FastAPI REST server with JWT authentication, relational ORM, and Alembic migrations.
- **Test Suite**: 96 automated tests covering computer vision invariance, sequence models, TTS queues, grammar correction, and API security contracts.
- **Containerization**: Multi-container Docker Compose with PostgreSQL 16, FastAPI backend, and Nginx reverse proxy.
