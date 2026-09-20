# SignBridge AI — Prototype Vocabulary Specification (Week 2 Day 2)

To establish a solid, scientifically verifiable baseline, Week 2 and Week 3 focus on a **curated Prototype Vocabulary of 12 distinct Indian Sign Language (ISL) classes**.

---

## 1. Prototype Vocabulary vs. Full ISL Vocabulary

| Metric / Aspect | Week 3 Prototype Vocabulary | Full Indian Sign Language (ISL) |
|---|---|---|
| **Vocabulary Size** | **12 classes** | **Thousands of signs** |
| **Grammar Structure** | Isolated static hand shapes & gestures | Topic-Comment grammar, spatial indexing, non-manual markers |
| **Modality** | Single-handed static configurations | Two-handed configurations, head tilt, facial expressions, lip patterns |
| **Temporal Dynamic** | Frame-by-frame landmark classification with temporal voting | Continuous video sequences, dynamic transitions, co-articulation |
| **Model Requirements** | Lightweight ML Classifier (Random Forest / SVM) | Multi-modal Temporal Deep Learning (CNN-LSTM / Spatial-Temporal GCN / Transformer) |
| **Scope Claim** | Working baseline prototype for demonstration | Comprehensive natural language communication system |

> [!IMPORTANT]
> **Academic Integrity Note:** This 12-class prototype is designed to prove the feasibility of the end-to-end computer-vision and feature extraction pipeline. It **does not** constitute full ISL vocabulary coverage.

---

## 2. Selected 12 Prototype Classes

The 12 classes are chosen to minimize confusing geometric overlap while providing both numeric, alphabetical, and conversational utility:

| Class Index | Label | Category | Hand Configuration Description | Linguistic Meaning |
|:---:|:---:|:---:|---|---|
| **0** | `1` | Numeral | Index finger extended vertically; other fingers closed into palm. | Number 1 |
| **1** | `2` | Numeral | Index and middle fingers extended in a V-shape; others closed. | Number 2 |
| **2** | `3` | Numeral | Index, middle, and ring fingers extended; thumb holds pinky. | Number 3 |
| **3** | `4` | Numeral | Four fingers (Index, Middle, Ring, Pinky) extended vertically; thumb folded. | Number 4 |
| **4** | `5` | Numeral | All five digits fully extended and spread out. | Number 5 |
| **5** | `A` | Alphabet | Fist formed with thumb resting alongside index finger. | Letter 'A' |
| **6** | `B` | Alphabet | Flat hand with four fingers straight up and thumb tucked across palm. | Letter 'B' |
| **7** | `C` | Alphabet | Fingers and thumb curved into a 'C' shape. | Letter 'C' |
| **8** | `L` | Alphabet | Thumb and index finger extended at a 90-degree angle ('L' shape). | Letter 'L' |
| **9** | `V` | Alphabet | Index and middle fingers extended apart; thumb across ring/pinky. | Letter 'V' / Peace |
| **10** | `Y` | Alphabet | Thumb and pinky extended outward; middle three fingers folded into palm. | Letter 'Y' |
| **11** | `HELLO` | Word | Open palm facing forward, fingers upright and relaxed in greeting pose. | Greeting ("Hello") |

---

## 3. Class Selection Rationale

1. **Geometric Distinction**:
   Classes were selected with diverse topological features (e.g. thumb extension in `Y`, right-angle in `L`, curvature in `C`, single extension in `1`), reducing inter-class confusion in distance-based classifiers.
2. **Standard ISL Consistency**:
   All 12 signs align with standard single-handed ISL finger-spelling charts and numeric representations published by the Indian Sign Language Research and Training Centre (ISLRTC).
3. **Conversational Baseline**:
   Combining numerals (`1–5`), alphabets (`A, B, C, L, V, Y`), and common greetings (`HELLO`) allows realistic interactive testing of the real-time recognition and text accumulation engines.
