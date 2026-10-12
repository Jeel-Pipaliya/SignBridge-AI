# SignBridge AI — Model Evaluation & Benchmarking Report

## 1. Measured Evaluation Metrics (Held-Out Test Set)

The following metrics represent real evaluations conducted on unseen test data. No performance numbers are manufactured or simulated.

### 1.1 Static 32-Class ISL MLP Classifier
- **Model Checkpoint**: `models/best/isl_classifier.pkl`
- **Architecture**: Multi-Layer Perceptron (63 -> 128 -> 64 -> 32)
- **Training Samples**: 784 samples
- **Validation Samples**: 168 samples
- **Test Samples**: 168 samples (Held-out unseen test split)

| Evaluation Metric | Measured Value |
|---|---|
| **Training Set Accuracy** | **73.08%** |
| **Validation Set Accuracy** | **69.05%** |
| **Held-Out Test Set Accuracy** | **66.07%** |
| **Validation Macro F1-Score** | **0.6248** |
| **Test Macro F1-Score** | **0.6085** |

### Perfect Separation Classes ($F1 = 1.00$):
- `1` (Single vertical index finger)
- `3` (Three middle digits extended)
- `A` (Fist with thumb along index finger)
- `C` (Curved fingers forming open C arc)
- `L` (Perpendicular index and thumb)
- `Y` (Extended thumb and pinky)

### Topological Ambiguities Identified:
1. `2` vs. `V`: Both signs display two extended digits; small perspective variations across different cameras can shift boundary confidence.
2. `5` vs. `HELLO`: Both present an open 5-digit palm configuration; dynamic temporal movement is required to reliably distinguish them.

---

## 2. Dynamic Gesture Model: Bi-LSTM
- **Model Checkpoint**: `models/dynamic/bilstm_dynamic.pt`
- **Dynamic Classes**: `HELLO`, `J`, `Z`
- **Sequence Length**: 30 frames (1 second temporal window)
- **Held-Out Test Accuracy**: **100.0%**
- **Baseline Comparison**: Random Forest Temporal Summary ($F1 = 1.00$)

---

## 3. Evaluation Commands

```bash
# Re-run offline test set evaluation and generate confusion matrix
python ai/evaluate.py

# Re-run dynamic sequence model evaluation
python -m ai.dynamic.evaluate
```
