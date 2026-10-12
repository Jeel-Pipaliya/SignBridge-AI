# SignBridge AI — Indian Sign Language Dataset Acquisition & Preparation Guide

## 1. Verified ISL Research Datasets & Licensing

Indian Sign Language (ISL) differs fundamentally in vocabulary, syntax, and two-handed finger-spelling conventions from American Sign Language (ASL). SignBridge AI explicitly benchmarks against verified ISL sources:

| Dataset Identifier | Author / Institution | Classes | Modality | Licensing & Access |
|---|---|---|---|---|
| **INCLUDE** | Sridhar et al. (ACM MM 2020) | 263 signs | RGB Video | Academic Research Use |
| **ISL-CSLTR** | IIT Guwahati / MeitY | 100+ signs | Multi-Camera Video | Research & Education |
| **ISLRTC National Corpus** | ISLRTC New Delhi | Standard ISL | Dictionary & Video | Public Reference Corpus |
| **SignBridge-ISL-32** | SignBridge AI Benchmark | 32 classes | 63D Normalized Landmarks | Open Source (MIT) |

---

## 2. Dataset Split & Anti-Leakage Methodology

To prevent data leakage across subjects:
- **Stratified Split**: 70% Training (`data/processed/train.csv`), 15% Validation (`data/processed/val.csv`), 15% Held-Out Test (`data/processed/test.csv`).
- **Class Balancing**: Stratified sampling ensures equitable representation across all 32 classes.
- **Verification Scripts**:
  ```bash
  # Check distribution statistics and class sample counts
  python -m ai.data_analysis.dataset_statistics

  # Validate coordinate integrity, NaN absence, and schema correctness
  python -m ai.data_analysis.validate_dataset
  ```

---

## 3. Data Collection Tool (Custom Sign Acquisition)

To collect custom sign samples using your local webcam:
```bash
python data/collect_samples.py
```
**Controls**:
- `[SPACE]`: Capture single sample.
- `[B]`: Burst capture mode (10 consecutive frames).
- `[N]`: Prompt for a new sign label.
- `[Q]`: Save and exit.
Output is serialized into `data/raw/collected_landmarks.csv`.
