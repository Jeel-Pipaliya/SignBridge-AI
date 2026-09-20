# Indian Sign Language (ISL) Dataset Research (Week 2 Day 1)

This document provides a comprehensive research assessment of publicly available, academic, and research-grade **Indian Sign Language (ISL)** datasets. All links, DOIs, licenses, and class structures have been independently verified.

---

## 1. Comparative Dataset Matrix

| Dataset Name | Source / Institution | Verified URL / DOI | License | Classes | Sample Count & Type | Advantages | Limitations |
|---|---|---|---|---|---|---|---|
| **ISL Hand Gesture Dataset** *(Recommended for Baseline)* | Mendeley Data / Sougatamoy Biswas (2024) | [DOI: 10.17632/n34wm8sb3x.1](https://doi.org/10.17632/n34wm8sb3x.1) | CC BY 4.0 | 26 classes (A–Z) | 14,300 RGB images (550 images/class) | • Perfectly balanced<br>• High-resolution RGB<br>• Verified peer-reviewed DOI | • Static single-handed alphabet only<br>• Does not include whole-word phrases |
| **Dataset for Everyday Phrases and Words in ISL** | Mendeley Data / Manisha et al. | [DOI: 10.17632/w7fgy7jvs8.3](https://doi.org/10.17632/w7fgy7jvs8.3) | CC BY 4.0 | 44 everyday phrases/words | ~4,400 RGB images | • Real conversational words (Hello, Thanks, etc.)<br>• Standard webcam capture | • Background variations across classes<br>• Some two-handed signs require full upper body |
| **AI4Bharat INCLUDE** | IIT Madras / ACM Multimedia 2020 | [GitHub: AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) | MIT License | 263 word classes (15 categories) | 4,284 videos (recorded by native signers) | • Gold standard academic ISL dataset<br>• Native Deaf community signers | • Video sequences requiring temporal models (LSTM/Transformer)<br>• Heavy computational overhead for Week 3 |
| **ISL-CSLTR (Continuous Sign Language)** | Mendeley Data (2023) | [DOI: 10.17632/kcmpdxky7p.1](https://doi.org/10.17632/kcmpdxky7p.1) | CC BY 4.0 | 100+ continuous sentences | Multi-signer video corpus | • Full sentence translation benchmark | • Dynamic continuous signs require alignment models (CTC/Seq2Seq) |
| **Indian Sign Language Gesture Dataset** | Kaggle / Prathamesh Pawar | [Kaggle Dataset](https://www.kaggle.com/datasets/prathamesh36/indian-sign-language-dataset) | Open Database License (ODbL) | 35 classes (1–9, A–Z) | ~42,000 images (1,200/class) | • Large volume per class<br>• High variety of lighting | • Community collected<br>• Some class imbalance and noise |

---

## 2. Detailed Dataset Reviews

### 1. ISL Hand Gesture Dataset (Mendeley Data, 2024)
* **Author:** Sougatamoy Biswas
* **Publication Date:** May 15, 2024
* **Identifier:** `10.17632/n34wm8sb3x.1`
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Characteristics:** 26 classes representing the 26 Indian Sign Language manual alphabets. Each class contains exactly 550 images captured with an RGB camera under consistent indoor lighting.
* **Suitability:** **Highest for Week 2–3 Baseline**. The balanced distribution eliminates class weighting issues in Random Forest/SVM classifiers.

### 2. Dataset for Everyday Phrases and Words in Indian Sign Language (Mendeley Data)
* **Identifier:** `10.17632/w7fgy7jvs8.3`
* **License:** CC BY 4.0
* **Characteristics:** Covers common words like `HELLO`, `PLEASE`, `THANK YOU`, `WATER`, `FOOD`, `HELP`.
* **Suitability:** **Ideal for Real-Time Word Demonstration**. Combines basic conversational phrases with static single-handed or clear two-handed postures.

### 3. AI4Bharat INCLUDE Dataset
* **Authors:** Prem Selvaraj, Gokul NC, Pratyush Kumar, Mitesh Khapra (IIT Madras)
* **Publication:** ACM Multimedia 2020
* **Characteristics:** 4,284 videos spanning 263 classes across 15 semantic domains. All signs performed by native Deaf individuals.
* **Suitability:** **Target for Future Phase (Weeks 4+)**. Because it consists of continuous and dynamic videos, it requires sequence models (CNN-LSTM or Transformer).

---

## 3. Dataset Download & Ingestion Guide

To download the primary Mendeley dataset:
1. Visit [https://doi.org/10.17632/n34wm8sb3x.1](https://doi.org/10.17632/n34wm8sb3x.1).
2. Download the zip archive (`ISL_Dataset.zip`).
3. Extract the contents into `data/raw/` preserving class folder names:
   ```text
   data/raw/
   ├── A/
   ├── B/
   ├── C/
   ...
   └── Z/
   ```
4. Run our automated preprocessing script:
   ```bash
   python ai/preprocessing.py
   ```
