# SignBridge AI — AI/ML Pipeline Specification

## 1. Mathematical Feature Extraction & Invariance Normalization

Raw MediaPipe Hands output produces 21 3D joint landmark coordinates:
$$\mathbf{L} = \{(x_i, y_i, z_i)\}_{i=0}^{20}$$

These raw coordinates are fundamentally sensitive to:
1. **Screen translation**: Moving closer to the camera edges shifts coordinates.
2. **Camera distance (Scale)**: Sitting closer enlarges the pixel coordinate span.

### Step 1: Translation Invariance (Wrist Centering)
Landmark index 0 ($\mathbf{L}_0$, the anatomical wrist) serves as the coordinate origin. All 21 joint positions are centered relative to the wrist:
$$\mathbf{L}'_i = \mathbf{L}_i - \mathbf{L}_0 \quad \forall i \in \{0, \dots, 20\}$$
Consequently, $\mathbf{L}'_0 = (0, 0, 0)$.

### Step 2: Scale Invariance (Middle MCP Normalization)
Landmark index 9 ($\mathbf{L}_9$) corresponds to the Metacarpophalangeal (MCP) joint at the base of the middle finger—a rigid skeletal reference dimension. We compute the Euclidean reference distance:
$$d_{\text{ref}} = \|\mathbf{L}'_9\|_2 = \sqrt{(x'_9)^2 + (y'_9)^2 + (z'_9)^2}$$
If $d_{\text{ref}} < \epsilon$ ($10^{-6}$), the maximum distance across all 21 joints is used as fallback. All centered coordinates are scaled:
$$\mathbf{L}''_i = \frac{\mathbf{L}'_i}{d_{\text{ref}}}$$

### Step 3: Flattening into 63-Dimensional Vector
$$\mathbf{f} = [x''_0, y''_0, z''_0, x''_1, y''_1, z''_1, \dots, x''_{20}, y''_{20}, z''_{20}] \in \mathbb{R}^{63}$$
Automated invariant unit tests confirm max differences below $10^{-6}$ across translations and scale variations.

---

## 2. Model Architectures

### 2.1 Static 32-Class ISL Classifier: Multi-Layer Perceptron (MLP)
- **Input Layer**: 63 features.
- **Hidden Layers**: Dense(128, ReLU) -> BatchNorm -> Dropout(0.3) -> Dense(64, ReLU) -> BatchNorm.
- **Output Layer**: Dense(32, Softmax).
- **Optimizer**: Adam ($\alpha=0.001$), Cross-Entropy Loss.
- **Artifacts**: `models/best/isl_classifier.pkl` and `models/label_encoder.pkl`.

### 2.2 Dynamic Sequence Model: Bidirectional LSTM
- **Input Dimension**: Sequence of 30 frames $\times$ 63 features: $(T=30, F=63)$.
- **Architecture**: Bi-LSTM (Hidden Dimension = 64, Layers = 2, Bidirectional = True, Output Dimension = 128).
- **Classification Head**: Dense(64, ReLU) -> Dropout(0.2) -> Dense(3, Softmax).
- **Target Classes**: `HELLO`, `J`, `Z`.
- **Artifact**: `models/dynamic/bilstm_dynamic.pt`.

---

## 3. Real-Time Inference & Smoothing

- **Sliding-Window Majority Vote**: Buffers the last 5 static predictions. Suppresses transient single-frame motion blur or landmark jitter.
- **Dynamic Energy Gating**: Calculates cumulative motion energy across 30 frames:
$$E = \sum_{t=1}^{29} \|\mathbf{f}_t - \mathbf{f}_{t-1}\|_2$$
If motion energy $E \ge 0.015$, the Bi-LSTM model is prioritized over the static classifier.
