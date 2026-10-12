# SignBridge AI — REST API Documentation

Base URL: `/api/v1`  
Interactive Swagger UI: `http://localhost:8000/docs`  
ReDoc Documentation: `http://localhost:8000/redoc`

---

## 1. Authentication Endpoints

### `POST /auth/register`
Creates a new user account and returns a signed JWT access token.
- **Request Body**:
  ```json
  {
    "email": "student@institution.edu",
    "password": "SecurePassword123",
    "full_name": "Aarav Sharma"
  }
  ```
- **Response `201 Created`**:
  ```json
  {
    "access_token": "eyJhbGciOi...",
    "token_type": "bearer",
    "user": {
      "id": 1,
      "email": "student@institution.edu",
      "full_name": "Aarav Sharma",
      "is_active": true,
      "is_superuser": false,
      "created_at": "2026-10-12T12:00:00Z"
    }
  }
  ```

### `POST /auth/login`
Authenticates existing credentials.
- **Request Body**:
  ```json
  {
    "email": "demo@signbridge.ai",
    "password": "SignBridge@2026"
  }
  ```
- **Response `200 OK`**: Returns access token and user metadata.

### `GET /auth/me`
Retrieves currently authenticated user's profile.
- **Headers**: `Authorization: Bearer <token>`

---

## 2. Real-Time Recognition & Inference

### `POST /recognition/predict-landmarks`
Runs high-speed static sign recognition on 21 3D hand joints or a flat 63-element coordinate vector.
- **Request Body**:
  ```json
  {
    "landmarks": [
      { "x": 0.51, "y": 0.62, "z": 0.01 },
      ...
    ],
    "handedness": "Right",
    "confidence_threshold": 0.70
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "predicted_sign": "A",
    "confidence": 0.9421,
    "is_recognized": true,
    "is_supported_vocabulary": true,
    "model_name": "SignBridge_ISL_MLP_NeuralNet",
    "model_version": "1.0",
    "top_candidates": [
      { "sign": "A", "confidence": 0.9421 },
      { "sign": "S", "confidence": 0.0315 },
      { "sign": "T", "confidence": 0.0120 }
    ],
    "inference_duration_ms": 3.82,
    "handedness": "Right"
  }
  ```

### `POST /recognition/predict-sequence`
Runs temporal Bi-LSTM inference on a 30-frame sequence buffer for dynamic gestures (`HELLO`, `J`, `Z`).
- **Request Body**:
  ```json
  {
    "sequence": [
      [ {"x": 0.5, "y": 0.6, "z": 0.0}, ... ], // Frame 0
      ... // Frames 1-29
    ],
    "confidence_threshold": 0.75
  }
  ```

### `POST /recognition/grammar-correct`
Reconstructs ordered sign tokens into natural English and translates into Hindi.
- **Request Body**:
  ```json
  {
    "tokens": ["YOU", "NAME", "WHAT"],
    "target_language": "hi"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "raw_tokens": ["YOU", "NAME", "WHAT"],
    "raw_sentence": "YOU NAME WHAT",
    "corrected_sentence": "What is your name?",
    "language": "hi",
    "translated_sentence": "आपका नाम क्या है?"
  }
  ```

---

## 3. Educational Tutor & Practice Sessions

### `GET /vocabulary`
Lists standard ISL vocabulary.
- **Query Parameters**:
  - `category`: `alphabet`, `number`, `greeting`, `phrase`
  - `gesture_type`: `static` or `dynamic`
  - `search`: Keyword string

### `POST /practice/sessions`
Logs user practice trial.
- **Headers**: `Authorization: Bearer <token>`
- **Request Body**:
  ```json
  {
    "target_sign": "L",
    "predicted_sign": "L",
    "confidence": 0.95,
    "accuracy_score": 0.96,
    "is_correct": true,
    "notes": "Good perpendicular angle"
  }
  ```

### `GET /practice/summary`
Returns user's streak, total sessions, and success rate.

---

## 4. Analytics & Diagnostics

### `GET /analytics/overview`
Provides comprehensive telemetry contrasting live user practice with held-out offline test benchmarks.

### `DELETE /history/clear`
GDPR privacy erasure endpoint purging stored recognition events and practice history.
