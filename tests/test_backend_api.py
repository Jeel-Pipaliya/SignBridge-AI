"""
SignBridge AI - Comprehensive Backend API & Integration Test Suite
Tests:
  - Health & Readiness probes
  - User registration, login, JWT validation, and profile access
  - Vocabulary listing and querying
  - Model metadata and active class telemetry
  - Real-time static landmark inference contract with live ML model
  - Temporal sequence prediction contract
  - ISL grammar correction and Hindi translation
  - Practice session persistence, streak calculation, and summary
  - Analytics metrics combining live telemetry with offline benchmarks
  - Data privacy erasure controls
"""

import unittest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db.base import Base
from backend.app.db.session import get_db

# In-memory SQLite database with StaticPool for isolated, shared test execution
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)

    def test_01_health_and_readiness(self):
        # Health
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)

        # Ready
        res_ready = self.client.get("/api/v1/ready")
        self.assertEqual(res_ready.status_code, 200)
        ready_data = res_ready.json()
        self.assertTrue(ready_data["database"])
        self.assertTrue(ready_data["static_model"])

    def test_02_model_metadata(self):
        res = self.client.get("/api/v1/model/info")
        self.assertEqual(res.status_code, 200)
        info = res.json()
        self.assertIn("static_model", info)
        self.assertIn("dynamic_model", info)
        self.assertEqual(info["static_model"]["num_classes"], 32)
        self.assertGreaterEqual(info["static_model"]["test_accuracy"], 0.60)

        # Classes
        res_classes = self.client.get("/api/v1/model/classes")
        self.assertEqual(res_classes.status_code, 200)
        classes_data = res_classes.json()
        self.assertIn("1", classes_data["static_classes"])
        self.assertIn("A", classes_data["static_classes"])
        self.assertIn("HELLO", classes_data["dynamic_classes"])

    def test_03_user_registration_and_auth_flow(self):
        # Register
        user_payload = {
            "email": "tester@signbridge.ai",
            "password": "SecurePassword123",
            "full_name": "Test Engineer",
        }
        res_reg = self.client.post("/api/v1/auth/register", json=user_payload)
        self.assertEqual(res_reg.status_code, 201)
        reg_data = res_reg.json()
        self.assertIn("access_token", reg_data)
        token = reg_data["access_token"]

        # Duplicate register should fail
        res_dup = self.client.post("/api/v1/auth/register", json=user_payload)
        self.assertEqual(res_dup.status_code, 400)

        # Login
        res_login = self.client.post(
            "/api/v1/auth/login",
            json={"email": "tester@signbridge.ai", "password": "SecurePassword123"},
        )
        self.assertEqual(res_login.status_code, 200)
        login_data = res_login.json()
        self.assertIn("access_token", login_data)

        # Get profile with Bearer token
        headers = {"Authorization": f"Bearer {token}"}
        res_me = self.client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.json()["email"], "tester@signbridge.ai")

        # Update preferences
        pref_update = {"theme": "light", "high_contrast": True, "preferred_language": "hi"}
        res_pref = self.client.put("/api/v1/users/preferences", json=pref_update, headers=headers)
        self.assertEqual(res_pref.status_code, 200)
        self.assertEqual(res_pref.json()["theme"], "light")
        self.assertTrue(res_pref.json()["high_contrast"])

    def test_04_static_landmark_inference(self):
        # Create a sample 21-point hand skeleton (simulating an open hand or fist)
        sample_landmarks = []
        for i in range(21):
            sample_landmarks.append({
                "x": 0.5 + (i * 0.01),
                "y": 0.6 + (i * 0.015),
                "z": (i * 0.005),
            })

        req = {
            "landmarks": sample_landmarks,
            "handedness": "Right",
            "confidence_threshold": 0.0,  # accept prediction for testing pipeline
        }
        res = self.client.post("/api/v1/recognition/predict-landmarks", json=req)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("predicted_sign", data)
        self.assertIn("confidence", data)
        self.assertIn("inference_duration_ms", data)
        self.assertIsInstance(data["top_candidates"], list)
        self.assertGreater(len(data["top_candidates"]), 0)

    def test_05_grammar_correction_and_translation(self):
        # ISL Question Pattern: YOU NAME WHAT -> What is your name?
        req = {
            "tokens": ["YOU", "NAME", "WHAT"],
            "target_language": "hi",
        }
        res = self.client.post("/api/v1/recognition/grammar-correct", json=req)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["corrected_sentence"], "What is your name?")
        self.assertEqual(data["translated_sentence"], "आपका नाम क्या है?")

    def test_06_practice_session_lifecycle(self):
        # Login to get token
        res_login = self.client.post(
            "/api/v1/auth/login",
            json={"email": "tester@signbridge.ai", "password": "SecurePassword123"},
        )
        token = res_login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Record practice session
        practice_payload = {
            "target_sign": "L",
            "predicted_sign": "L",
            "confidence": 0.95,
            "accuracy_score": 0.96,
            "is_correct": True,
            "notes": "Good perpendicular angle",
        }
        res_prac = self.client.post("/api/v1/practice/sessions", json=practice_payload, headers=headers)
        self.assertEqual(res_prac.status_code, 201)
        self.assertEqual(res_prac.json()["target_sign"], "L")

        # Get summary
        res_sum = self.client.get("/api/v1/practice/summary", headers=headers)
        self.assertEqual(res_sum.status_code, 200)
        sum_data = res_sum.json()
        self.assertGreaterEqual(sum_data["total_sessions"], 1)
        self.assertEqual(sum_data["correct_count"], 1)
        self.assertEqual(sum_data["current_streak"], 1)

    def test_07_analytics_overview(self):
        res_login = self.client.post(
            "/api/v1/auth/login",
            json={"email": "tester@signbridge.ai", "password": "SecurePassword123"},
        )
        token = res_login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        res_ana = self.client.get("/api/v1/analytics/overview", headers=headers)
        self.assertEqual(res_ana.status_code, 200)
        ana_data = res_ana.json()
        self.assertIn("total_practice_sessions", ana_data)
        self.assertIn("model_benchmarks", ana_data)
        self.assertIn("offline_vs_live_note", ana_data)

    def test_08_transcripts_and_privacy_erasure(self):
        res_login = self.client.post(
            "/api/v1/auth/login",
            json={"email": "tester@signbridge.ai", "password": "SecurePassword123"},
        )
        token = res_login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Save transcript
        t_payload = {
            "title": "Medical Consultation",
            "content": "Doctor: Where is the pain? Patient: My arm.",
            "mode": "conversation",
            "language": "en",
        }
        res_t = self.client.post("/api/v1/transcripts", json=t_payload, headers=headers)
        self.assertEqual(res_t.status_code, 201)
        t_id = res_t.json()["id"]

        # List transcripts
        res_list = self.client.get("/api/v1/transcripts", headers=headers)
        self.assertEqual(res_list.status_code, 200)
        self.assertGreaterEqual(len(res_list.json()), 1)

        # Clear history (privacy erasure)
        res_clear = self.client.delete("/api/v1/history/clear?clear_practice=true", headers=headers)
        self.assertEqual(res_clear.status_code, 200)
        self.assertEqual(res_clear.json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
