"""
SignBridge AI - Database Seed Script
Populates initial vocabulary, model metadata, and a demo account with realistic sample data.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.session import SessionLocal, engine
from backend.app.db.base import Base
from backend.app.models.user import User, UserPreference
from backend.app.models.vocabulary import VocabularySign
from backend.app.models.model_version import ModelVersion
from backend.app.models.practice import PracticeSession
from backend.app.models.recognition import RecognitionEvent
from backend.app.core.security import get_password_hash

# 32 Standard Indian Sign Language Vocabulary Definitions
ISL_VOCABULARY = [
    # Numbers
    {"sign_name": "1", "category": "number", "gesture_type": "static", "difficulty": "beginner",
     "description": "Single index finger extended vertically from closed fist.",
     "anatomical_tips": "Index pointing straight up; thumb locks folded middle, ring, and pinky fingers."},
    {"sign_name": "2", "category": "number", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index and middle fingers extended in a V formation.",
     "anatomical_tips": "Keep index and middle fingers separated; thumb holds ring and little finger."},
    {"sign_name": "3", "category": "number", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index, middle, and ring fingers extended upward.",
     "anatomical_tips": "Spread three middle fingers evenly; thumb secures pinky across the palm."},
    {"sign_name": "4", "category": "number", "gesture_type": "static", "difficulty": "beginner",
     "description": "Four fingers (index, middle, ring, pinky) extended upward; thumb folded across palm.",
     "anatomical_tips": "Keep all four fingers upright and separated; fold thumb firmly into palm."},
    {"sign_name": "5", "category": "number", "gesture_type": "static", "difficulty": "beginner",
     "description": "Open hand with all five fingers extended and separated.",
     "anatomical_tips": "Full palm extension; spread all digits outward facing the camera."},

    # Alphabet
    {"sign_name": "A", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Closed fist with thumb resting alongside the index finger.",
     "anatomical_tips": "Curl fingers tight into palm; thumb straight alongside the outer index knuckle."},
    {"sign_name": "B", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Flat palm with four fingers extended together and thumb folded across palm.",
     "anatomical_tips": "Keep 4 fingers straight and touching; thumb bent flat against the palm."},
    {"sign_name": "C", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Curved hand forming a C shape with fingers and thumb.",
     "anatomical_tips": "Curve fingers and thumb as if grasping a cup; palm faces sideways."},
    {"sign_name": "D", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index finger extended vertically; thumb touches curved middle, ring, and pinky tips.",
     "anatomical_tips": "Form a closed loop with thumb and other fingers; index points straight up."},
    {"sign_name": "E", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Fingers curled inward with fingertips touching the folded thumb.",
     "anatomical_tips": "Bend all 4 fingertips down; tuck thumb horizontally underneath fingertips."},
    {"sign_name": "F", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Index finger and thumb form a circle; remaining three fingers extended upward.",
     "anatomical_tips": "Tips of index and thumb touch; middle, ring, and pinky fan upwards."},
    {"sign_name": "G", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Index finger and thumb extended horizontally parallel to each other.",
     "anatomical_tips": "Point index finger horizontally to the side; thumb parallel pointing sideways."},
    {"sign_name": "H", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Index and middle fingers extended horizontally together.",
     "anatomical_tips": "Both fingers parallel and pointing sideways; thumb tucked against ring finger."},
    {"sign_name": "I", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Pinky finger extended vertically from closed fist.",
     "anatomical_tips": "Pinky stands straight up; thumb curls over the folded fingers."},
    {"sign_name": "J", "category": "alphabet", "gesture_type": "dynamic", "difficulty": "intermediate",
     "description": "Pinky extended tracing a downward J curve in the air.",
     "anatomical_tips": "Extend pinky, dip downward and curve up smoothly in a J trajectory."},
    {"sign_name": "K", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Index finger upright; middle finger pointing forward; thumb resting between them.",
     "anatomical_tips": "Form a V with index and middle; tuck thumb tip into middle finger base."},
    {"sign_name": "L", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index finger pointing up and thumb pointing out at a 90-degree angle (L shape).",
     "anatomical_tips": "Create a sharp right angle between index finger and thumb; palm faces forward."},
    {"sign_name": "M", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Thumb tucked under the first three curled fingers.",
     "anatomical_tips": "Drape index, middle, and ring fingers over the horizontal thumb."},
    {"sign_name": "N", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Thumb tucked under the first two curled fingers (index and middle).",
     "anatomical_tips": "Drape index and middle fingers over the thumb tip."},
    {"sign_name": "O", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Fingers and thumb curved touching tips to form an O shape.",
     "anatomical_tips": "Bring all 4 fingertips to meet the thumb tip, creating an open oval."},
    {"sign_name": "P", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "K shape inverted pointing downward.",
     "anatomical_tips": "Index points downward; middle points out; thumb rests on joint."},
    {"sign_name": "Q", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "G shape pointing downward.",
     "anatomical_tips": "Index finger and thumb point down with slight separation."},
    {"sign_name": "R", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index and middle fingers crossed vertically.",
     "anatomical_tips": "Cross middle finger over the back of the upright index finger."},
    {"sign_name": "S", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Fist with thumb wrapped across the front of the curled fingers.",
     "anatomical_tips": "Curl all four fingers and cross thumb over the knuckles."},
    {"sign_name": "T", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Thumb tucked between index and middle fingers in a fist.",
     "anatomical_tips": "Thumb tip emerges between index and middle knuckles."},
    {"sign_name": "U", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index and middle fingers extended straight up and held closely together.",
     "anatomical_tips": "Keep index and middle fingers touching vertically; thumb holds other fingers."},
    {"sign_name": "V", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index and middle fingers extended straight up in an open V shape.",
     "anatomical_tips": "Separate index and middle into a distinct V angle; palm faces front."},
    {"sign_name": "W", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Index, middle, and ring fingers extended upward in a W shape.",
     "anatomical_tips": "Spread three central digits; thumb pins the pinky tip."},
    {"sign_name": "X", "category": "alphabet", "gesture_type": "static", "difficulty": "intermediate",
     "description": "Index finger bent into a hook from a closed fist.",
     "anatomical_tips": "Curl index into a prominent hook shape; keep other fingers folded."},
    {"sign_name": "Y", "category": "alphabet", "gesture_type": "static", "difficulty": "beginner",
     "description": "Thumb and pinky finger extended outward (shaka shape); central three fingers curled.",
     "anatomical_tips": "Point thumb and pinky far apart; palm faces user or forward."},
    {"sign_name": "Z", "category": "alphabet", "gesture_type": "dynamic", "difficulty": "intermediate",
     "description": "Index finger traces a Z trajectory in the air.",
     "anatomical_tips": "Extend index finger and draw horizontal right, diagonal down-left, horizontal right."},

    # Dynamic Greetings / Common Phrases
    {"sign_name": "HELLO", "category": "greeting", "gesture_type": "dynamic", "difficulty": "beginner",
     "description": "Open hand raised from temple/forehead outward in a friendly salute gesture.",
     "anatomical_tips": "Start near forehead with open palm, sweep gently outward toward interlocutor."},
]


def seed_database():
    """Initializes tables and populates seed data."""
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. Seed Demo User
        demo_email = "demo@signbridge.ai"
        user = db.query(User).filter(User.email == demo_email).first()
        if not user:
            print(f"Creating demo user: {demo_email}")
            user = User(
                email=demo_email,
                full_name="Aarav Sharma (Demo Student)",
                hashed_password=get_password_hash("SignBridge@2026"),
                is_active=True,
                is_superuser=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            pref = UserPreference(
                user_id=user.id,
                theme="dark",
                font_size="normal",
                high_contrast=False,
                reduced_motion=False,
                tts_speed=1.0,
                tts_volume=1.0,
                preferred_language="en",
                auto_speak=True,
                store_history=True,
            )
            db.add(pref)
            db.commit()
            print("Demo user created successfully.")
        else:
            print(f"Demo user {demo_email} already exists.")

        # 2. Seed Vocabulary Signs
        print("Seeding Indian Sign Language Vocabulary (32 signs)...")
        seeded_count = 0
        for item in ISL_VOCABULARY:
            existing = db.query(VocabularySign).filter(VocabularySign.sign_name == item["sign_name"]).first()
            if not existing:
                sign = VocabularySign(
                    sign_name=item["sign_name"],
                    category=item["category"],
                    description=item["description"],
                    gesture_type=item["gesture_type"],
                    difficulty=item["difficulty"],
                    anatomical_tips=item["anatomical_tips"],
                )
                db.add(sign)
                seeded_count += 1
        db.commit()
        print(f"Seeded {seeded_count} new vocabulary signs.")

        # 3. Seed Model Versions
        print("Seeding Model Version telemetry...")
        existing_model = db.query(ModelVersion).filter(ModelVersion.version == "1.0").first()
        if not existing_model:
            mlp_model = ModelVersion(
                model_name="SignBridge_ISL_MLP_NeuralNet",
                version="1.0",
                architecture="MLP",
                num_classes=32,
                confidence_threshold=0.70,
                train_accuracy=0.7308,
                val_accuracy=0.6905,
                test_accuracy=0.6607,
                test_macro_f1=0.6085,
                is_active=True,
            )
            bilstm_model = ModelVersion(
                model_name="SignBridge_Dynamic_BiLSTM",
                version="1.0",
                architecture="BiLSTM",
                num_classes=3,
                confidence_threshold=0.75,
                train_accuracy=1.0,
                val_accuracy=1.0,
                test_accuracy=1.0,
                test_macro_f1=1.0,
                is_active=True,
            )
            db.add(mlp_model)
            db.add(bilstm_model)
            db.commit()
            print("Seeded active model versions.")

        # 4. Seed Sample Practice Sessions for Demo User
        if user:
            existing_sessions = db.query(PracticeSession).filter(PracticeSession.user_id == user.id).count()
            if existing_sessions == 0:
                print("Seeding sample practice history for demo user...")
                sample_practices = [
                    ("A", "A", 0.94, 0.95, True, "Good thumb placement alongside index knuckle."),
                    ("B", "B", 0.91, 0.92, True, "Four fingers straight and thumb tucked."),
                    ("C", "C", 0.88, 0.90, True, "Nice circular arc palm curvature."),
                    ("1", "1", 0.96, 0.98, True, "Clear vertical index finger extension."),
                    ("L", "L", 0.95, 0.96, True, "Sharp 90-degree angle between thumb and index."),
                    ("Y", "Y", 0.92, 0.94, True, "Extended thumb and pinky clearly separated."),
                    ("V", "2", 0.72, 0.65, False, "Slight angle variation caused confusion with sign 2."),
                    ("HELLO", "HELLO", 0.92, 0.93, True, "Smooth outward motion trajectory recognized."),
                ]
                now = datetime.now(timezone.utc)
                for i, (tgt, pred, conf, acc, ok, notes) in enumerate(sample_practices):
                    s = PracticeSession(
                        user_id=user.id,
                        target_sign=tgt,
                        predicted_sign=pred,
                        confidence=conf,
                        accuracy_score=acc,
                        is_correct=ok,
                        notes=notes,
                        created_at=now - timedelta(hours=24 - (i * 2)),
                    )
                    db.add(s)

                    # Also seed recognition event
                    rec = RecognitionEvent(
                        user_id=user.id,
                        sign_detected=pred,
                        confidence=conf,
                        model_name="SignBridge_ISL_MLP_NeuralNet",
                        model_version="1.0",
                        inference_duration_ms=4.2,
                        was_corrected=not ok,
                        corrected_sign=tgt if not ok else None,
                        created_at=now - timedelta(hours=24 - (i * 2)),
                    )
                    db.add(rec)
                db.commit()
                print("Sample practice and recognition events seeded.")

        print("Database seeding completed successfully.")

    except Exception as err:
        db.rollback()
        print(f"Error seeding database: {err}")
        raise err
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
