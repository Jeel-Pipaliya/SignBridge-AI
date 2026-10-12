# SignBridge AI — Database Architecture & Schema Specification

## 1. Relational Entity-Relationship Diagram

```
+--------------------+            +-----------------------+
|       users        | 1        1 |   user_preferences    |
|--------------------|------------|-----------------------|
| id (PK)            |            | id (PK)               |
| email (Unique)     |            | user_id (FK -> users) |
| full_name          |            | theme                 |
| hashed_password    |            | font_size             |
| is_active          |            | high_contrast         |
| is_superuser       |            | reduced_motion        |
| created_at         |            | tts_speed             |
| updated_at         |            | tts_volume            |
+--------------------+            | preferred_language    |
         | 1                      | auto_speak            |
         |                        | store_history         |
         |                        +-----------------------+
         |
         +-------------+ 1..*
         |             |
         v             v
+------------------+  +--------------------+  +----------------------+
|practice_sessions |  | recognition_events |  |     transcripts      |
|------------------|  |--------------------|  |----------------------|
| id (PK)          |  | id (PK)            |  | id (PK)              |
| user_id (FK)     |  | user_id (FK, Null) |  | user_id (FK, Null)   |
| target_sign      |  | sign_detected      |  | title                |
| predicted_sign   |  | confidence         |  | content              |
| confidence       |  | model_name         |  | mode                 |
| accuracy_score   |  | model_version      |  | language             |
| is_correct       |  | duration_ms        |  | created_at           |
| notes            |  | was_corrected      |  +----------------------+
| created_at       |  | corrected_sign     |
+------------------+  | created_at         |
                      +--------------------+

+----------------------+       +-----------------------+
|   vocabulary_signs   |       |    model_versions     |
|----------------------|       |-----------------------|
| id (PK)              |       | id (PK)               |
| sign_name (Unique)   |       | model_name            |
| category             |       | version               |
| description          |       | architecture          |
| gesture_type         |       | num_classes           |
| difficulty           |       | confidence_threshold  |
| thumbnail_url        |       | train_accuracy        |
| anatomical_tips      |       | val_accuracy          |
| created_at           |       | test_accuracy         |
+----------------------+       | test_macro_f1         |
                               | is_active             |
                               | created_at            |
                               +-----------------------+
```

---

## 2. Table Specifications & Constraints

### 2.1 `users`
- `id`: Integer Primary Key.
- `email`: Varchar(255), Unique, Indexed, Non-null. Stored in lowercase.
- `hashed_password`: Varchar(255), Non-null. Salted bcrypt hash (rounds=12).
- `is_active`: Boolean, default `true`.
- `created_at` / `updated_at`: UTC timestamps.

### 2.2 `user_preferences`
- `user_id`: Foreign key referencing `users.id` with `ON DELETE CASCADE`. Unique constraint enforces 1-to-1 relationship.
- `store_history`: Boolean flag granting or revoking consent for logging recognition events.

### 2.3 `vocabulary_signs`
- Pre-seeded with 32 standard Indian Sign Language signs (`1`-`5`, `A`-`Z`, `HELLO`).
- Contains anatomical placement instructions and categorizations.

### 2.4 `practice_sessions`
- Stores individual student practice attempts with measured similarity scores (0.0 to 1.0) and success flags.

---

## 3. Migration & Seeding Commands

```bash
# Apply migrations to database head
alembic upgrade head

# Seed 32 ISL vocabulary items, active model telemetry, and demo user
python -m backend.app.db.seed
```
