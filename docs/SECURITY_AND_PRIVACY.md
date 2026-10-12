# SignBridge AI — Security & Privacy Architecture

## 1. Authentication & Cryptography
- **Password Hashing**: Salted `bcrypt` with 12 computational rounds. Plaintext passwords are never logged, persisted, or leaked.
- **JWT Tokens**: Signed using `HS256` with configurable 24-hour expiration (`ACCESS_TOKEN_EXPIRE_MINUTES=1440`).
- **Subject Scoping**: All user-specific queries (`/users/me`, `/practice/sessions`, `/history/recognition`) enforce server-side validation against the authenticated `sub` ID.

---

## 2. Privacy-First Biometric Media Handling
- **No Video Storage**: Raw webcam video and audio streams are processed ephemerally in-browser or streamed to memory. No biometric video footage is stored on the server by default.
- **Explicit Recording Consent**: Live captioning and meeting features show persistent, prominent visual indicators (`MIC RECORDING`) whenever audio capture is active.
- **User-Controlled History**: Users can disable telemetry logging via `/users/preferences` (`store_history = false`).
- **GDPR "Right to be Forgotten"**: Endpoint `DELETE /api/v1/history/clear?clear_practice=true` provides instant, irreversible data erasure of all user recognition events and practice logs.

---

## 3. Input Validation & Injection Protection
- **Pydantic Validation**: Every request payload undergoes schema validation before reaching business logic.
- **Parameterized SQL**: All database operations utilize SQLAlchemy parameterized ORM queries, completely mitigating SQL injection risks.
- **No Path Traversal**: Model paths are strictly hardcoded to project configuration constants; no client request can supply arbitrary filesystem paths.
