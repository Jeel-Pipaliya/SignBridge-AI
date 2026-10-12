# SignBridge AI — Diagnostics & Troubleshooting Guide

## 1. Common Issues & Solutions

### 1.1 "Camera permission denied" or Black Video Feed
- **Cause**: Browser blocked access to `navigator.mediaDevices.getUserMedia`.
- **Solution**:
  1. Click the site security lock icon in your browser URL bar (e.g. `http://localhost:5173`).
  2. Toggle **Camera** to **Allow**.
  3. Refresh the page or click **Start Webcam Recognition**.
  4. Alternatively, use the **Direct Test Samples** buttons on `/recognize` to verify the AI recognition pipeline in headless or camera-less environments.

### 1.2 "ImportError: DLL load failed while importing _util_cy: An Application Control policy has blocked this file" (Windows)
- **Cause**: Windows AppLocker or Application Control policy blocks newly unpacked `.pyd` compiled C-extensions in user directories.
- **Solution**: SQLAlchemy has pure-Python implementations for all Cython modules (`_util_cy.py`, `_row_cy.py`, etc.). Renaming `.pyd` to `.pyd.disabled` causes Python to automatically load the safe `.py` equivalent without functionality loss.

### 1.3 Windows Audio pyttsx3 SAPI5 Transliteration
- **Cause**: Windows SAPI5 voice engine throws character mapping errors when attempting to speak raw Devanagari Unicode characters without a dedicated Hindi speech pack installed.
- **Solution**: `speech/tts.py` incorporates an automatic Romanized phonetic transliteration filter (e.g. `नमस्ते!` -> `Namaste!`), ensuring continuous speech output on all Windows versions.

### 1.4 Database Migration "no such table: users"
- **Solution**: Run Alembic migrations and database seeding:
  ```bash
  alembic upgrade head
  python -m backend.app.db.seed
  ```
