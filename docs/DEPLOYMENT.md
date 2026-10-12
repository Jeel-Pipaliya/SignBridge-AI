# SignBridge AI — DevOps & Deployment Guide

## 1. Local Development Setup (Quick Start)

### Prerequisites:
- Python 3.10+ (Tested on Python 3.12)
- Node.js 18+ (Tested on Node 22.22)
- Standard USB / integrated webcam

### 1.1 Backend Setup
```bash
# Clone repository
git clone https://github.com/Jeel-Pipaliya/SignBridge-AI.git
cd SignBridge-AI

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic pydantic-settings sqlalchemy alembic bcrypt pyjwt python-multipart httpx pytest email-validator

# Run database migrations and seed data
alembic upgrade head
python -m backend.app.db.seed

# Start FastAPI backend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API will be live at `http://127.0.0.1:8000` (Docs at `/docs`).

### 1.2 Frontend Setup
```bash
cd frontend

# Install npm dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 2. Docker & Containerized Production Deployment

SignBridge AI provides a production-grade multi-container Docker Compose configuration consisting of:
1. `db`: PostgreSQL 16 Alpine
2. `backend`: FastAPI Python service with OpenCV & MediaPipe dependencies
3. `frontend`: Nginx Alpine serving the production-built React application

```bash
# Configure environment variables
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Launch entire stack in detached mode
docker-compose up -d --build

# Verify container health
docker-compose ps
```
- Frontend Web App: `http://localhost`
- Backend REST API: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`

---

## 3. Automated Test Execution

```bash
# Run complete test suite (96 automated tests)
python -m unittest discover -s tests -p "test_*.py"

# Run frontend production build check
cd frontend && npm run build
```
