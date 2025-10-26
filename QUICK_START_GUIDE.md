# 🚀 NextStepAI - Quick Start Guide

## One-Click Startup

**Double-click this file to start everything:**
```
START_SYSTEM.bat
```

This will:
1. ✅ Start Backend API (port 8000)
2. ✅ Start User App with Admin Dashboard (port 8501)
3. ✅ Open automatically in 5 seconds

---

## Access URLs

### For All Users:
**Main App:** http://localhost:8501

### For Admins:
1. Login with: `admin@gmail.com` / `admin`
2. Look for **"Admin Controls"** in sidebar
3. Check **"📊 Show Admin Dashboard"**
4. View admin panel with stats, users, charts

---

## Admin Credentials

```
Email:    admin@gmail.com
Password: admin
```

---

## System Architecture

```
┌─────────────────────────────────────┐
│   http://localhost:8501             │
│   ┌───────────────────────────────┐ │
│   │  Regular User View            │ │
│   │  - CV Analyzer                │ │
│   │  - AI Career Advisor          │ │
│   │  - Resume Analyzer using JD   │ │
│   │  - My History                 │ │
│   └───────────────────────────────┘ │
│                                     │
│   ┌───────────────────────────────┐ │
│   │  Admin View (toggle)          │ │
│   │  - System Overview            │ │
│   │  - User Management            │ │
│   │  - Activity Charts            │ │
│   │  - System Info                │ │
│   └───────────────────────────────┘ │
└─────────────────────────────────────┘
              ↕ API Calls
┌─────────────────────────────────────┐
│   Backend API (FastAPI)             │
│   http://localhost:8000             │
│   - User Authentication             │
│   - Resume Analysis                 │
│   - Career Advice                   │
│   - RAG System                      │
│   - Admin Stats                     │
└─────────────────────────────────────┘
```

---

## Features

### For Regular Users:
- 📄 **CV Analyzer** - Upload resume, get skill gap analysis
- 💬 **AI Career Advisor** - Get career path recommendations
- 🧑‍💼 **Resume Analyzer using JD** - Match resume to job description
- 🗂️ **My History** - View past analyses and queries

### For Admins (admin@gmail.com):
- 📊 **System Overview** - User counts, analyses, queries
- 👥 **User Management** - View all users, status, activity
- 📈 **Activity Charts** - 7-day trends, visualizations
- ⚙️ **System Info** - Retention rates, average scores

---

## Troubleshooting

### ❌ ERR_CONNECTION_REFUSED

**Problem:** Backend API not running

**Solution:**
1. Check if backend terminal shows:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```
2. If not, restart: `START_SYSTEM.bat`
3. Or manually start backend first

### ❌ Admin Controls Not Showing

**Problem:** Not logged in as admin

**Solution:**
1. Logout if logged in
2. Login with: `admin@gmail.com` / `admin`
3. Admin controls should appear in sidebar

### ❌ History Not Saving

**Problem:** Not logged in OR backend not running

**Solution:**
1. Make sure you're logged in (see email in sidebar)
2. Ensure backend is running
3. Try the feature again

---

## Manual Startup (If Batch File Doesn't Work)

### Terminal 1 - Backend:
```bash
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2 - Frontend:
```bash
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
streamlit run app.py
```

### Open Browser:
```
http://localhost:8501
```

---

## Service Status Check

### Backend (Must be running):
- Open: http://localhost:8000/docs
- Should see: FastAPI documentation page
- If connection refused: Backend not running!

### Frontend:
- Open: http://localhost:8501
- Should see: NextStepAI login/main page

---

## Files Overview

| File | Purpose |
|------|---------|
| `START_SYSTEM.bat` | **One-click startup** (use this!) |
| `app.py` | Main user interface + admin dashboard |
| `backend_api.py` | Backend API server |
| `rag_coach.py` | RAG system for resume analysis |
| `TROUBLESHOOTING.md` | Detailed troubleshooting guide |

---

## Project Structure

```
NextStepAI/
├── START_SYSTEM.bat          ← Use this to start!
├── TROUBLESHOOTING.md         ← Help with issues
├── app.py                     ← Main UI (port 8501)
├── backend_api.py             ← API server (port 8000)
├── rag_coach.py               ← RAG system
├── models.py                  ← Database models
├── requirements.txt           ← Dependencies
├── career_coach/              ← Virtual environment
│   └── Scripts/
│       ├── activate.bat
│       ├── python.exe
│       └── streamlit.exe
├── uploads/                   ← User uploaded files
└── rag_coach_index/          ← Vector store
```

---

## Default Users

### Admin User:
```
Email:    admin@gmail.com
Password: admin
Role:     admin
```

### Test User (if needed):
```
Register new user through the app
Role will be: user
```

---

## Important Notes

⚠️ **Always start Backend FIRST** before Frontend  
⚠️ **Backend must run on port 8000** (required for admin dashboard)  
⚠️ **Admin controls only visible to admin users**  
⚠️ **History only saved for logged-in users**  

---

## Quick Commands

```bash
# Start everything
START_SYSTEM.bat

# Kill all processes (if stuck)
taskkill /F /IM python.exe
taskkill /F /IM streamlit.exe

# Check backend
http://localhost:8000/docs

# Check frontend
http://localhost:8501
```

---

## Need More Help?

See: `TROUBLESHOOTING.md` for detailed solutions
