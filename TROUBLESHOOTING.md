# Quick Troubleshooting Guide

## ERR_CONNECTION_REFUSED - Backend Not Running

### Problem
When you see "ERR_CONNECTION_REFUSED" or admin dashboard fails to load, it means the **Backend API (port 8000) is not running**.

### Solution

#### Option 1: Use the Quick Start Script (EASIEST)
1. **Double-click:** `START_SYSTEM.bat`
2. Wait for both services to start (5-10 seconds)
3. Open browser: http://localhost:8501
4. Done!

#### Option 2: Manual Start (Two Terminals)

**Terminal 1 - Start Backend:**
```bash
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

**Wait for this message:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Terminal 2 - Start Frontend:**
```bash
cd E:\NextStepAI
.\career_coach\Scripts\activate.bat
streamlit run app.py
```

**Open:** http://localhost:8501

---

## How to Verify Backend is Running

### Test 1: Check API Docs
Open in browser: http://localhost:8000/docs

- ✅ **If you see FastAPI docs** → Backend is running!
- ❌ **If connection refused** → Backend is NOT running

### Test 2: Check Terminal
Look for these messages in backend terminal:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
[OK] Resume analysis models loaded
[OK] Gemini LLM instance initialized
```

---

## Admin Dashboard Access

### Step 1: Login as Admin
1. Go to http://localhost:8501
2. Login with:
   - Email: `admin@gmail.com`
   - Password: `admin`

### Step 2: Enable Admin View
1. Look at **sidebar** (left side)
2. Find section: **"👨‍💼 Admin Controls"**
3. Check the box: **"📊 Show Admin Dashboard"**
4. Page will refresh and show admin panel

### If Admin Controls Not Showing:
- ✅ Make sure you're logged in as `admin@gmail.com` (not another user)
- ✅ Check that user role is 'admin' in database
- ✅ Try logging out and logging in again

---

## Common Issues

### Issue 1: "No admin controls in sidebar"
**Cause:** Not logged in as admin user

**Solution:**
1. Logout if logged in
2. Login with: admin@gmail.com / admin
3. Admin controls should appear

### Issue 2: "Failed to load admin stats"
**Cause:** Backend API not running

**Solution:**
1. Check if backend is running: http://localhost:8000/docs
2. If not, start backend first (see above)
3. Refresh the page

### Issue 3: "Port already in use"
**Cause:** Services already running in background

**Solution:**
```bash
# Kill all Python and Streamlit processes
taskkill /F /IM python.exe
taskkill /F /IM streamlit.exe

# Then restart
START_SYSTEM.bat
```

### Issue 4: "History not saving"
**Cause:** Missing fields in database save

**Solution:**
- This was fixed in the latest update
- Make sure you're using the updated backend_api.py
- History should now save with query_length and answer_length fields

---

## Service Ports

| Service | Port | URL | Status Check |
|---------|------|-----|--------------|
| Backend API | 8000 | http://localhost:8000/docs | Must be running ⚠️ |
| User App + Admin | 8501 | http://localhost:8501 | Main interface |

---

## Starting Fresh

If nothing works, do a clean restart:

```bash
# 1. Kill all processes
taskkill /F /IM python.exe
taskkill /F /IM streamlit.exe

# 2. Wait 5 seconds
timeout /t 5

# 3. Start fresh
START_SYSTEM.bat
```

---

## Checking Logs

### Backend Logs
Look at the terminal where you started the backend.

**Good signs:**
```
[OK] Resume analysis models loaded
[OK] Gemini LLM instance initialized
[OK] RAG Coach initialized
INFO: 127.0.0.1 - "GET /admin/stats HTTP/1.1" 200 OK
```

**Bad signs:**
```
ERROR: Connection refused
ERROR: Module not found
ERROR: Port already in use
```

### Frontend Logs
Look at the terminal where you started streamlit.

**Good signs:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

## Testing the System

### Test Admin Dashboard:
1. ✅ Start backend and frontend
2. ✅ Open http://localhost:8501
3. ✅ Login as admin@gmail.com
4. ✅ See "Admin Controls" in sidebar
5. ✅ Check "Show Admin Dashboard"
6. ✅ See stats: Total Users, Analyses, etc.
7. ✅ See user table with data

### Test User Features:
1. ✅ Uncheck "Show Admin Dashboard"
2. ✅ See normal tabs: CV Analyzer, AI Career Advisor, etc.
3. ✅ Upload resume in CV Analyzer
4. ✅ Get analysis results
5. ✅ Check "My History" tab
6. ✅ See saved analyses

---

## Need Help?

1. Check both terminals (backend + frontend) for error messages
2. Verify backend is accessible: http://localhost:8000/docs
3. Try clean restart (kill processes, start fresh)
4. Check if ports 8000 and 8501 are free

---

## Quick Reference

**Start Everything:**
```
START_SYSTEM.bat
```

**Admin Login:**
```
Email: admin@gmail.com
Password: admin
```

**Main URL:**
```
http://localhost:8501
```

**Backend API Docs:**
```
http://localhost:8000/docs
```
