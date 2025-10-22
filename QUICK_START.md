# 🚀 QUICK START - How to Run Your Project

## ✅ SIMPLE METHOD (Recommended):

### Just double-click this file:
```
RESTART_BACKEND.bat
```
This will:
- ✅ Stop any existing backend on port 8000
- ✅ Activate virtual environment
- ✅ Start backend with all RAG Coach fixes

Then in a **new terminal**, run:
```
START_FRONTEND.bat
```

---

## 🔧 MANUAL METHOD:

If you prefer command line:

### Step 1: Stop existing backend
```cmd
for /f "tokens=5" %a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do taskkill /F /PID %a
```

### Step 2: Start backend
```cmd
cd E:\NextStepAI
career_coach\Scripts\activate
uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
```

### Step 3: Start frontend (in NEW terminal)
```cmd
cd E:\NextStepAI
career_coach\Scripts\activate
streamlit run app.py
```

---

## ⚡ FASTEST METHOD:

### Terminal 1:
```cmd
RESTART_BACKEND.bat
```

### Terminal 2:
```cmd
START_FRONTEND.bat
```

### Browser:
```
http://localhost:8501
```

---

## 🐛 Fixing "Port 8000 in use" Error:

### Quick Fix:
```cmd
netstat -ano | findstr ":8000"
```
Find the PID (last column), then:
```cmd
taskkill /F /PID <PID_NUMBER>
```

### Or use the RESTART_BACKEND.bat - it does this automatically!

---

## ✅ Expected Output:

### Backend Terminal:
```
INFO: Uvicorn running on http://127.0.0.1:8000
[OK] Embedding model initialized
[OK] Resume analysis models loaded
[OK] Career Guide RAG chain created
[OK] Job Search RAG chain created
INFO: Application startup complete
```

### Frontend Terminal:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

## 🎯 Access Your Application:

- **Main App:** http://localhost:8501
- **API Docs:** http://127.0.0.1:8000/docs

---

## 🎉 That's It!

Your project is now running with:
- ⚡ Fast AI responses (8-15 seconds)
- 🚀 RAG Coach with PDF uploads
- ✅ All optimizations applied

**Enjoy your NextStepAI Career Advisor!**
