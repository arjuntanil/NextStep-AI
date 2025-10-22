# 🚀 HOW TO RUN YOUR PROJECT

## ✅ CURRENT STATUS

Your backend is **ALREADY RUNNING** on port 8000 (PID: 23036)

---

## 📋 SIMPLE STEPS TO RUN:

### Method 1: Quick Start (EASIEST)

**1. Backend (Already Running!)**
```
✅ Backend is running at: http://127.0.0.1:8000
```

**2. Start Frontend:**
Open a **NEW terminal** and run:
```powershell
cd E:\NextStepAI
.\career_coach\Scripts\activate
streamlit run app.py
```

OR just double-click:
```
START_FRONTEND.bat
```

**3. Open your browser:**
- Frontend: http://localhost:8501
- Backend API docs: http://127.0.0.1:8000/docs

---

## 🔄 If You Need to Restart Backend:

### Stop Backend:
```powershell
Get-Process -Name "python" | Where-Object {$_.CommandLine -like "*uvicorn*"} | Stop-Process -Force
```

### Start Backend:
**Option A - Using batch file:**
```
Double-click: START_BACKEND.bat
```

**Option B - Using terminal:**
```powershell
cd E:\NextStepAI
.\career_coach\Scripts\activate
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

---

## 📂 Complete Startup Commands:

### Terminal 1 - Backend:
```powershell
cd E:\NextStepAI
.\career_coach\Scripts\activate
uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

### Terminal 2 - Frontend:
```powershell
cd E:\NextStepAI
.\career_coach\Scripts\activate  
streamlit run app.py
```

---

## 🎯 What You'll See:

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

## 🌐 Access Your Application:

Once both servers are running:

1. **Main Application:** http://localhost:8501
   - Resume Analyzer
   - AI Career Advisor (⚡ 8-15 second responses)
   - RAG Coach
   - History

2. **API Documentation:** http://127.0.0.1:8000/docs
   - Interactive API testing
   - All endpoints documented

---

## ⚡ Quick Test:

**Test Backend:**
```powershell
curl http://127.0.0.1:8000/docs
```

**Test Frontend:**
```powershell
curl http://localhost:8501
```

---

## 🎮 Using the Application:

### 1. Resume Analyzer Tab:
- Upload your resume (PDF/DOCX)
- Get instant skill extraction
- See job recommendations
- Get learning resource links

### 2. AI Career Advisor Tab:
- Set **Response Length: 80** for fastest results (8-15s)
- Set **Creativity: 0.5** for focused answers
- Ask any career question
- Get detailed advice quickly

### 3. RAG Coach Tab:
- Upload PDFs (resumes, job descriptions, guides)
- Ask questions about uploaded content
- Get context-aware responses

---

## 🐛 Troubleshooting:

### Port Already in Use:
```powershell
# Kill process on port 8000:
$pid = (netstat -ano | findstr ":8000.*LISTENING" | ForEach-Object {($_ -split '\s+')[-1]})[0]
Stop-Process -Id $pid -Force

# Restart backend:
START_BACKEND.bat
```

### Backend Not Responding:
```powershell
# Check if running:
Get-Process -Name "python" | Where-Object {$_.CommandLine -like "*uvicorn*"}

# If not, start it:
START_BACKEND.bat
```

### Frontend Won't Start:
```powershell
# Make sure virtual environment is activated:
.\career_coach\Scripts\activate

# Check Streamlit is installed:
streamlit --version

# Start frontend:
streamlit run app.py
```

---

## 📝 Environment Setup (Only Once):

If this is your first time or after pulling new code:

```powershell
# Activate virtual environment:
cd E:\NextStepAI
.\career_coach\Scripts\activate

# Install/update dependencies:
pip install -r requirements.txt

# For RAG Coach (Ollama should already be installed):
# Check: ollama list
# Should show: mistral:7b-instruct
```

---

## ✅ Quick Checklist:

Before running, make sure you have:
- [x] Python virtual environment (`career_coach`)
- [x] All dependencies installed (from requirements.txt)
- [x] Ollama installed (for RAG Coach)
- [x] Mistral model downloaded (`mistral:7b-instruct`)

---

## 🎉 YOU'RE READY!

**Right now:**
✅ Backend is running on port 8000

**To complete setup:**
1. Open new terminal
2. Run: `streamlit run app.py`
3. Open browser to http://localhost:8501
4. Start using your AI Career Advisor! ⚡

**Response times:**
- AI Career Advisor: 8-15 seconds
- RAG Coach: 10-15 seconds
- Resume Analysis: Instant

---

## 🚀 ONE-COMMAND START (Next Time):

**Terminal 1:**
```powershell
START_BACKEND.bat
```

**Terminal 2:**
```powershell
START_FRONTEND.bat
```

**That's it!** 🎯
