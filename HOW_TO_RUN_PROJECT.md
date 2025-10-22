# 🚀 How to Run NextStepAI Project

Complete step-by-step guide to run the NextStepAI Career Platform with RAG Coach.

---

## 📋 Prerequisites

- **Python 3.10** installed
- **Ollama** installed at `C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe`
- **Virtual Environment** at `E:\NextStepAI\career_coach\`
- **Internet connection** (for first-time model downloads)

---

## 🎯 Quick Start (3 Steps)

### Step 1: Start Ollama in CPU Mode

Open **PowerShell** and run:

```powershell
cd E:\NextStepAI
powershell -ExecutionPolicy Bypass -File "START_OLLAMA_CPU_MODE.ps1"
```

**Expected Output:**
```
========================================
  Starting Ollama in CPU-Only Mode
========================================
[OK] Ollama is running in CPU-ONLY mode
URL: http://127.0.0.1:11434
========================================
```

**Leave this window open!**

---

### Step 2: Start Backend API

Open a **NEW PowerShell window** and run:

```powershell
cd E:\NextStepAI
.\RESTART_BACKEND.bat
```

**Expected Output:**
```
========================================
  RESTARTING NextStepAI Backend
========================================
[OK] Backend will run on: http://127.0.0.1:8000
```

**Wait 15-20 seconds** for models to load. You'll see:
```
[OK] Resume analysis models loaded
[OK] Career Guide RAG chain created
[OK] Job Search RAG chain created
INFO: Uvicorn running on http://127.0.0.1:8000
```

**Leave this window open!**

---

### Step 3: Start Frontend

Open a **NEW PowerShell window** and run:

```powershell
cd E:\NextStepAI
.\START_FRONTEND.bat
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Your browser should automatically open!**

---

## 🌐 Access the Application

Once all three steps are complete:

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Ollama**: http://127.0.0.1:11434

---

## 🎨 Using the Features

### 1️⃣ **Resume Analyzer**
1. Click **"Resume Analyzer"** tab
2. Upload your resume (PDF, DOCX, or TXT)
3. Click **"Analyze Resume"**
4. Get instant analysis with:
   - Skills extraction
   - Experience summary
   - Improvement suggestions
   - ATS score

### 2️⃣ **AI Career Advisor** (Ultra-Fast Mode)
1. Click **"AI Career Advisor"** tab
2. Enter your question or career goal
3. Adjust **Response Length** slider (50-120 tokens)
4. Click **"Get AI Advice"**
5. Get response in **8-15 seconds** ⚡

**Example Questions:**
- "What skills do I need for Data Science?"
- "How to transition from developer to ML engineer?"
- "Best certifications for cloud computing?"

### 3️⃣ **RAG Coach** (Document-Based Q&A)
1. Click **"RAG Coach"** tab
2. **Upload PDFs**:
   - Resume
   - Job descriptions
   - Career guides
3. Click **"Upload Documents"**
4. **Ask Questions**:
   - "What are my key strengths based on my resume?"
   - "Am I qualified for this job posting?"
   - "What skills should I highlight?"
5. Get context-aware answers in **20-40 seconds**

### 4️⃣ **History**
- View all past queries and responses
- Review previous career advice
- Track your progress

---

## ⚙️ Technical Details

### System Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (Streamlit)                   │
│  Port: 8501                             │
└─────────────┬───────────────────────────┘
              │ HTTP Requests
              ▼
┌─────────────────────────────────────────┐
│  Backend (FastAPI)                      │
│  Port: 8000                             │
│  - Resume Analysis                      │
│  - AI Career Advisor (GPT-2 Medium)     │
│  - RAG Coach Integration                │
└─────────────┬───────────────────────────┘
              │ LLM Requests
              ▼
┌─────────────────────────────────────────┐
│  Ollama (CPU Mode)                      │
│  Port: 11434                            │
│  Model: TinyLlama (637 MB)              │
└─────────────────────────────────────────┘
```

### Models Used

| Component | Model | Size | Speed |
|-----------|-------|------|-------|
| AI Career Advisor | GPT-2 Medium (Fine-tuned) | 355M params | 8-15s |
| RAG Coach | TinyLlama | 637 MB | 20-40s |
| Embeddings | all-MiniLM-L6-v2 | 80 MB | <1s |

### Resource Requirements

- **RAM**: Minimum 2 GB free (4 GB recommended)
- **Disk**: 6 GB for models
- **CPU**: Multi-core recommended (runs on any CPU)

---

## 🔧 Troubleshooting

### ❌ Problem: "Port 8000 already in use"

**Solution:**
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
# Find PID (last column), then:
taskkill /F /PID <PID>

# Or just run:
.\RESTART_BACKEND.bat
```

---

### ❌ Problem: "Ollama memory error"

**Solution:**
1. Close unnecessary programs (Chrome, etc.)
2. Restart Ollama:
```powershell
Stop-Process -Name "ollama" -Force
powershell -ExecutionPolicy Bypass -File "START_OLLAMA_CPU_MODE.ps1"
```

---

### ❌ Problem: "RAG Coach not ready"

**Solution:**
1. Make sure Ollama is running (check Step 1)
2. Upload PDFs first in RAG Coach tab
3. Wait 5-10 seconds for background indexing
4. Try query again

---

### ❌ Problem: "AI Career Advisor timeout"

**Solution:**
1. Reduce **Response Length** to 50-80 tokens
2. Keep questions short and specific
3. Make sure backend is running (check Step 2)

---

### ❌ Problem: "Module not found" errors

**Solution:**
```powershell
cd E:\NextStepAI
& E:/NextStepAI/career_coach/Scripts/Activate.ps1
pip install -r requirements.txt
```

---

## 🛑 How to Stop Everything

### Stop Frontend
- Press `Ctrl+C` in the frontend PowerShell window

### Stop Backend
- Press `Ctrl+C` in the backend PowerShell window

### Stop Ollama
```powershell
Stop-Process -Name "ollama" -Force
```

### Stop All at Once
```powershell
# Stop all services
taskkill /F /IM python.exe
Stop-Process -Name "ollama" -Force
```

---

## 🔄 Restart Everything

If something goes wrong, restart all services:

```powershell
# 1. Stop everything
taskkill /F /IM python.exe
Stop-Process -Name "ollama" -Force

# 2. Wait 5 seconds
Start-Sleep -Seconds 5

# 3. Start Ollama (in Window 1)
powershell -ExecutionPolicy Bypass -File "START_OLLAMA_CPU_MODE.ps1"

# 4. Start Backend (in Window 2 - wait 10 seconds)
.\RESTART_BACKEND.bat

# 5. Start Frontend (in Window 3 - wait 20 seconds)
.\START_FRONTEND.bat
```

---

## 📊 Performance Optimization

### For Faster AI Career Advisor:
1. Set **Response Length** to 50-80 tokens
2. Keep questions under 100 characters
3. Use simple, direct questions

### For Faster RAG Coach:
1. Upload smaller PDFs (<5 pages each)
2. Limit to 3-5 documents max
3. Ask specific questions (not general ones)

### To Free Up Memory:
1. Close Chrome/Edge browsers
2. Close other heavy applications
3. Restart Ollama to clear cached models

---

## 📝 Batch Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `START_OLLAMA_CPU_MODE.ps1` | Start Ollama in CPU-only mode | Step 1 |
| `RESTART_BACKEND.bat` | Kill old backend, start new | Step 2 |
| `START_FRONTEND.bat` | Start Streamlit frontend | Step 3 |
| `test_rag_cpu_mode.py` | Test RAG Coach functionality | Testing |

---

## 🎓 Features Overview

### ✅ Working Features:
- ✅ Resume Analysis (instant)
- ✅ AI Career Advisor (8-15 seconds)
- ✅ RAG Coach with PDF uploads (20-40 seconds)
- ✅ Job recommendations
- ✅ Skill analysis
- ✅ Career path suggestions
- ✅ Query history

### 🚀 Optimizations Applied:
- ✅ CPU-only mode (no GPU required)
- ✅ TinyLlama for low memory usage
- ✅ Extreme speed optimizations for Career Advisor
- ✅ Background PDF indexing
- ✅ Auto-initialization for RAG Coach
- ✅ Reduced token limits

---

## 📞 Support

### Check Logs:
- **Backend logs**: In the backend PowerShell window
- **Frontend logs**: In the frontend PowerShell window  
- **Ollama logs**: In the Ollama PowerShell window

### Test Individual Components:

**Test Backend:**
```powershell
curl http://127.0.0.1:8000/
```

**Test Ollama:**
```powershell
ollama list
```

**Test RAG Coach:**
```powershell
python test_rag_cpu_mode.py
```

---

## 🎉 You're All Set!

Your NextStepAI platform is ready to use. Follow the 3-step Quick Start guide and start using all features!

**Need help?** Check the troubleshooting section or review the logs in each PowerShell window.

---

**Last Updated:** October 18, 2025  
**Version:** 2.0 (RAG Coach + Ultra-Fast Mode)
