# Complete Setup Guide - RAG Coach with Ollama

## ✅ Current Status

Based on your terminal output, here's what's already done:

- ✅ **Ollama Installed** - Version 0.12.6
- ✅ **Ollama PATH Updated** - Working after refresh
- ✅ **Model Download Started** - `mistral:7b-instruct-q4_K_M` is downloading
- ✅ **Dependencies Installed** - langchain-community, ollama, pypdf
- ✅ **RAG Coach Code Updated** - Auto-detects available Mistral models

## 🎯 What You Need to Do Now

### Step 1: Wait for Model Download to Complete

Your terminal is currently downloading `mistral:7b-instruct-q4_K_M` (4.4 GB).

**Progress visible in your terminal:**
```
pulling manifest
pulling faf975975644:  12% ▕███████                                                    ▏ 536 MB/4.4 GB
```

**Wait until you see:**
```
✅ success
```

This will take 5-15 minutes depending on your internet speed.

### Step 2: Verify Installation

Once download completes, in your PowerShell terminal (with career_coach activated):

```powershell
# Check installed models
ollama list
```

**Expected output:**
```
NAME                              ID              SIZE    MODIFIED
mistral:7b-instruct-q4_K_M       faf975975644    4.4 GB  X minutes ago
```

### Step 3: Test Ollama Works

```powershell
# Quick test
ollama run mistral:7b-instruct-q4_K_M
```

Type: `Hello, introduce yourself briefly`

Then type: `/bye` to exit

### Step 4: Run Verification Script

In your career_coach environment:

```powershell
E:/NextStepAI/career_coach/Scripts/python.exe verify_rag_coach_setup.py
```

This will check:
- ✅ Ollama command is available
- ✅ Mistral model is installed
- ✅ RAG Coach dependencies work
- ✅ Ollama LLM connection works

### Step 5: Start Your Application

**Terminal 1 - Backend:**
```powershell
E:/NextStepAI/career_coach/Scripts/python.exe -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

Wait for:
```
✅ Application startup complete.
```

**Terminal 2 - Frontend:**
```powershell
cd E:\NextStepAI
E:/NextStepAI/career_coach/Scripts/Activate.ps1
streamlit run app.py
```

### Step 6: Test RAG Coach

1. **Open Browser**: http://localhost:8501
2. **Click**: "🧑‍💼 RAG Coach" tab
3. **Upload**: 
   - Your resume PDF
   - Job description PDF (optional)
4. **Click**: "📤 Upload Documents to RAG Coach"
5. **Wait**: For index building (1-2 minutes)
6. **Ask**: "Based on my resume, what skills should I develop?"
7. **See**: AI-generated answer with source attribution

## 🔧 Troubleshooting

### Issue: "ollama: command not found" after restart

**Solution:**
```powershell
# Refresh PATH in current session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify
ollama --version
```

**Or:** Close all PowerShell windows and open a new one

### Issue: Model download interrupted

**Solution:**
```powershell
# Resume download
ollama pull mistral:7b-instruct-q4_K_M
```

Ollama will resume from where it stopped.

### Issue: "Model not found" when using RAG Coach

**Solution:**

The RAG Coach now auto-detects your installed model. If you have `mistral:7b-instruct-q4_K_M`, it will use it automatically.

You can verify by checking the backend logs when it initializes:
```
✅ Auto-detected Ollama model: mistral:7b-instruct-q4_k_m
```

### Issue: RAG Coach index building fails

**Possible causes:**
1. Ollama service not running
2. Model not fully downloaded
3. Not enough RAM (need ~6GB free)

**Solution:**
```powershell
# Check Ollama is running
Get-Process ollama -ErrorAction SilentlyContinue

# If not, run any ollama command to start it
ollama list

# Verify model is fully downloaded
ollama list
```

### Issue: Backend startup shows warnings

If you see:
```
⚠️  No Mistral model found in Ollama, using default: mistral:7b-instruct
```

This is OK! The backend will try to use the default model name, but when you actually use RAG Coach, it will auto-detect your installed model.

## 📊 Model Variants Explained

Your RAG Coach will work with any of these:

| Model Name | Size | Quantization | Recommended |
|------------|------|--------------|-------------|
| `mistral` | 4.1 GB | Full precision | ✅ Yes |
| `mistral:7b-instruct` | 4.1 GB | Full precision | ✅ Yes (best) |
| `mistral:7b-instruct-q4_K_M` | 4.4 GB | 4-bit (Q4) | ✅ Yes (downloading) |
| `mistral:latest` | 4.1 GB | Full precision | ✅ Yes |

**You're downloading:** `mistral:7b-instruct-q4_K_M` ✅

This is a good choice! The Q4 quantization makes it fast while maintaining quality.

## 🎯 Quick Commands Reference

```powershell
# Check Ollama version
ollama --version

# List installed models
ollama list

# Pull a model
ollama pull mistral:7b-instruct

# Test a model
ollama run mistral:7b-instruct

# Verify RAG Coach setup
E:/NextStepAI/career_coach/Scripts/python.exe verify_rag_coach_setup.py

# Start backend
E:/NextStepAI/career_coach/Scripts/python.exe -m uvicorn backend_api:app --reload

# Start frontend (in new terminal)
streamlit run app.py
```

## 📝 Files You Can Use

1. **`setup_ollama_in_terminal.ps1`** - Interactive setup script
2. **`verify_rag_coach_setup.py`** - Verification script
3. **`OLLAMA_INSTALLATION_GUIDE.md`** - Detailed installation guide
4. **`RAG_COACH_SETUP_GUIDE.md`** - Complete RAG Coach guide

## 🎉 You're Almost There!

Just wait for the model download to complete, then follow steps 2-6 above. Your RAG Coach will be ready to use!

---

**Need help?** Check the troubleshooting section or run the verification script.
