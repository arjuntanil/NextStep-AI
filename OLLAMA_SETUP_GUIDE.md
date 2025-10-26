# 🚀 Quick Fix: Port 8000 + Ollama Setup

## Problem 1: Port 8000 Already in Use ❌
**Error:** `error while attempting to bind on address ('127.0.0.1', 8000)`

### Solution:
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
# Find the PID (last column), then:
taskkill /F /PID <PID_NUMBER>
```

---

## Problem 2: Ollama Model Not Found ❌
**Error:** `pull model manifest: file does not exist` when pulling `mistral:7b-q4`

### Root Cause:
The model name `mistral:7b-q4` doesn't exist in Ollama's registry. Use `tinyllama` instead (much smaller and faster).

---

## ✅ COMPLETE SOLUTION (One-Click Fix)

### Option 1: Automated Setup (RECOMMENDED)
```powershell
# Run this script - it does EVERYTHING:
.\SETUP_OLLAMA_TINYLLAMA.bat
```

**What it does:**
1. ✅ Kills port 8000
2. ✅ Starts Ollama in CPU mode
3. ✅ Installs tinyllama (637 MB, ~2-5 min download)
4. ✅ Tests the model
5. ✅ Ready for backend!

---

### Option 2: Manual Steps

#### Step 1: Free Port 8000
```powershell
netstat -ano | findstr :8000
taskkill /F /PID <PID>  # Replace <PID> with actual number
```

#### Step 2: Start Ollama
```powershell
# Stop existing Ollama
taskkill /F /IM ollama.exe

# Start in CPU mode
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OLLAMA_NUM_GPU = "0"
Start-Process "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" -ArgumentList "serve"
```

#### Step 3: Install tinyllama
```powershell
# Check what's installed
ollama list

# Pull tinyllama (637 MB, fastest)
ollama pull tinyllama

# Test it
ollama run tinyllama "Hello, introduce yourself briefly"
```

#### Step 4: Start Backend
```powershell
.\START_BACKEND.bat
```

**Backend will auto-detect tinyllama** (it's already configured as priority #1 in `rag_coach.py`!)

---

## 🎯 Why tinyllama?

| Model | RAM Usage | Speed | Status |
|-------|-----------|-------|--------|
| **tinyllama** | 637 MB | ⚡ Fastest | ✅ Works! |
| mistral:7b-q4 | N/A | N/A | ❌ Doesn't exist |
| mistral:7b-instruct | 1.6 GB | Medium | ✅ Alternative |

**Your backend already prioritizes tinyllama!** Check `rag_coach.py` line 112:
```python
preferred_models = [
    'tinyllama',                     # 637 MB RAM - BEST for very low memory
    'mistral:7b-instruct-q2_k',      # 800 MB RAM
    # ... other models
]
```

---

## 📋 Verification Checklist

After running setup, verify:

```powershell
# 1. Check Ollama is running
ollama list
# Should show: tinyllama

# 2. Test model works
ollama run tinyllama "Hello"
# Should respond with text

# 3. Check port 8000 is free
netstat -ano | findstr :8000
# Should be empty

# 4. Start backend
.\START_BACKEND.bat
# Should start without port error

# 5. Check backend logs
# Should see: "✅ Auto-detected Ollama model: tinyllama"
```

---

## 🚀 Full Restart Sequence

If anything goes wrong, restart everything:

```powershell
# 1. Stop everything
taskkill /F /IM python.exe
taskkill /F /IM ollama.exe
taskkill /F /IM streamlit.exe

# 2. Run complete setup
.\SETUP_OLLAMA_TINYLLAMA.bat

# 3. Start backend (NEW terminal)
.\START_BACKEND.bat

# 4. Start frontend (NEW terminal)
.\START_FRONTEND.bat
```

---

## 💡 Alternative Models (if tinyllama too slow)

```powershell
# Slightly larger but more capable:
ollama pull mistral:7b-instruct-q2_k  # 800 MB

# Or full quality (slower):
ollama pull mistral:7b-instruct       # 1.6 GB
```

Backend will auto-detect whichever you install!

---

## ⚙️ Backend Auto-Detection

Your `rag_coach.py` automatically detects models in this order:
1. **tinyllama** ← Fastest, lowest RAM
2. mistral:7b-instruct-q2_k
3. mistral:7b-instruct-q4_k_m
4. mistral:7b-instruct
5. mistral:latest

**No configuration needed!** Just install any model from the list.
