# RAG COACH GPU ERROR FIX

## Problem
```
Ollama call failed with status code 500
error loading model: unable to allocate CUDA0 buffer
```

This happens because:
- Ollama is trying to use GPU (CUDA) by default
- Your GPU doesn't have enough VRAM for Mistral 7B model
- Or CUDA drivers aren't properly configured

## Solution: Force CPU-Only Mode

### Step 1: Restart Ollama in CPU Mode

Run this command:
```powershell
.\RESTART_OLLAMA_CPU_MODE.bat
```

This will:
- Stop Ollama service
- Set environment variables to disable GPU
- Restart Ollama in CPU-only mode

### Step 2: Restart Backend

After Ollama is running in CPU mode, restart the backend:
```powershell
.\RESTART_BACKEND.bat
```

### Step 3: Test RAG Coach

1. Go to http://localhost:8501
2. Click "RAG Coach" tab
3. Upload PDFs (resume, job description)
4. Ask a question: "What are my key skills?"
5. Wait 20-30 seconds for first query (auto-builds index)

## What Changed?

### rag_coach.py
- Added `CUDA_VISIBLE_DEVICES=-1` to force CPU
- Added `num_gpu=0` parameter to Ollama
- Added `num_thread=4` for CPU threading

### Expected Performance (CPU Mode)
- First query: 25-35 seconds (builds index)
- Subsequent queries: 15-25 seconds
- Slightly slower than GPU, but **STABLE** and **NO CRASHES**

## Verification

After running both scripts, check Ollama is working:
```powershell
ollama run mistral:7b-instruct "Hello"
```

Should respond in 10-15 seconds without GPU errors.

## Alternative: Use Smaller Model

If CPU is too slow, you can use a smaller Mistral variant:

```powershell
# Stop Ollama
taskkill /F /IM ollama.exe

# Pull quantized 2-bit model (much faster on CPU)
ollama pull mistral:7b-instruct-q2_k

# Restart with RESTART_OLLAMA_CPU_MODE.bat
```

The RAG Coach will auto-detect the new model on next restart.

## Troubleshooting

### Still getting GPU errors?
1. Completely uninstall Ollama
2. Delete: `C:\Users\Arjun T Anil\.ollama`
3. Reinstall Ollama
4. Set environment variable BEFORE first run:
   ```powershell
   setx OLLAMA_NUM_GPU "0"
   ```
5. Pull model: `ollama pull mistral:7b-instruct`

### Ollama not starting?
Check if port 11434 is in use:
```powershell
netstat -ano | findstr :11434
```

Kill the process:
```powershell
taskkill /F /PID <process_id>
```

### Still slow on CPU?
Switch to ultra-fast Q2 quantized model:
```powershell
ollama pull mistral:7b-instruct-q2_k
```

This is 4x faster on CPU with minimal quality loss.
