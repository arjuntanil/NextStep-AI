@echo off
echo ============================================
echo   FIX PORT 8000 + START OLLAMA + BACKEND
echo ============================================
echo.

echo [STEP 1/5] Killing any process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo   Found PID: %%a
    taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak >nul
echo   Port 8000 is now free!
echo.

echo [STEP 2/5] Stopping any running Ollama...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul
echo.

echo [STEP 3/5] Starting Ollama in CPU mode...
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0
start "" "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 3 /nobreak >nul
echo   Ollama started!
echo.

echo [STEP 4/5] Checking available models...
ollama list
echo.

echo [STEP 5/5] Instructions for next steps:
echo.
echo   If you see 'tinyllama' in the list above:
echo     - Good! The model is already installed
echo     - Skip to STEP 6
echo.
echo   If NO models listed or tinyllama missing:
echo     - Run: ollama pull tinyllama
echo     - Wait for download to complete
echo.
echo   [STEP 6] Start backend:
echo     - Open NEW terminal
echo     - Run: .\START_BACKEND.bat
echo.
echo ============================================
pause
