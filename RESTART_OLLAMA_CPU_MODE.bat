@echo off
echo ============================================
echo   RESTART OLLAMA IN CPU-ONLY MODE
echo ============================================
echo.

echo [1/4] Stopping Ollama service...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Setting CPU-only environment...
setx CUDA_VISIBLE_DEVICES "-1"
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0

echo [3/4] Starting Ollama in CPU mode...
start "" "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" serve

echo [4/4] Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   OLLAMA RESTARTED IN CPU MODE
echo ============================================
echo.
echo Ollama is now running in CPU-only mode.
echo This avoids GPU memory errors.
echo.
echo Next: Run RESTART_BACKEND.bat
echo.
pause
