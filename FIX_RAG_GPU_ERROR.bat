@echo off
echo ============================================
echo   COMPLETE RAG COACH FIX - GPU ERROR
echo ============================================
echo.
echo This script will:
echo 1. Stop Ollama and restart in CPU mode
echo 2. Stop backend and restart with CPU settings
echo 3. Make RAG Coach work without GPU errors
echo.
echo Press any key to start...
pause >nul

echo.
echo ============================================
echo   STEP 1: RESTART OLLAMA (CPU MODE)
echo ============================================
echo.

echo [1/4] Stopping Ollama service...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul

echo [2/4] Setting CPU-only environment...
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0

echo [3/4] Starting Ollama in CPU mode...
start "" "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" serve

echo [4/4] Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

echo.
echo [OK] Ollama restarted in CPU mode
echo.

echo ============================================
echo   STEP 2: RESTART BACKEND (CPU MODE)
echo ============================================
echo.

cd /d E:\NextStepAI

echo [1/3] Stopping any existing backend...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    echo Killing process ID: %%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo [2/3] Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo [3/3] Starting backend server...
echo.
echo ============================================
echo   BACKEND STARTING IN CPU MODE
echo ============================================
echo.
echo Backend URL: http://127.0.0.1:8000
echo Frontend URL: http://localhost:8501
echo.
echo RAG Coach is now in CPU-only mode.
echo Expected performance:
echo - First query: 25-35 seconds (builds index)
echo - Subsequent queries: 15-25 seconds
echo.
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\python.exe -m uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
