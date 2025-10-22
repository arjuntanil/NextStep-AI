@echo off
echo ============================================
echo   AUTO-FIX RAG COACH GPU ERROR
echo ============================================
echo.

REM STEP 1: Restart Ollama in CPU mode
echo [STEP 1/2] Restarting Ollama in CPU mode...
echo.

echo Stopping Ollama...
taskkill /F /IM ollama.exe 2>nul
timeout /t 2 /nobreak >nul

echo Setting CPU-only environment...
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0

echo Starting Ollama (CPU mode)...
start "" "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 5 /nobreak >nul
echo [OK] Ollama started in CPU mode
echo.

REM STEP 2: Restart Backend
echo [STEP 2/2] Restarting Backend with CPU settings...
echo.

cd /d E:\NextStepAI

echo Stopping existing backend...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo Starting backend server...
echo.
echo ============================================
echo   ALL FIXED! BACKEND RUNNING (CPU MODE)
echo ============================================
echo.
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:8501
echo.
echo RAG Coach Performance (CPU mode):
echo - First query: 25-35 seconds
echo - Subsequent queries: 15-25 seconds
echo.
echo Press Ctrl+C to stop
echo.

E:\NextStepAI\career_coach\Scripts\python.exe -m uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
