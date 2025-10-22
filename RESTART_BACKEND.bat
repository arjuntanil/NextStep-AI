@echo off
echo ========================================
echo   RESTARTING NextStepAI Backend
echo ========================================
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

echo [2.5/3] Setting CPU-only mode for Ollama...
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0

echo [3/3] Starting backend server...
echo.
echo Backend will run on: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\python.exe -m uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000

pause
