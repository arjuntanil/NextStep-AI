@echo off
REM Quick Start Script for NextStepAI with RAG Coach
REM Run this after Mistral model download completes

echo.
echo ========================================
echo   NextStepAI - Quick Start
echo ========================================
echo.

REM Step 1: Check if Ollama model exists
echo [1/3] Checking Mistral model...
"C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" list | findstr /i "mistral" >nul
if errorlevel 1 (
    echo   WARNING: No Mistral model found!
    echo   Run: ollama pull mistral:7b-instruct
    echo.
    pause
    exit /b 1
) else (
    echo   OK: Mistral model is ready!
)
echo.

REM Step 2: Start Backend
echo [2/3] Starting Backend Server...
echo   Starting FastAPI backend on http://127.0.0.1:8000
echo   Close this window to stop the backend
echo.
start "NextStepAI Backend" /MIN cmd /c "E:\NextStepAI\career_coach\Scripts\python.exe -m uvicorn backend_api:app --reload & pause"

REM Wait for backend to start
timeout /t 10 /nobreak >nul

REM Step 3: Start Frontend
echo [3/3] Starting Frontend...
echo   Starting Streamlit on http://localhost:8501
echo   Your browser will open automatically
echo.
start "NextStepAI Frontend" cmd /c "cd E:\NextStepAI & E:\NextStepAI\career_coach\Scripts\activate.bat & streamlit run app.py"

echo.
echo ========================================
echo   NextStepAI is starting!
echo ========================================
echo.
echo   Backend:  http://127.0.0.1:8000/docs
echo   Frontend: http://localhost:8501
echo.
echo   To stop: Close the terminal windows
echo.
pause
