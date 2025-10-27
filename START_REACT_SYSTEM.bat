@echo off
echo ====================================
echo NextStepAI - Complete System Startup
echo ====================================
echo.
echo This will start:
echo 1. Backend API (FastAPI)
echo 2. React Frontend
echo.
echo Press Ctrl+C in each window to stop
echo ====================================
echo.

echo Starting Backend API...
start cmd /k "cd /d %~dp0 && call START_BACKEND.bat"

timeout /t 5

echo Starting React Frontend...
start cmd /k "cd /d %~dp0 && call START_REACT_FRONTEND.bat"

echo.
echo ====================================
echo All systems starting!
echo ====================================
echo Backend API: http://127.0.0.1:8000
echo React Frontend: http://localhost:3000
echo API Docs: http://127.0.0.1:8000/docs
echo ====================================
