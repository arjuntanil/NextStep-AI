@echo off
echo ========================================
echo   NextStepAI - Quick Start
echo ========================================
echo.

cd /d E:\NextStepAI

echo Step 1: Activating virtual environment...
call .\career_coach\Scripts\activate.bat

echo.
echo Step 2: Starting Backend API on port 8000...
echo [This window will show backend logs]
echo.
start "Backend API - Port 8000" cmd /k "cd /d E:\NextStepAI && .\career_coach\Scripts\activate.bat && python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Step 3: Starting User App (with Admin Dashboard) on port 8501...
echo.
echo ========================================
echo   SYSTEM READY!
echo ========================================
echo.
echo Open in browser: http://localhost:8501
echo.
echo ADMIN LOGIN:
echo   Email:    admin@gmail.com
echo   Password: admin
echo.
echo NOTE: Admin users will see "Admin Controls" in sidebar
echo       Check "Show Admin Dashboard" to access admin panel
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

streamlit run app.py

pause
