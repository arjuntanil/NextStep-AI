@echo off
echo ========================================
echo   Starting NextStepAI Backend Server
echo ========================================
echo.

cd /d E:\NextStepAI

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo.
echo Starting backend on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\python.exe -m uvicorn backend_api:app --host 127.0.0.1 --port 8000

pause
