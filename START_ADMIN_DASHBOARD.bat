@echo off
echo ========================================
echo   Starting Admin Dashboard
echo ========================================
echo.

cd /d E:\NextStepAI

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo.
echo Starting Admin Dashboard on http://localhost:8502
echo.
echo Admin Login Credentials:
echo   Email:    admin@gmail.com
echo   Password: admin
echo.
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\streamlit.exe run admin_dashboard.py --server.port 8502

pause
