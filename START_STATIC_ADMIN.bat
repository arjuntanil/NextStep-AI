@echo off
echo ========================================
echo   Starting Static Admin Dashboard
echo ========================================
echo.

cd /d E:\NextStepAI

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo.
echo Starting Static Admin Dashboard on http://localhost:8503
echo.
echo Admin Login Credentials:
echo   Email:    admin@gmail.com
echo   Password: admin
echo.
echo Features:
echo   - 5 Static Users
echo   - Interactive Charts
echo   - User Management
echo   - Analytics Dashboard
echo   - Clean Professional Design
echo.
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\streamlit.exe run static_admin_dashboard.py --server.port 8503

pause
