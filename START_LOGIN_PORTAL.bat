@echo off
echo ========================================
echo   Starting NextStepAI Login Portal
echo ========================================
echo.

cd /d E:\NextStepAI

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo.
echo Starting Login Portal on http://localhost:8500
echo.
echo This portal will automatically redirect:
echo   - Admin users to Admin Dashboard (port 8502)
echo   - Regular users to User App (port 8501)
echo.
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\streamlit.exe run login_portal.py --server.port 8500

pause
