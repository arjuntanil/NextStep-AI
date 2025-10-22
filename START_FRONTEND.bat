@echo off
echo ========================================
echo   Starting NextStepAI Frontend
echo ========================================
echo.

cd /d E:\NextStepAI

echo Activating virtual environment...
call E:\NextStepAI\career_coach\Scripts\activate.bat

echo.
echo Starting Streamlit frontend on http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

E:\NextStepAI\career_coach\Scripts\streamlit.exe run app.py

pause
