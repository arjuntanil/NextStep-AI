@echo off
echo ============================================================
echo   AI CAREER ADVISOR - COMPLETE STARTUP SCRIPT
echo   New LLM_FineTuned Model - Optimized for Speed
echo ============================================================
echo.

REM Change to project directory
cd /d E:\NextStepAI

echo [STEP 1] Checking if backend is already running...
netstat -ano | findstr :8000 >nul
if %errorlevel%==0 (
    echo [WARNING] Port 8000 already in use!
    echo [ACTION] Killing existing process...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    echo [OK] Port 8000 freed
    timeout /t 2 >nul
) else (
    echo [OK] Port 8000 is available
)

echo.
echo [STEP 2] Testing new LLM model (optional, press CTRL+C to skip)...
timeout /t 3
call career_coach\Scripts\Activate.ps1
python TEST_NEW_MODEL.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Model test failed, but continuing anyway...
    echo [INFO] Backend will try to use fallback model if needed
    timeout /t 3
)

echo.
echo ============================================================
echo   READY TO START!
echo ============================================================
echo.
echo You need to open TWO separate PowerShell terminals:
echo.
echo TERMINAL 1 - Backend API:
echo   cd E:\NextStepAI
echo   .\career_coach\Scripts\Activate.ps1
echo   python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
echo.
echo TERMINAL 2 - Frontend (wait for backend to finish loading):
echo   cd E:\NextStepAI
echo   .\career_coach\Scripts\Activate.ps1
echo   streamlit run app.py
echo.
echo Then open: http://localhost:8501
echo.
echo ============================================================
echo   OR USE THESE QUICK COMMANDS:
echo ============================================================
echo.
echo Backend (copy this entire line):
echo cd E:\NextStepAI ; .\career_coach\Scripts\Activate.ps1 ; python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
echo.
echo Frontend (copy this entire line):
echo cd E:\NextStepAI ; .\career_coach\Scripts\Activate.ps1 ; streamlit run app.py
echo.
echo ============================================================
echo Press any key to exit...
pause >nul
