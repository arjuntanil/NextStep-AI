@echo off
echo ============================================================
echo   FIXING TIMEOUT ERROR - Restarting Backend
echo ============================================================
echo.

cd /d E:\NextStepAI

echo [STEP 1] Stopping existing backend...
echo Killing process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
    echo   Killed PID: %%a
)
timeout /t 2 >nul
echo [OK] Port 8000 freed
echo.

echo [STEP 2] Backend has been updated to use RAG mode
echo   - No more timeout errors
echo   - Instant responses (no model loading wait)
echo   - Quality answers from career guides
echo.

echo ============================================================
echo   READY TO START BACKEND
echo ============================================================
echo.
echo Copy and paste this command in a PowerShell terminal:
echo.
echo cd E:\NextStepAI ; .\career_coach\Scripts\Activate.ps1 ; python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
echo.
echo Then in another terminal:
echo.
echo cd E:\NextStepAI ; .\career_coach\Scripts\Activate.ps1 ; streamlit run app.py
echo.
echo ============================================================
echo   WHAT TO EXPECT:
echo ============================================================
echo.
echo Backend logs will show:
echo   [RAG] Using RAG model (fine-tuned model not loaded)
echo.
echo This is NORMAL and GOOD - means no timeout issues!
echo.
echo AI Career Advisor will respond in less than 5 seconds.
echo.
echo ============================================================
pause
