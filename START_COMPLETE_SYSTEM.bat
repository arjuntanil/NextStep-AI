@echo off
REM ========================================
REM NextStepAI - Complete System with Login Portal
REM ========================================
echo.
echo ========================================
echo   NextStepAI - Starting All Systems
echo ========================================
echo.

cd /d E:\NextStepAI

REM Start Backend API
echo [1/4] Starting Backend API (Port 8000)...
start "Backend API - Port 8000" cmd /k "cd /d E:\NextStepAI && call .\career_coach\Scripts\activate.bat && python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 >nul

REM Start Login Portal
echo [2/4] Starting Login Portal (Port 8500)...
start "Login Portal - Port 8500" cmd /k "cd /d E:\NextStepAI && call .\career_coach\Scripts\activate.bat && streamlit run login_portal.py --server.port 8500"
timeout /t 3 >nul

REM Start User App
echo [3/4] Starting User App (Port 8501)...
start "User App - Port 8501" cmd /k "cd /d E:\NextStepAI && call .\career_coach\Scripts\activate.bat && streamlit run app.py"
timeout /t 3 >nul

REM Start Admin Dashboard
echo [4/4] Starting Admin Dashboard (Port 8502)...
start "Admin Dashboard - Port 8502" cmd /k "cd /d E:\NextStepAI && call .\career_coach\Scripts\activate.bat && streamlit run admin_dashboard.py --server.port 8502"
timeout /t 2 >nul

echo.
echo ========================================
echo    All Systems Started!
echo ========================================
echo.
echo [LOGIN PORTAL]    http://localhost:8500  ^<-- START HERE!
echo [USER APP]        http://localhost:8501
echo [ADMIN DASHBOARD] http://localhost:8502
echo [API DOCS]        http://localhost:8000/docs
echo.
echo ========================================
echo   How It Works:
echo ========================================
echo.
echo 1. Go to http://localhost:8500 (Login Portal)
echo 2. Enter your credentials:
echo    - Admin: admin@gmail.com / admin
echo    - User:  your registered credentials
echo 3. System will automatically redirect you to:
echo    - Admins  --^> Admin Dashboard (port 8502)
echo    - Users   --^> User App (port 8501)
echo.
echo Press any key to exit (windows will remain open)...
pause
