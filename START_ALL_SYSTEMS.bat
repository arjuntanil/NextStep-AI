@echo off
REM ========================================
REM NextStepAI - Complete System Startup
REM ========================================
echo.
echo ========================================
echo    NextStepAI - Starting All Systems
echo ========================================
echo.

REM Check if already running
tasklist /FI "WINDOWTITLE eq Backend API*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] Backend already running
) else (
    echo [1/3] Starting Backend API (Port 8000)...
    start "Backend API - Port 8000" cmd /k "cd /d %~dp0 && .\career_coach\Scripts\activate.bat && python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload"
    timeout /t 3 >nul
)

tasklist /FI "WINDOWTITLE eq User App*" 2>NUL | find /I /N "streamlit.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] User App already running
) else (
    echo [2/3] Starting User App (Port 8501)...
    start "User App - Port 8501" cmd /k "cd /d %~dp0 && .\career_coach\Scripts\activate.bat && streamlit run app.py"
    timeout /t 3 >nul
)

tasklist /FI "WINDOWTITLE eq Admin Dashboard*" 2>NUL | find /I /N "streamlit.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [INFO] Admin Dashboard already running
) else (
    echo [3/3] Starting Admin Dashboard (Port 8502)...
    start "Admin Dashboard - Port 8502" cmd /k "cd /d %~dp0 && .\career_coach\Scripts\activate.bat && streamlit run admin_dashboard.py --server.port 8502"
    timeout /t 2 >nul
)

echo.
echo ========================================
echo    All Systems Started!
echo ========================================
echo.
echo [USER APP]        http://localhost:8501
echo [ADMIN DASHBOARD] http://localhost:8502
echo [API DOCS]        http://localhost:8000/docs
echo.
echo [ADMIN LOGIN]
echo   Email:    admin@gmail.com
echo   Password: admin
echo.
echo Press any key to open all URLs in browser...
pause >nul

start http://localhost:8501
start http://localhost:8502
start http://localhost:8000/docs

echo.
echo All systems are running! Close this window when done.
echo.
pause
