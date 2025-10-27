@echo off
echo ====================================
echo NextStepAI React Frontend Setup
echo ====================================
echo.
echo This script will:
echo 1. Install Node.js dependencies
echo 2. Verify installation
echo 3. Prepare the React app
echo.
echo Please wait...
echo ====================================
echo.

cd frontend

echo Installing npm packages...
call npm install

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ====================================
    echo ERROR: Installation failed!
    echo ====================================
    echo.
    echo Please ensure:
    echo 1. Node.js is installed (v16 or higher)
    echo 2. You have internet connection
    echo 3. No firewall blocking npm
    echo.
    echo Download Node.js from: https://nodejs.org/
    echo ====================================
    pause
    exit /b 1
)

echo.
echo ====================================
echo Installation completed successfully!
echo ====================================
echo.
echo Next steps:
echo 1. Run START_REACT_SYSTEM.bat to start everything
echo    OR
echo 2. Run these commands separately:
echo    - START_BACKEND.bat (in one terminal)
echo    - START_REACT_FRONTEND.bat (in another terminal)
echo.
echo Your React app will be available at:
echo http://localhost:3000
echo.
echo Backend API will be at:
echo http://127.0.0.1:8000
echo ====================================
pause
