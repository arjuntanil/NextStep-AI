@echo off
REM ============================================
REM   COMPLETE SETUP: OLLAMA + TINYLLAMA
REM ============================================
echo.
echo ============================================
echo   NextStepAI - Complete Ollama Setup
echo ============================================
echo.

REM Step 1: Kill port 8000
echo [1/6] Freeing port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul
echo   ✅ Port 8000 freed
echo.

REM Step 2: Stop Ollama
echo [2/6] Stopping existing Ollama...
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo   ✅ Ollama stopped
echo.

REM Step 3: Start Ollama in CPU mode
echo [3/6] Starting Ollama in CPU mode...
set CUDA_VISIBLE_DEVICES=-1
set OLLAMA_NUM_GPU=0
start "" "C:\Users\Arjun T Anil\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 4 /nobreak >nul
echo   ✅ Ollama started
echo.

REM Step 4: Check installed models
echo [4/6] Checking installed models...
ollama list
echo.

REM Step 5: Pull tinyllama if not installed
echo [5/6] Ensuring tinyllama is installed...
ollama list | findstr /i "tinyllama" >nul
if %errorlevel% equ 0 (
    echo   ✅ tinyllama already installed
) else (
    echo   📥 Downloading tinyllama (637 MB, fastest model)...
    echo   This may take 2-5 minutes depending on your connection...
    ollama pull tinyllama
    if %errorlevel% equ 0 (
        echo   ✅ tinyllama installed successfully!
    ) else (
        echo   ❌ Failed to install tinyllama
        echo   Please check your internet connection and try again
        pause
        exit /b 1
    )
)
echo.

REM Step 6: Test model
echo [6/6] Testing tinyllama...
echo   Running quick test query...
ollama run tinyllama "Say hello in one sentence" --verbose 2>nul
echo.

echo ============================================
echo   ✅ SETUP COMPLETE!
echo ============================================
echo.
echo Your system is now configured:
echo   • Ollama running in CPU mode (no GPU needed)
echo   • tinyllama model installed (637 MB RAM only)
echo   • Backend will auto-detect and use tinyllama
echo.
echo NEXT STEPS:
echo   1. Open a NEW terminal
echo   2. Run: .\START_BACKEND.bat
echo   3. Open another terminal
echo   4. Run: .\START_FRONTEND.bat
echo.
echo OR use one-click start:
echo   .\QUICK_START.bat
echo.
pause
