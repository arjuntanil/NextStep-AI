@echo off
echo ====================================
echo Starting React Frontend
echo ====================================
cd frontend
echo Installing dependencies...
call npm install
echo.
echo Starting development server...
call npm start
