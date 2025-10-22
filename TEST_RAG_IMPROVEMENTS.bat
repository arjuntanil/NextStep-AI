@echo off
echo ========================================
echo NextStepAI - Quick Test RAG System
echo ========================================
echo.

echo [1/3] Testing improved skill extraction...
call E:\NextStepAI\career_coach\Scripts\python.exe test_improved_skill_extraction.py
echo.

echo ========================================
echo Expected Result:
echo - Only 4 skills missing: Azure, CI/CD, OOP, Problem-Solving
echo - All skills from your resume properly matched
echo ========================================
echo.

pause
