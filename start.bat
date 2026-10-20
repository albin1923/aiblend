@echo off
title SENTINEL-CV AI Surveillance System
echo ===============================================================
echo     SENTINEL-CV ^| AI Shoplifting Detection Command Center
echo ===============================================================
echo.

cd /d "%~dp0"

echo [1/2] Verifying Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please make sure Python 3.10+ is installed and added to PATH.
    pause
    exit /b 1
)

echo [2/2] Starting SENTINEL-CV server on http://localhost:8000 ...
start "" http://localhost:8000
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
