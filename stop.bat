@echo off
title Stopping SENTINEL-CV Surveillance Server
echo ===============================================================
echo     Stopping SENTINEL-CV Server on port 8000...
echo ===============================================================
echo.

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Terminating server process PID %%a listening on port 8000...
    taskkill /F /PID %%a >nul 2>&1
    set "FOUND=1"
)

echo.
if "%FOUND%"=="1" (
    echo [SUCCESS] SENTINEL-CV server stopped successfully.
) else (
    echo [INFO] No active server process found on port 8000.
)
echo.
pause

