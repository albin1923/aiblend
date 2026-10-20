@echo off
title Push SENTINEL-CV to GitHub
echo ===============================================================
echo     Pushing to GitHub: https://github.com/albin1923/aiblend.git
echo ===============================================================
echo.

cd /d "%~dp0"

echo Running: git push origin main ...
git push origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ===============================================================
    echo   [SUCCESS] All changes successfully pushed to GitHub!
    echo ===============================================================
) else (
    echo ===============================================================
    echo   [ERROR] Push failed.
    echo   If prompted, please sign in with your GitHub account.
    echo ===============================================================
)

echo.
pause
