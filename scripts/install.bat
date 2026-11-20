@echo off
REM =========================================================
REM  GestureFlow — Install Script (Windows)
REM  Creates .venv, upgrades pip, installs requirements.
REM =========================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo [GestureFlow] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3.11+ is installed and on PATH.
    exit /b 1
)

echo [GestureFlow] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

echo [GestureFlow] Installing dependencies...
.venv\Scripts\pip.exe install -r requirements.txt

echo [GestureFlow] Installing gestureflow package...
.venv\Scripts\pip.exe install --no-build-isolation --no-deps -e .

echo.
echo [GestureFlow] Installation complete!
echo Run: scripts\download_model.bat   to fetch the hand landmark model.
echo Run: scripts\launch.bat           to start the app.
endlocal
