@echo off
REM =========================================================
REM  GestureFlow — Launch Script (Windows)
REM =========================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run scripts\install.bat first.
    exit /b 1
)

.venv\Scripts\python.exe main.py %*
endlocal
