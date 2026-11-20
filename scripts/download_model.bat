@echo off
REM =========================================================
REM  GestureFlow — Download MediaPipe Model (Windows)
REM =========================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo [GestureFlow] Downloading hand_landmarker.task model...
.venv\Scripts\python.exe scripts\download_model.py

endlocal
