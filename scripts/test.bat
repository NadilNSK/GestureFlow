@echo off
REM =========================================================
REM  GestureFlow — Run Test Suite (Windows)
REM =========================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

.venv\Scripts\python.exe -m pytest tests/ -v --tb=short %*
endlocal
