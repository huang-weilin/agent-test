@echo off
rem Double-click this file to open the sandbox UI. No commands to type.
rem
rem This file is deliberately pure ASCII. cmd.exe reads .bat files by byte
rem offset, and non-ASCII bytes make that offset drift: a 2KB GBK batch file
rem was measured getting chopped up from line 6 onward, reporting
rem "'xx' is not recognized as an internal or external command" -- which
rem points the reader at a problem that does not exist. All the Chinese
rem messages live in scripts\start_ui.py instead, where Python controls the
rem encoding. Do not add non-ASCII text to this file.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo   .venv not found: %PY%
    echo   The virtualenv has not been created yet, or it was moved.
    echo   Please send a screenshot of this window to me.
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0scripts\start_ui.py"

rem If it failed, keep the window open so the message can actually be read.
if errorlevel 1 pause
exit /b 0
