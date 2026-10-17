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
    rem First run: .venv is missing. Try one-time setup automatically;
    rem install.bat detects Python itself and explains failures.
    echo.
    echo   First run detected. Setting up dependencies (one time)...
    echo.
    call install.bat
    if errorlevel 1 (
        echo.
        echo   Setup did not finish. Run install.bat manually and follow the messages.
        pause
        exit /b 1
    )
)

if not exist "%PY%" (
    echo.
    echo   .venv still missing after setup. Run install.bat manually.
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0scripts\start_ui.py"

rem If it failed, keep the window open so the message can actually be read.
if errorlevel 1 pause
exit /b 0
