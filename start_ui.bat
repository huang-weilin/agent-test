@echo off
rem Double-click this file to open the sandbox UI. No commands to type.
rem
rem This file is deliberately pure ASCII. cmd.exe reads .bat files by byte
rem offset, and non-ASCII bytes make that offset drift. All Chinese messages
rem live in scripts\start_ui.py instead, where Python controls the encoding.
rem Do not add non-ASCII text to this file.
rem
rem NOTE: keep every line free of ( ) characters outside plain commands.
rem cmd.exe pre-parses if-blocks as a whole and a stray paren inside a
rem block -- even inside echo text -- breaks the whole file with
rem "was unexpected at this time".
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

if exist "%PY%" goto run

rem First run: no .venv yet, set it up once.
echo.
echo   First run detected. Setting up dependencies...
echo   This may take a few minutes the first time.
echo.
call install.bat
if errorlevel 1 goto setup_failed

if not exist "%PY%" goto setup_failed

:run
"%PY%" "%~dp0scripts\start_ui.py"
if errorlevel 1 goto run_failed
exit /b 0

:setup_failed
echo.
echo   Setup did not finish. Run install.bat manually and follow the messages.
pause
exit /b 1

:run_failed
echo.
echo   The UI stopped with an error. Read the message above, fix it, then run again.
pause
exit /b 1
