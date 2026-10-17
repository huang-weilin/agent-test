@echo off
rem One-time setup: create .venv and install dependencies.
rem Double-click this file once after downloading, then use start_ui.bat.
rem
rem Pure ASCII only; all Chinese messages live in scripts\install.py.
rem Keep every line free of ( ) characters -- cmd pre-parses block bodies
rem as a whole and a stray paren breaks the whole file.
setlocal
cd /d "%~dp0"

rem Prefer the Python launcher, fall back to PATH python.
set "PYEXE="
where py >nul 2>nul
if not errorlevel 1 set "PYEXE=py -3.12"
if defined PYEXE goto have_python
where python >nul 2>nul
if not errorlevel 1 set "PYEXE=python"
if defined PYEXE goto have_python

echo.
echo   Python 3.12+ was not found on this machine.
echo   Install it from https://www.python.org/downloads/
echo   and check "Add python.exe to PATH", then run this file again.
echo.
pause
exit /b 1

:have_python
echo.
echo   Setting up the sandbox for the first time...
echo   This creates .venv and installs dependencies. It may take a few minutes.
echo.
%PYEXE% "%~dp0scripts\install.py"
if errorlevel 1 goto failed
echo.
echo   Done. Double-click start_ui.bat to open the UI.
pause
exit /b 0

:failed
echo.
echo   Setup did not finish. Read the message above and fix it, then retry.
pause
exit /b 1
