@echo off
rem One-time setup: create .venv and install dependencies.
rem Double-click this file once after downloading, then use start_ui.bat.
rem
rem Pure ASCII only; all Chinese messages live in scripts\install.py.
rem Keep every line free of ( ) characters -- cmd pre-parses block bodies
rem as a whole and a stray paren breaks the whole file.
setlocal
cd /d "%~dp0"

set "PYEXE="

rem --- find a usable Python 3.12+: launcher, then PATH, then common paths ---
where py >nul 2>nul
if errorlevel 1 goto check_path_python
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 goto check_path_python
set "PYEXE=py -3.12"
goto have_python

:check_path_python
where python >nul 2>nul
if errorlevel 1 goto check_known_paths
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul
if errorlevel 1 goto check_known_paths
set "PYEXE=python"
goto have_python

:check_known_paths
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYEXE goto have_python
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if defined PYEXE goto have_python
if exist "C:\Python312\python.exe" set "PYEXE=C:\Python312\python.exe"
if defined PYEXE goto have_python
if exist "C:\Python313\python.exe" set "PYEXE=C:\Python313\python.exe"
if defined PYEXE goto have_python

rem --- no usable Python found: try to install it automatically ---
echo.
echo   No usable Python 3.12+ found. This tool will try to install it for you.
echo.

rem --- first try winget ---
where winget >nul 2>nul
if errorlevel 1 goto try_download
echo   Installing Python 3.12 via winget. This may take a few minutes...
winget install --id Python.Python.3.12 --exact --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto try_download
call :find_python_after_install
if defined PYEXE goto have_python
goto manual

:try_download
echo   winget not available or failed. Downloading the Python installer...
set "PYINSTALLER=%TEMP%\python-3.12.10-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '%TEMP%\python-3.12.10-amd64.exe' -TimeoutSec 300 -UseBasicParsing; Write-Output ok } catch { Write-Output fail }" >nul
if not exist "%PYINSTALLER%" powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -Uri 'https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '%TEMP%\python-3.12.10-amd64.exe' -TimeoutSec 300 -UseBasicParsing; Write-Output ok } catch { Write-Output fail }" >nul
if not exist "%PYINSTALLER%" goto manual
echo   Installing Python 3.12 silently. This may take a few minutes...
"%PYINSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1 Shortcuts=0
if errorlevel 1 goto manual
call :find_python_after_install
if defined PYEXE goto have_python
goto manual

:find_python_after_install
set "PYEXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYEXE exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if defined PYEXE exit /b 0
if exist "C:\Python312\python.exe" set "PYEXE=C:\Python312\python.exe"
if defined PYEXE exit /b 0
if exist "C:\Python313\python.exe" set "PYEXE=C:\Python313\python.exe"
if defined PYEXE exit /b 0
exit /b 0

:manual
echo.
echo   Automatic install did not finish. Please install Python 3.12+ manually:
echo   https://www.python.org/downloads/
echo   Check "Add python.exe to PATH" during install, then run this file again.
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