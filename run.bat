@echo off
rem Start Group Diversifier on Windows (double-click or run from a terminal).
rem First run: creates .venv and installs the requirements, then opens the app.
rem Needs either uv (https://docs.astral.sh/uv/) or Python 3.12+ (py launcher).
rem The app has no login: it is meant for one person on this computer.
rem Extra arguments go to "streamlit run", e.g.  run.bat --server.port 8600
rem Set GD_SETUP_ONLY=1 to create the environment and stop before starting.
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\requirements.installed"

if exist "%PY%" goto have_venv
echo Creating the virtual environment in %VENV% ...
where uv >nul 2>nul
if not errorlevel 1 (
    uv venv "%VENV%" --python ">=3.12,<3.14"
    if errorlevel 1 goto failed
    goto have_venv
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv "%VENV%"
) else (
    python -m venv "%VENV%"
)
if errorlevel 1 goto failed

:have_venv
rem Install (again) when requirements.txt is new or has changed.
fc /b requirements.txt "%STAMP%" >nul 2>nul
if not errorlevel 1 goto start
echo Installing requirements ...
where uv >nul 2>nul
if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements.txt
) else (
    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
)
if errorlevel 1 goto failed
copy /y requirements.txt "%STAMP%" >nul

:start
if "%GD_SETUP_ONLY%"=="1" (
    echo Environment ready. Start the app with run.bat
    exit /b 0
)
"%PY%" -m streamlit run app/main.py %*
exit /b %errorlevel%

:failed
echo.
echo Setup failed. Python 3.12 or newer is required: install it from python.org
echo or install uv from https://docs.astral.sh/uv/ and run this file again.
pause
exit /b 1
