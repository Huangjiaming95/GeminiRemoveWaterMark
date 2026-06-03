@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "LOG_DIR=logs"
set "LOG_FILE=%LOG_DIR%\webui.log"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "VENV_PY=.venv\Scripts\python.exe"
set "REBUILD_VENV=0"
if exist "%VENV_PY%" (
    "%VENV_PY%" -V >nul 2>nul
    if errorlevel 1 set "REBUILD_VENV=1"
) else (
    set "REBUILD_VENV=1"
)

if "%REBUILD_VENV%"=="1" (
    set "PY_CMD="
    where py >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
    if not defined PY_CMD (
        where python >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )

    if not defined PY_CMD (
        echo [ERROR] Python 3 was not found.
        echo Please install Python 3.10+ and run this script again.
        pause
        exit /b 1
    )

    if exist ".venv" (
        echo [INIT] Removing broken virtual environment...
        rmdir /s /q ".venv"
    )

    echo [INIT] Creating local virtual environment...
    call %PY_CMD% -m venv .venv || goto :venv_fail
)

echo [CHECK] Checking Python packages...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>nul
if exist "requirements-web.txt" (
    "%VENV_PY%" -m pip install -r requirements-web.txt || goto :pip_fail
) else (
    "%VENV_PY%" -c "import flask, cv2, numpy" >nul 2>nul
    if errorlevel 1 (
        echo [INIT] Installing required packages...
        "%VENV_PY%" -m pip install flask opencv-python numpy || goto :pip_fail
    )
)

echo [START] Closing process on port 5050 if needed...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5050" ^| findstr "LISTENING"') do taskkill /PID %%p /F >nul 2>nul

echo [START] Starting WebUI service...
start "NanoBanana-WebUI" cmd /c ""%VENV_PY%" web_app.py > "%LOG_FILE%" 2>&1"

echo [WAIT] Waiting for WebUI...
set "READY=0"
for /l %%i in (1,1,25) do (
    netstat -ano | findstr ":5050" | findstr "LISTENING" >nul
    if not errorlevel 1 (
        set "READY=1"
        goto :ready
    )
    timeout /t 1 >nul
)

:ready
if "%READY%"=="1" (
    echo [OK] WebUI is ready. Opening browser...
    start "" "http://127.0.0.1:5050/"
    exit /b 0
) else (
    echo [ERROR] WebUI did not start. Log: %LOG_FILE%
    if exist "%LOG_FILE%" (
        echo ===== Log tail =====
        powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Tail 80"
    )
    pause
    exit /b 1
)

:venv_fail
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:pip_fail
echo [ERROR] Failed to install required packages.
pause
exit /b 1
