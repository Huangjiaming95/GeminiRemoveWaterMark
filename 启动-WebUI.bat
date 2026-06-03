@echo off
setlocal
cd /d "%~dp0"

for %%f in ("%~dp0*.bat") do (
    if /I not "%%~nxf"=="%~nx0" (
        call "%%~ff"
        exit /b %errorlevel%
    )
)

echo [ERROR] Main startup script was not found.
pause
exit /b 1
