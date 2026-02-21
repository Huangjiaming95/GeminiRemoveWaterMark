@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m pip install -r requirements-web.txt
python web_app.py
pause
