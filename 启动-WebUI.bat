@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo [1/4] 检查依赖...
python -m pip install -r requirements-web.txt
if errorlevel 1 (
  echo 依赖安装失败，请检查 Python/pip 环境。
  pause
  exit /b 1
)

echo [2/4] 关闭旧的 5050 端口服务...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5050" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F >nul 2>nul
)

echo [3/4] 启动最新 WebUI 服务...
start "Gemini-Watermark-Remover" cmd /c "python web_app.py"

echo [4/4] 打开浏览器...
timeout /t 2 >nul
start "" "http://127.0.0.1:5050/"

echo 已启动（自动重启旧服务并加载最新代码）。
exit /b 0
