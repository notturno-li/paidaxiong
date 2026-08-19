@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
python -m app.today_main
if errorlevel 1 pause
endlocal
