@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m app.modular_main
if errorlevel 1 pause
