@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m dataset_studio.server --host 0.0.0.0 --port 8765 --camera opencv --camera-index 0
if errorlevel 1 pause
