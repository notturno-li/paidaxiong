@echo off
chcp 65001 >nul
cd /d "%~dp0"
python tools\probe_dvs.py --discover
pause
