@echo off
chcp 65001 >nul
cd /d "%~dp0"
python tools\mock_dvs_server.py --count 8
pause
