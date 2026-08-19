@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 scripts\hand_eye_menu.py
if errorlevel 1 pause
