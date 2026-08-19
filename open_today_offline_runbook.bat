@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" notepad.exe "%~dp0TODAY_OFFLINE_RUNBOOK.md"

