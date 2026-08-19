@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist wheelhouse mkdir wheelhouse
python -m pip download --only-binary=:all: -r requirements_competition.txt -d wheelhouse
if errorlevel 1 (
  echo Offline dependency download failed. Keep the current environment and retry while online.
  pause
  exit /b 1
)
echo Offline dependencies are ready in wheelhouse.
pause
