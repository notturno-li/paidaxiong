@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist wheelhouse (
  echo wheelhouse is missing. Run prepare_offline_dependencies.bat on an online Windows computer first.
  pause
  exit /b 1
)
python -m pip install --no-index --find-links=wheelhouse -r requirements_competition.txt
pause
