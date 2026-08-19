@echo off
chcp 65001 >nul
cd /d "%~dp0"
python tools\preflight.py
if errorlevel 1 (
  echo Fix all blocking preflight items before creating the competition package.
  pause
  exit /b 1
)
if exist wheelhouse (
  python tools\build_field_package.py --include-wheelhouse
) else (
  python tools\build_field_package.py
)
pause
