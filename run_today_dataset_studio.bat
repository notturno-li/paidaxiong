@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
python -m dataset_studio.server --host 0.0.0.0 --port 8765 --camera realsense --config configs/today.yaml --project today_shape --classes 六棱柱,正方体,圆柱,平行四边形,长方体,大圆柱,梯形
if errorlevel 1 pause
endlocal
