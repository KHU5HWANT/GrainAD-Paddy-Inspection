@echo off
cd /d "%~dp0"
set "TORCH_HOME=%~dp0datasets\torch-cache"
set "MPLCONFIGDIR=%~dp0outputs\matplotlib"
if not exist .venv\Scripts\python.exe goto missing
if "%~1"=="" (
 .venv\Scripts\python.exe scripts\demo_grainad.py --device cpu
) else (
 .venv\Scripts\python.exe scripts\demo_grainad.py --device cpu --image "%~1"
)
if errorlevel 1 goto failed
start "" "outputs\demo_top5\summary.png"
pause
exit /b
:missing
echo Run SETUP_WINDOWS.cmd first.
pause
exit /b 1
:failed
echo Prediction failed. Read the error above.
pause
exit /b 1
