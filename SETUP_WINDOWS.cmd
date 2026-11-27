@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
if errorlevel 1 goto failed
echo Setup complete. Run RUN_DEMO.cmd to see four examples.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.12 and check internet connectivity.
pause
exit /b 1
