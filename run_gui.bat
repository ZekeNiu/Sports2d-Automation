@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=D:\Application\Anaconda\envs\sports3d\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Cannot find Python: %PYTHON_EXE%
  echo Please update run_gui.bat if your sports3d environment moved.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m sports2d_automation.gui
if errorlevel 1 pause
