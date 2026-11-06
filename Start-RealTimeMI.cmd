@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0build\venv\Scripts\python.exe" (
  "%~dp0build\venv\Scripts\python.exe" -m realtimemi %*
) else (
  py -3.12 -m realtimemi %*
)

if errorlevel 1 (
  echo.
  echo RealTimeMI could not start. See the error above.
  pause
)
