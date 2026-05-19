@echo off
set "REPO=%~dp0.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "LOG_DIR=%REPO%\data\training"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "CMD_LOG=%LOG_DIR%\resume_yolo_v2_latest.log"

echo [%DATE% %TIME%] Starting SkyScouter YOLO v2 resume > "%CMD_LOG%"
cd /d "%REPO%"
"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\resume_yolo_v2_training.ps1" -Workers 8 >> "%CMD_LOG%" 2>&1
echo [%DATE% %TIME%] Training command exited with code %ERRORLEVEL% >> "%CMD_LOG%"
