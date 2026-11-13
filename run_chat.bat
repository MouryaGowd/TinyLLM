@echo off
rem Double-click to chat with the trained model.
cd /d "%~dp0"
call setup.bat
if errorlevel 1 exit /b 1

if not exist "ckpt.pt" (
    echo ckpt.pt not found. Run train.bat first, or copy a trained ckpt.pt into this folder.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" chat.py %*
echo.
pause
