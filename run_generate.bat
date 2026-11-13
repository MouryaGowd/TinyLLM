@echo off
rem Double-click to generate sample text. Optional: run_generate.bat --prompt "ROMEO:" --temperature 0.7
cd /d "%~dp0"
call setup.bat
if errorlevel 1 exit /b 1

if not exist "ckpt.pt" (
    echo ckpt.pt not found. Run train.bat first, or copy a trained ckpt.pt into this folder.
    pause
    exit /b 1
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" generate.py --prompt "ROMEO:" --tokens 500
) else (
    ".venv\Scripts\python.exe" generate.py %*
)
echo.
pause
