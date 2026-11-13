@echo off
rem Creates the virtual environment and installs dependencies on first run.
rem Called by run_chat.bat, run_generate.bat and train.bat. Safe to run repeatedly.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import torch" >nul 2>&1
    if not errorlevel 1 goto :ready
)

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then run this file again.
    echo.
    pause
    exit /b 1
)

echo First-time setup: creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo Installing PyTorch and dependencies. This downloads a few hundred MB and can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip

rem PyPI's Windows torch is CPU-only. If an NVIDIA GPU is present, try the CUDA build first.
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA GPU detected - installing the CUDA build of PyTorch...
    ".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 echo CUDA build unavailable - falling back to the CPU build.
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

:ready
exit /b 0
