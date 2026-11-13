@echo off
rem Double-click to (re)train the model. Overwrites ckpt.pt.
rem Optional: train.bat --max_iters 500   (quick test run)
cd /d "%~dp0"
call setup.bat
if errorlevel 1 exit /b 1

echo.
echo Training uses an NVIDIA GPU if the CUDA build of PyTorch is installed; otherwise the CPU.
echo On a CPU this is slow. Use "train.bat --max_iters 300" for a quick test.
echo.

".venv\Scripts\python.exe" train.py %*
echo.
pause
