@echo off
setlocal
cd /d "%~dp0"
echo Piano2MIDI setup
echo.
where ffmpeg >nul 2>nul || echo WARNING: FFmpeg not found on PATH. Install it with:  winget install ffmpeg

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
echo.
echo Installing PyTorch (CUDA 12.8). This is a large download.
".venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
echo.
echo Setup done. Double-click run.bat to start.
pause
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
