@echo off
REM build.bat — Build the Autonomous Engine desktop app on Windows
REM
REM Requirements:
REM   pip install pywebview pyinstaller
REM   On Windows, pywebview uses WebView2 (built into Windows 11, free download for Win10)
REM
REM Output: dist\AutonomousEngine\AutonomousEngine.exe

setlocal enabledelayedexpansion

echo.
echo =====================================================
echo  Autonomous AI Engine — Desktop Build (Windows)
echo =====================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    exit /b 1
)

REM Install/upgrade build deps
echo [1/4] Installing build dependencies...
pip install pywebview pyinstaller --quiet --upgrade
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

REM Clean previous build
echo [2/4] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist\AutonomousEngine rmdir /s /q dist\AutonomousEngine

REM Run PyInstaller
echo [3/4] Building executable (this takes 2-5 minutes)...
pyinstaller autonomous_engine.spec --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

REM Copy .env.example if no .env exists in dist
echo [4/4] Finalising...
if not exist dist\AutonomousEngine\.env (
    if exist env.example (
        copy env.example dist\AutonomousEngine\.env.example >nul
        echo NOTE: Copy dist\AutonomousEngine\.env.example to dist\AutonomousEngine\.env and fill in your API keys
    )
)

echo.
echo =====================================================
echo  Build complete!
echo  Executable: dist\AutonomousEngine\AutonomousEngine.exe
echo.
echo  To run:  dist\AutonomousEngine\AutonomousEngine.exe
echo  To ship: zip the entire dist\AutonomousEngine\ folder
echo =====================================================
echo.
