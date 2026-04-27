@echo off
setlocal enabledelayedexpansion
title Autonomous AI Engine

REM ====================================================================
REM  Autonomous AI Engine -- run.bat
REM
REM  Does everything run_server.bat used to do:
REM    - Creates .venv if missing
REM    - Installs / updates all requirements
REM    - Creates .env from env.example if missing
REM    - Checks Ollama (non-blocking warning)
REM  Then launches the desktop app (launcher.py) instead of a browser.
REM ====================================================================

cd /d C:\Users\grumk\OneDrive\Desktop\projects\autonomous_engine
set "ROOT=%cd%"
set "ACTIVATE=%ROOT%\.venv\Scripts\activate.bat"

echo.
echo  ================================================================
echo   Autonomous AI Engine
echo  ================================================================
echo.

REM ── .env setup ───────────────────────────────────────────────────────
if not exist "%ROOT%\.env" (
    if exist "%ROOT%\env.example" (
        copy "%ROOT%\env.example" "%ROOT%\.env" >nul
        echo [SETUP] .env created from env.example.
        echo [INFO]  Open .env and fill in your API keys before using LLMs.
        echo.
    ) else (
        echo [INFO]  No .env found - using built-in defaults.
    )
)

REM ── Virtual environment ───────────────────────────────────────────────
if not exist "%ACTIVATE%" (
    echo [SETUP] Creating virtual environment...
    python -m venv "%ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Make sure Python 3.11+ is installed and on PATH.
        goto :fail
    )
    echo [OK]    Virtual environment created.
)

call "%ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto :fail
)

REM ── Dependencies ─────────────────────────────────────────────────────
echo [DEPS]  Upgrading pip...
python -m pip install --upgrade pip --quiet 2>nul

echo [DEPS]  Installing/updating requirements...
pip install -r "%ROOT%\requirements.txt" --quiet --upgrade
if errorlevel 1 (
    echo [WARN]  Some packages may have failed. Continuing anyway...
    timeout /t 3 >nul
) else (
    echo [OK]    All requirements satisfied.
)

REM ── pywebview check ───────────────────────────────────────────────────
python -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo [DEPS]  Installing pywebview ^(needed for desktop window^)...
    pip install pywebview --quiet
    if errorlevel 1 (
        echo [WARN]  pywebview install failed - will fall back to browser.
    ) else (
        echo [OK]    pywebview installed.
    )
)

REM ── Ollama check (non-blocking) ───────────────────────────────────────
set "OLLAMA_URL=http://localhost:11434"
if exist "%ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ROOT%\.env") do (
        if /i "%%A"=="OLLAMA_BASE_URL" set "OLLAMA_URL=%%B"
    )
)
for /f "tokens=* delims= " %%A in ("!OLLAMA_URL!") do set "OLLAMA_URL=%%A"

echo [CHECK] Checking Ollama at !OLLAMA_URL!...
curl -s --max-time 3 "!OLLAMA_URL!/api/tags" >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Ollama not reachable. LLM calls will fail until it starts.
    echo         Run: ollama serve
) else (
    echo [OK]    Ollama is reachable.
)
echo.

REM ── Launch ────────────────────────────────────────────────────────────
echo [START] Launching desktop app...
echo.
python "%ROOT%\launcher.py"
goto :done

:fail
echo.
echo [ERROR] Startup failed. See messages above.
echo.
pause
exit /b 1

:done
endlocal
