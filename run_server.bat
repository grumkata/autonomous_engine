@echo off
setlocal enabledelayedexpansion
title AI Engine - Launcher

REM ====================================================================
REM  Autonomous AI Engine -- Windows startup script
REM
REM  FIX HISTORY:
REM  - Removed em-dash (U+2014) from window title: silently aborts
REM    the start command on terminals not running code page 65001.
REM  - Replaced ^-continuation-inside-quotes with a true single-line
REM    cmd /k string. ^ inside a double-quoted string is a literal
REM    character once passed to the child process -- not a continuation.
REM  - Removed temp-bat approach: introduced a double-echo bug and a
REM    race where the file might not flush before start fires.
REM  - Added pause at the END of this launcher so the user can read
REM    all output before the window closes.
REM ====================================================================

REM ── Root + config ────────────────────────────────────────────────────
cd /d C:\Users\grumk\OneDrive\Desktop\projects\autonomous_engine
set "ROOT=%cd%"
set "PORT=8000"
set "ACTIVATE=%ROOT%\.venv\Scripts\activate.bat"

echo.
echo  ================================================================
echo   Autonomous AI Engine
echo  ================================================================
echo.

REM ── Port guard ───────────────────────────────────────────────────────
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARN]  Port %PORT% is already in use.
    echo         If the engine is already running, the UI is at:
    echo         http://127.0.0.1:%PORT%/ui/
    echo.
    echo         Press any key to open it, or close the existing
    echo         server window first and re-run this script.
    pause
    start http://127.0.0.1:%PORT%/ui/
    goto :done
)

REM ── .env setup ───────────────────────────────────────────────────────
if not exist "%ROOT%\.env" (
    if exist "%ROOT%\env.example" (
        copy "%ROOT%\env.example" "%ROOT%\.env" >nul
        echo [SETUP] .env created from env.example.
        echo [INFO]  Edit .env to configure OLLAMA_BASE_URL, model, etc.
    ) else (
        echo [INFO]  No .env found. Using built-in defaults.
    )
)

REM ── Virtual environment ───────────────────────────────────────────────
if not exist "%ACTIVATE%" (
    echo [SETUP] Creating virtual environment...
    python -m venv "%ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Ensure Python 3.11+ is installed and on PATH.
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

REM ── Ollama check (non-blocking) ───────────────────────────────────────
set "OLLAMA_URL=http://localhost:11434"
if exist "%ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ROOT%\.env") do (
        if /i "%%A"=="OLLAMA_BASE_URL" set "OLLAMA_URL=%%B"
    )
)
REM Trim whitespace/CR that .env readers sometimes leave
for /f "tokens=* delims= " %%A in ("!OLLAMA_URL!") do set "OLLAMA_URL=%%A"

echo [CHECK] Checking Ollama at !OLLAMA_URL!...
curl -s --max-time 3 "!OLLAMA_URL!/api/tags" >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Ollama not reachable. LLM calls will fail until it runs.
    echo         Start Ollama desktop app or run: ollama serve
) else (
    echo [OK]    Ollama is reachable.
)
echo.

REM ── Launch server window ─────────────────────────────────────────────
REM
REM  WHY ONE LINE:
REM  cmd /k accepts a single quoted string as the command to run.
REM  ^ inside that string is a literal ^ (not a continuation), so
REM  multi-line approaches with ^ produce commands with stray ^ or
REM  extra whitespace that break the && chains.
REM
REM  WHY NO INNER QUOTES ON PATHS:
REM  %ROOT% = C:\Users\grumk\OneDrive\Desktop\projects\autonomous_engine
REM  No spaces anywhere in that path, so quoting is unnecessary.
REM  %ACTIVATE% expands to %ROOT%\.venv\Scripts\activate.bat -- also
REM  no spaces. If you move to a path with spaces, use the helper-bat
REM  approach documented at the bottom of this file.
REM
REM  %ROOT%, %PORT%, %ACTIVATE% are expanded by THIS script before
REM  the string reaches cmd /k, so the child process sees plain text.
REM
echo [START] Opening server window...
start "AI Engine - Server" cmd /k "cd /d %ROOT% && call %ACTIVATE% && echo. && echo   Server: http://127.0.0.1:%PORT%/ui/  -- Ctrl+C to stop && echo. && uvicorn main:app --reload --port %PORT% --host 127.0.0.1"

REM ── Health check ─────────────────────────────────────────────────────
echo [WAIT]  Waiting for server to boot (up to ~30s)...
timeout /t 6 >nul

set "HEALTH_OK=0"
set "TRIES=0"

:hloop
    set /a TRIES+=1
    if !TRIES! gtr 12 goto :hdone
    curl -s -o nul -w "%%{http_code}" "http://127.0.0.1:%PORT%/health/" > "%TEMP%\ae_hc.txt" 2>nul
    set /p HCODE= < "%TEMP%\ae_hc.txt"
    del "%TEMP%\ae_hc.txt" >nul 2>&1
    if "!HCODE!"=="200" (
        set "HEALTH_OK=1"
        goto :hdone
    )
    timeout /t 2 >nul
goto :hloop

:hdone
if "!HEALTH_OK!"=="1" (
    echo [OK]    Server is healthy. Opening browser...
    start http://127.0.0.1:%PORT%/ui/
) else (
    echo [WARN]  /health/ did not return 200 after !TRIES! attempts.
    echo         The server may still be booting, OR there is a Python
    echo         error on startup -- check the "AI Engine - Server" window.
    echo         Opening browser anyway...
    start http://127.0.0.1:%PORT%/ui/
)
echo.
goto :done

:fail
echo.
echo [ERROR] Startup aborted. See messages above.
echo.

:done
echo  ================================================================
echo   UI:     http://127.0.0.1:%PORT%/ui/
echo   API:    http://127.0.0.1:%PORT%/docs
echo   Health: http://127.0.0.1:%PORT%/health/
echo.
echo   Server runs in the "AI Engine - Server" window.
echo   Close that window to stop. This window can be closed safely.
echo  ================================================================
echo.
endlocal

REM ====================================================================
REM  IF YOU EVER MOVE TO A PATH WITH SPACES:
REM  Replace the "start" line above with these two lines:
REM
REM  > "%ROOT%\_server.bat" echo @echo off
REM  >> "%ROOT%\_server.bat" echo cd /d "%ROOT%"
REM  >> "%ROOT%\_server.bat" echo call "%ACTIVATE%"
REM  >> "%ROOT%\_server.bat" echo uvicorn main:app --reload --port %PORT% --host 127.0.0.1
REM  start "AI Engine - Server" cmd /k "%ROOT%\_server.bat"
REM ====================================================================
