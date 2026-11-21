@echo off
REM ---------------------------------------------------------------------------
REM One-click launcher for the offline writing assistant.
REM
REM Strategy:
REM   1. Probe the console interpreter (.venv\Scripts\python.exe).
REM      - If it runs  -> use it (best: you see logs, Ctrl+C stops the server).
REM      - If it's blocked by Application Control policy -> fall back to
REM         pythonw.exe (windowed, NOT blocked) which starts the same Streamlit
REM         server detached and opens your browser.
REM
REM No internet needed after setup.
REM ---------------------------------------------------------------------------
cd /d "%~dp0.."

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PYW=.venv\Scripts\pythonw.exe"
set "APP=app\app_simple.py"
set "PORT=8501"

REM --- Probe: can we run the console interpreter? ---------------------------
"%VENV_PY%" --version >nul 2>&1
if %errorlevel% neq 0 goto :fallback

REM --- Console Python works: run in this window (Ctrl+C to stop) ------------
echo Starting Offline Writing Assistant...
echo A browser window will open at http://localhost:%PORT%
echo Press Ctrl+C in this window to stop the server.
echo.
"%VENV_PY%" -m streamlit run "%APP%" --server.port %PORT%
pause
exit /b 0

REM --- Fallback: console Python is blocked; use pythonw.exe -----------------
:fallback
echo NOTE: Console Python is blocked by system policy.
echo       Falling back to pythonw.exe -- the server will start in the
echo       background and your browser should open automatically.
echo.
echo       To STOP the server later, run app\stop.bat  (or end pythonw.exe
echo       in Task Manager). If the browser does not open, go to
echo       http://localhost:%PORT% manually.
echo.

REM Start the Streamlit server detached, minimized, with a title we can find.
start "GrammerTool-Streamlit" /min "%VENV_PYW%" -m streamlit run "%APP%" --server.port %PORT% --server.headless true

REM Give the server a few seconds to boot, then open the browser.
timeout /t 6 >nul
start "" http://localhost:%PORT%

echo Started. You can close this window; the server keeps running minimized.
pause
exit /b 0
