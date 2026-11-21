@echo off
REM ---------------------------------------------------------------------------
REM Stop the Offline Writing Assistant server.
REM
REM Useful when app\run.bat had to fall back to pythonw.exe (detached server,
REM no Ctrl+C). Kills the Streamlit server bound to this app's port only.
REM ---------------------------------------------------------------------------
set "PORT=8501"

echo Looking for the Offline Writing Assistant server on port %PORT%...

powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if($c){$c|ForEach-Object{Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue};Write-Output 'Server stopped.'}else{Write-Output 'No server found on this port.'}"

echo.
pause
exit /b 0
