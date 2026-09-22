@echo off
rem ======================================================================
rem  Realtime Subtitles - one-click launcher
rem
rem    start_subtitles.bat         start servers + app, stop servers on exit
rem    start_subtitles.bat keep    leave the servers running after the app closes
rem
rem  Servers that were already running before this script started are never
rem  stopped by it. Server logs go to logs\asr_server*.log and
rem  logs\translator_server*.log.
rem ======================================================================
setlocal EnableExtensions
title Realtime Subtitles
cd /d "%~dp0"
set "ROOT=%~dp0"

set "ASR_URL=http://127.0.0.1:8090/health"
set "MT_URL=http://127.0.0.1:8081/health"
set "WAIT_LIMIT=180"
set "KEEP=0"
if /i "%~1"=="keep" set "KEEP=1"

rem winget installs llama-server here; make sure the servers can find it.
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"

echo.
echo  Realtime Subtitles
echo  ------------------

rem ---------------------------------------------------------- preflight
where python >nul 2>&1
if errorlevel 1 (
    echo  [x] Python was not found on PATH.
    goto :fail
)
set "MISSING=0"
for %%f in (Confucius4-R2T2-Q8_0.gguf mmproj-Confucius4-R2T2-f16.gguf Qwen3-4B-Instruct-2507-Q4_K_M.gguf) do (
    if not exist "%ROOT%models\%%f" (
        echo  [x] Missing model: models\%%f
        set "MISSING=1"
    )
)
if "%MISSING%"=="1" (
    echo      Run: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
    goto :fail
)

rem ------------------------------------------------------ start servers
set "STARTED_ASR=0"
set "STARTED_MT=0"

call :is_up "%ASR_URL%"
if errorlevel 1 (
    echo  [..] Starting speech recognition server ^(R2T2, port 8090^)
    call :launch run_asr_server asr_server
    set "STARTED_ASR=1"
) else (
    echo  [ok] Speech recognition server already running
)

call :is_up "%MT_URL%"
if errorlevel 1 (
    echo  [..] Starting translation server ^(Qwen3-4B, port 8081^)
    call :launch run_translator_server translator_server
    set "STARTED_MT=1"
) else (
    echo  [ok] Translation server already running
)

rem ------------------------------------------------- wait until healthy
echo  [..] Waiting for the models to load onto the GPU ^(about 20 seconds^)
set /a WAITED=0
:wait
call :is_up "%ASR_URL%"
if errorlevel 1 goto :not_yet
call :is_up "%MT_URL%"
if errorlevel 1 goto :not_yet
goto :ready

:not_yet
rem A server that crashed on startup will never answer; fail fast instead
rem of waiting out the full limit.
if "%STARTED_ASR%"=="1" (
    call :alive asr_server
    if errorlevel 1 goto :timeout
)
if "%STARTED_MT%"=="1" (
    call :alive translator_server
    if errorlevel 1 goto :timeout
)
set /a WAITED+=2
if %WAITED% geq %WAIT_LIMIT% goto :timeout
rem ping instead of timeout: timeout aborts when stdin is redirected.
ping -n 3 127.0.0.1 >nul
goto :wait

:ready
echo  [ok] Both servers are ready
echo.
echo  Opening the subtitle app. Close its control panel to quit.
echo  Ctrl+Shift+S hides/shows the subtitles.
echo.
python -m rtsubs
if errorlevel 1 (
    echo.
    echo  [x] The app exited with an error - see the messages above.
    pause
)
goto :cleanup

rem ------------------------------------------------------------ timeout
:timeout
echo  [x] A model server failed to start ^(or did not become ready within %WAIT_LIMIT% s^).
echo.
echo  --- last lines of logs\asr_server.err.log ---
powershell -NoProfile -Command "Get-Content -Tail 12 '%ROOT%logs\asr_server.err.log' -ErrorAction SilentlyContinue"
echo  --- last lines of logs\translator_server.err.log ---
powershell -NoProfile -Command "Get-Content -Tail 12 '%ROOT%logs\translator_server.err.log' -ErrorAction SilentlyContinue"
set "KEEP=0"
call :cleanup_servers
goto :fail

rem ------------------------------------------------------------ cleanup
:cleanup
if "%KEEP%"=="1" (
    echo  [ok] Leaving the servers running ^("keep" was given^)
    goto :end
)
call :cleanup_servers
goto :end

:cleanup_servers
if "%STARTED_ASR%"=="1" call :stop asr_server
if "%STARTED_MT%"=="1" call :stop translator_server
if "%STARTED_ASR%%STARTED_MT%"=="00" echo  [ok] Servers were already running before; leaving them up
exit /b 0

:fail
echo.
pause
exit /b 1

:end
endlocal
exit /b 0

rem ======================================================== subroutines

rem :is_up <url>  -> errorlevel 0 when the URL answers HTTP 200
:is_up
set "CODE="
for /f %%c in ('curl.exe -s -o nul -m 2 -w "%%{http_code}" "%~1" 2^>nul') do set "CODE=%%c"
if "%CODE%"=="200" exit /b 0
exit /b 1

rem :launch <script name> <log name>  -> starts a hidden server, records its PID
:launch
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process powershell.exe -WindowStyle Hidden -PassThru -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%ROOT%scripts\%~1.ps1' -RedirectStandardOutput '%ROOT%logs\%~2.log' -RedirectStandardError '%ROOT%logs\%~2.err.log'; $p.Id | Out-File -Encoding ascii '%ROOT%logs\%~2.pid'"
exit /b 0

rem :alive <log name>  -> errorlevel 0 while the launched server is still running
:alive
if not exist "%ROOT%logs\%~1.pid" exit /b 1
for /f "usebackq" %%p in ("%ROOT%logs\%~1.pid") do (
    tasklist /FI "PID eq %%p" /NH 2>nul | find "%%p" >nul
    exit /b
)
exit /b 1

rem :stop <log name>  -> kills the server started by :launch (and its children)
:stop
if not exist "%ROOT%logs\%~1.pid" exit /b 0
for /f "usebackq" %%p in ("%ROOT%logs\%~1.pid") do taskkill /PID %%p /T /F >nul 2>&1
del "%ROOT%logs\%~1.pid" >nul 2>&1
echo  [ok] Stopped %~1
exit /b 0
