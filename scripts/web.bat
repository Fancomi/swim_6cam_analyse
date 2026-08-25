@echo off
goto :run

rem ==========================================================================
rem  WEB DASHBOARD - serves the live panorama plus the analysis results to a
rem  browser and opens it automatically. Same per-frame algorithm as the other
rem  two launchers; the only difference is where results go.
rem
rem  The three overlays (keypoints / analysis / pool grid) are checkboxes and are
rem  redrawn in the browser, so toggling them costs the pipeline nothing. Click a
rem  swimmer in the panorama to get a follow view plus that person's stats.
rem
rem  Double-click to run (reads the already-stitched panorama by default), or
rem  name a source:
rem      scripts\web.bat              stitched panorama video (default)
rem      scripts\web.bat 6cam         six raw 4K clips, stitched on the GPU
rem      scripts\web.bat <video>      a specific panorama video (or drag it here)
rem      scripts\web.bat <folder>     a six-camera clip folder (or drag it here)
rem      scripts\web.bat rtsp://...   live stream
rem  Anything else is passed straight through, e.g.
rem      scripts\web.bat 6cam --web-port 9000
rem      scripts\web.bat --web-bind 0.0.0.0     see the warning below
rem
rem  The server has NO authentication and NO TLS, so it binds 127.0.0.1 only.
rem  --web-bind 0.0.0.0 opens the live pool footage to everyone on the subnet.
rem
rem  Want a window instead?  ->  scripts\preview.bat
rem  Want files instead?     ->  scripts\analyse.bat
rem  Never built it?         ->  scripts\build.bat
rem
rem  UTF-8 without BOM + CRLF + ASCII only: cmd.exe parses .bat with the system
rem  ANSI codepage, so non-ASCII text here breaks on non-936 machines.
rem  Chinese notes live in cpp\README.md; encoding rules in docs\windows.md.
rem ==========================================================================

:run
setlocal
rem This script lives in scripts\ but all paths below are repo-root relative,
rem so cd to the parent of %~dp0. Double-click still works from anywhere.
cd /d "%~dp0.."

rem env.bat parses the command line, resolves the source (canvas / 6cam / file /
rem dir / URL) and checks the toolchain, setting MODE / INPUT / ARGS / LABEL /
rem EXE. This launcher only decides where results go: a browser.
call "%~dp0env.bat" %* || goto :end

echo.
echo   %LABEL%
echo   the browser opens by itself once the engines are ready
echo   first run builds TensorRT engines - that takes a few minutes
echo   press Ctrl-C here to stop the server
echo.

"%EXE%" %MODE% "%INPUT%" --models cpp\models --web --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] swim_analyse exited with code %RC%

:end
echo.
echo Press any key to close...
pause >nul
endlocal
