@echo off
goto :run

rem ==========================================================================
rem  LIVE PREVIEW - opens a window and runs detect + pose + tracking on every
rem  frame as it arrives. Nothing is cached or replayed: what the window shows
rem  is what was just computed for that frame.
rem
rem  Double-click to run (reads the already-stitched panorama by default), or
rem  name a source:
rem      scripts\preview.bat              stitched panorama video (default)
rem      scripts\preview.bat 6cam         six raw 4K clips, stitched on the GPU
rem      scripts\preview.bat <video>      a specific panorama video (or drag it here)
rem      scripts\preview.bat <folder>     a six-camera clip folder (or drag it here)
rem      scripts\preview.bat rtsp://...   live stream
rem  Anything else is passed straight through, e.g.
rem      scripts\preview.bat 6cam --preview-scale 0.5
rem
rem  Want json / mp4 instead of a window?  ->  scripts\analyse.bat
rem  Never built it?                       ->  scripts\build.bat
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
rem EXE. This launcher only decides where results go: a window.
call "%~dp0env.bat" %* || goto :end

echo.
echo   %LABEL%
echo   every frame runs detect + pose + tracking live; nothing is replayed
echo   first run builds TensorRT engines - that takes a few minutes
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" %MODE% "%INPUT%" --models cpp\models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] swim_analyse exited with code %RC%

:end
echo.
echo Press any key to close...
pause >nul
endlocal
