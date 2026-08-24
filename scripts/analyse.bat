@echo off
goto :run

rem ==========================================================================
rem  BATCH ANALYSE - runs the whole clip and writes results to files (json, and
rem  optionally an annotated mp4). No window.
rem
rem  Same algorithm as preview.bat, frame by frame, nothing cached; the only
rem  difference is that results land on disk instead of on screen.
rem
rem  Double-click to run (reads the already-stitched panorama by default), or
rem  name a source:
rem      scripts\analyse.bat              stitched panorama video (default)
rem      scripts\analyse.bat 6cam         six raw 4K clips, stitched on the GPU
rem      scripts\analyse.bat <video>      a specific panorama video (or drag it here)
rem      scripts\analyse.bat <folder>     a six-camera clip folder (or drag it here)
rem  Anything else is passed straight through; the two common ones:
rem      scripts\analyse.bat --out o.mp4            also write an annotated video
rem      scripts\analyse.bat 6cam --max-frames 300  stop after 300 frames
rem
rem  A live stream has no end, so it cannot be batched - use scripts\preview.bat.
rem  Never built it?  ->  scripts\build.bat
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
rem dir / URL) and checks the toolchain, setting MODE / INPUT / ARGS / NAME /
rem LABEL / EXE / IS_URL. This launcher only decides where results go: files.
call "%~dp0env.bat" %* || goto :end

if defined IS_URL (
  echo [error] a stream has no end, so it cannot be batch-processed: %INPUT%
  echo         use scripts\preview.bat for rtsp:// / rtmp:// / http:// input.
  goto :end
)

if not exist output mkdir output
set "JSON=output\%NAME%_cpp.json"

echo.
echo   %LABEL%
echo   json  : %JSON%
echo   detect + pose + tracking on every frame; no results are cached
echo   add --out o.mp4 for an annotated video (needs ffmpeg on PATH)
echo.

"%EXE%" %MODE% "%INPUT%" --models cpp\models --json "%JSON%" --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (echo [error] swim_analyse exited with code %RC%) else (echo [ok] wrote %JSON%)

:end
echo.
echo Press any key to close...
pause >nul
endlocal
