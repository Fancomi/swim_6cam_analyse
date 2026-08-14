@echo off
rem swim_analyse - realtime preview with the C++ pipeline (double-click to run).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Notes live in cpp/README.md, not here.
rem
rem   double-click              -> preview the default canvas video
rem   drag a video onto me      -> preview that file
rem   preview.bat URL           -> rtsp:// / rtmp:// / http:// live stream
rem   preview.bat --preview-scale 0.5       -> flags are passed through
rem
rem Every frame runs detect + pose + tracking live; nothing is replayed.
rem Press q or ESC on the window to stop.
rem
rem Write json / mp4 instead of a window: analyse.bat
rem Build / export ONNX first:            build.bat
rem Override the TensorRT lib dir:        set SWIM_TRT_LIB=<dir>

setlocal
rem This script lives in scripts\ but all paths below are repo-root relative,
rem so cd to the parent of %~dp0. Double-click still works from anywhere.
cd /d "%~dp0.."
call "%~dp0env.bat" || goto :end

rem Arg 1 is the input only when it does not start with "-"; everything else is
rem passed through. Collected with a shift loop (not %1..%9) so quoting survives
rem and there is no 9-argument cap. No parentheses around the set/test pair:
rem cmd expands a whole block before running it, so a variable set inside a block
rem cannot be read in the same block.
set "INPUT="
set "ARGS="
set "A=%~1"
if defined A if not "%A:~0,1%"=="-" goto :take_input
goto :collect
:take_input
set "INPUT=%~1"
shift
:collect
if "%~1"=="" goto :parsed
set "ARGS=%ARGS% %1"
shift
goto :collect
:parsed

if not defined INPUT set "INPUT=data\20260730\merged_3000f.mp4"
rem Substring test via pure batch expansion: calling find/where here would pick
rem up the Unix tools when launched from a Git Bash shell.
if not "%INPUT%"=="%INPUT://=%" set "IS_URL=1"
if not defined IS_URL if not exist "%INPUT%" (
  echo [error] input not found: %INPUT%
  goto :end
)

echo.
echo   input : %INPUT%
echo   every frame runs detect + pose + tracking live; nothing is replayed
echo   first run builds TensorRT engines - that takes a few minutes
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" --input "%INPUT%" --models cpp\models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] swim_analyse exited with code %RC%

:end
echo.
echo Press any key to close...
pause >nul
endlocal
