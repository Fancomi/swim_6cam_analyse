@echo off
rem swim_analyse - batch analyse with the C++ realtime pipeline (double-click).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Notes live in cpp/README.md, not here.
rem
rem   double-click              -> analyse the default canvas, write json only
rem   drag a video onto me      -> analyse that file
rem   analyse.bat f --out o.mp4     -> also write an annotated video
rem   analyse.bat --max-frames 300  -> default input, flags passed through
rem
rem Live preview instead of batch: preview.bat
rem Build / export ONNX first:     build.bat
rem Override the TensorRT lib dir: set SWIM_TRT_LIB=<dir>

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
set "NAME="
set "ARGS="
set "A=%~1"
if defined A if not "%A:~0,1%"=="-" goto :take_input
goto :collect
:take_input
set "INPUT=%~1"
set "NAME=%~n1"
shift
:collect
if "%~1"=="" goto :parsed
set "ARGS=%ARGS% %1"
shift
goto :collect
:parsed

if not defined INPUT set "INPUT=data\20260730\merged_3000f.mp4"
if not defined NAME  set "NAME=merged_3000f"

rem Substring test via pure batch expansion: calling find/where here would pick
rem up the Unix tools when launched from a Git Bash shell.
if not "%INPUT%"=="%INPUT://=%" (
  echo [error] batch mode needs a file, not a stream: %INPUT%
  echo         use preview.bat for rtsp:// / rtmp:// / http:// input.
  goto :end
)
if not exist "%INPUT%" (
  echo [error] input not found: %INPUT%
  goto :end
)

if not exist output mkdir output
set "JSON=output\%NAME%_cpp.json"

echo.
echo   input : %INPUT%
echo   json  : %JSON%
echo   detect + pose + tracking on every frame; no results are cached
echo   add --out o.mp4 for an annotated video (needs ffmpeg on PATH)
echo.

"%EXE%" --input "%INPUT%" --models cpp\models --json "%JSON%" --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (echo [error] swim_analyse exited with code %RC%) else (echo [ok] wrote %JSON%)

:end
echo.
echo Press any key to close...
pause >nul
endlocal
