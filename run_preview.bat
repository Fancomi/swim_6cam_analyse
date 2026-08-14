@echo off
rem swim_analyse - realtime preview launcher (double-click to run).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Notes are kept in README.md / docs, not here.
rem
rem   double-click            -> analyse the default canvas video
rem   drag a video onto me    -> analyse that file
rem   run_preview.bat URL     -> rtsp:// / rtmp:// / http:// live stream
rem   run_preview.bat f --out o.mp4   -> extra flags are passed through
rem
rem Override the TensorRT lib dir with:  set SWIM_TRT_LIB=<dir>

setlocal
cd /d "%~dp0"

set "TRT_LIB=%SWIM_TRT_LIB%"
if not defined TRT_LIB set "TRT_LIB=D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33\lib"
if not exist "%TRT_LIB%\nvinfer_10.dll" goto :no_trt
set "PATH=%TRT_LIB%;%PATH%"

set "EXE=cpp\build\Release\swim_analyse.exe"
if not exist "%EXE%" goto :no_exe
if not exist "cpp\models\detect.onnx" goto :no_model
if not exist "cpp\models\pose.onnx"   goto :no_model

set "INPUT=%~1"
if not defined INPUT set "INPUT=data\20260730\merged_3000f.mp4"
rem Substring test done with pure batch expansion: calling find/where here would
rem pick up the Unix tools when launched from a Git Bash shell.
if not "%INPUT%"=="%INPUT://=%" set "IS_URL=1"
if not defined IS_URL if not exist "%INPUT%" goto :no_input

"%SystemRoot%\System32\where.exe" ffmpeg >nul 2>nul || echo [warn] ffmpeg not on PATH - decode falls back to OpenCV (MSMF), slower
echo.
echo   input : %INPUT%
echo   every frame runs detect + pose + tracking live; nothing is replayed
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" --input "%INPUT%" --models cpp\models --preview --show-fps %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] swim_analyse exited with code %RC%
goto :end

:no_trt
echo [error] TensorRT not found: "%TRT_LIB%\nvinfer_10.dll"
echo         set SWIM_TRT_LIB to the TensorRT 10.11 lib directory and retry.
goto :end

:no_exe
echo [error] %EXE% missing - build it first:
echo         cd cpp ^&^& cmake --build build --config Release
goto :end

:no_model
echo [error] cpp\models\detect.onnx or pose.onnx missing - export them first:
echo         .venv\Scripts\python.exe cpp\tools\export_onnx.py --out cpp\models
goto :end

:no_input
echo [error] input not found: %INPUT%
goto :end

:end
echo Press any key to close...
pause >nul
endlocal
