@echo off
rem Shared preflight for the C++ realtime pipeline launchers.
rem Called (not executed) by preview.bat / analyse.bat, so it must NOT use
rem setlocal: PATH and EXE have to survive back into the caller.
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem
rem   call "%~dp0env.bat" || goto :fail      (caller has cd'd to the repo root)
rem
rem On success: PATH contains the TensorRT lib dir, EXE points at the binary.
rem Override the TensorRT lib dir with:  set SWIM_TRT_LIB=<dir>

set "TRT_LIB=%SWIM_TRT_LIB%"
if not defined TRT_LIB set "TRT_LIB=D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33\lib"
if not exist "%TRT_LIB%\nvinfer_10.dll" (
  echo [error] TensorRT not found: "%TRT_LIB%\nvinfer_10.dll"
  echo         set SWIM_TRT_LIB to the TensorRT 10.11 lib directory and retry.
  exit /b 1
)
set "PATH=%TRT_LIB%;%PATH%"

set "EXE=cpp\build\Release\swim_analyse.exe"
if not exist "%EXE%" (
  echo [error] %EXE% missing - build it first: double-click scripts\build.bat
  exit /b 1
)
if not exist "cpp\models\detect.onnx" goto :no_model
if not exist "cpp\models\pose.onnx"   goto :no_model

rem ffmpeg is the default decode path and the only usable encoder for the
rem 5002-wide canvas (MSMF cannot open it for writing).
"%SystemRoot%\System32\where.exe" ffmpeg >nul 2>nul || echo [warn] ffmpeg not on PATH - decode falls back to OpenCV (MSMF, slower, different pixels)
exit /b 0

:no_model
echo [error] cpp\models\detect.onnx or pose.onnx missing - run scripts\build.bat
exit /b 1
