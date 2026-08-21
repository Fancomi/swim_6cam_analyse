@echo off
rem Shared preflight + source resolution for the C++ launchers.
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem
rem Called (not executed) by preview.bat / analyse.bat, so it must NOT use
rem setlocal: PATH / EXE / MODE / INPUT / NAME / LABEL have to survive back into
rem the caller.
rem
rem   call "%~dp0env.bat" %*     (caller has already cd'd to the repo root)
rem
rem The two launchers differ ONLY in where results go (window vs files); the
rem command line is identical, so it is parsed here once for both. Arg 1 is the
rem source when it does not start with "-"; everything else lands in ARGS and is
rem passed through to the binary untouched.
rem
rem   (nothing) / canvas   the already-stitched panorama mp4   -> --input
rem   6cam                 six raw camera clips, GPU stitch    -> --cam-dir
rem   <file>               that panorama video                -> --input
rem   <dir>                six raw camera clips in that dir    -> --cam-dir
rem   <rtsp:// ...>        live panorama stream                -> --input
rem
rem On success it sets:
rem   PATH   with the TensorRT lib dir prepended
rem   EXE    the binary
rem   MODE   --input or --cam-dir
rem   INPUT  the value to pass after MODE
rem   ARGS   the remaining flags, verbatim
rem   NAME   basename for output files
rem   LABEL  one line describing the source, for the caller to echo
rem   IS_URL set when INPUT is a stream (batch mode must refuse it)
rem
rem Override the defaults:
rem   set SWIM_CANVAS=<stitched panorama mp4>
rem   set SWIM_CAM_DIR=<dir holding one *_<camera>.mp4 per camera>
rem   set SWIM_TRT_LIB=<TensorRT 10.11 lib dir>

set "IS_URL="
set "TRT_LIB=%SWIM_TRT_LIB%"
if not defined TRT_LIB set "TRT_LIB=D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33\lib"
if not exist "%TRT_LIB%\nvinfer_10.dll" goto :no_trt
set "PATH=%TRT_LIB%;%PATH%"

set "EXE=cpp\build\Release\swim_analyse.exe"
if not exist "%EXE%" goto :no_exe
if not exist "cpp\models\detect.onnx" goto :no_model
if not exist "cpp\models\pose.onnx"   goto :no_model

if not defined SWIM_CANVAS  set "SWIM_CANVAS=data\20260730\merged_3000f.mp4"
if not defined SWIM_CAM_DIR set "SWIM_CAM_DIR=D:\WindowsProject\workspace\SWIM\20260730-4k-raw"

rem Split the command line. A shift loop rather than %1..%9 so quoting survives
rem and there is no 9-argument cap. No parentheses around a set/test pair: cmd
rem expands a whole block before running it, so a variable set inside a block
rem cannot be read in the same block.
set "ARG="
set "ARGS="
set "A=%~1"
rem Two separate ifs, NOT `if defined A if not "%A:~0,1%"=="-"`: cmd expands the
rem whole line before running any of it, so the substring operator is evaluated
rem even when A is empty - and ":~0,1" on an undefined name expands to the bare
rem text ~0,1 while eating the quotes, leaving the line unbalanced. Symptom:
rem "The syntax of the command is incorrect." on the NO-ARGUMENT path only, i.e.
rem exactly the double-click path that never gets tested from a shell.
if not defined A goto :collect
if "%A:~0,1%"=="-" goto :collect
set "ARG=%~1"
shift
:collect
if "%~1"=="" goto :resolve
set "ARGS=%ARGS% %1"
shift
goto :collect

:resolve
rem Keyword or path. Expanding the keyword first means the tests below see a path
rem either way, so there is one code path instead of two.
if not defined ARG      set "ARG=canvas"
if /I "%ARG%"=="canvas" set "ARG=%SWIM_CANVAS%"
if /I "%ARG%"=="6cam"   set "ARG=%SWIM_CAM_DIR%"

set "INPUT=%ARG%"
set "MODE=--input"
for %%I in ("%INPUT%") do set "NAME=%%~nI"
rem A directory means "six raw camera clips": stitch them on the GPU instead of
rem reading an already-stitched panorama. Same pipeline downstream either way.
if exist "%INPUT%\" goto :six
rem Substring test via pure batch expansion: calling find/where here would pick
rem up the Unix tools when launched from a Git Bash shell.
if not "%INPUT%"=="%INPUT://=%" goto :url
if not exist "%INPUT%" goto :no_input
set "LABEL=canvas : %INPUT%"
goto :ffmpeg

:six
set "MODE=--cam-dir"
if not exist "cpp\models\stitch.lut" goto :no_lut
set "LABEL=6cam   : %INPUT%  (NVDEC decode + GPU stitch, no intermediate file)"
goto :ffmpeg

:url
set "IS_URL=1"
set "NAME=stream"
set "LABEL=stream : %INPUT%"

:ffmpeg
rem ffmpeg is the default decode path for --input and the only usable encoder for
rem the 5002-wide canvas (MSMF cannot open it for writing). --cam-dir decodes
rem with NVDEC in-process and needs it only for --out.
"%SystemRoot%\System32\where.exe" ffmpeg >nul 2>nul || echo [warn] ffmpeg not on PATH - --out will fail, and --input falls back to OpenCV (MSMF, slower, different pixels)
exit /b 0

:no_trt
echo [error] TensorRT not found: "%TRT_LIB%\nvinfer_10.dll"
echo         set SWIM_TRT_LIB to the TensorRT 10.11 lib directory and retry.
exit /b 1

:no_exe
echo [error] %EXE% missing - build it first: double-click scripts\build.bat
exit /b 1

:no_model
echo [error] cpp\models\detect.onnx or pose.onnx missing - run scripts\build.bat
exit /b 1

:no_lut
echo [error] cpp\models\stitch.lut missing - six-camera mode needs it.
echo         run scripts\build.bat to bake it from configs\pool_mesh.json.
exit /b 1

:no_input
echo [error] input not found: %INPUT%
echo         pass "canvas", "6cam", a video file, a six-camera folder, or a URL.
exit /b 1
