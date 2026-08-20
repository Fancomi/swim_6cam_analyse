@echo off
rem swim_analyse - build the C++/CUDA realtime pipeline on Windows (double-click).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Notes live in cpp/README.md, not here.
rem
rem Does four things, each skipped when already done:
rem   1) cmake configure  (Visual Studio 2022 + CUDA + TensorRT + vcpkg OpenCV)
rem   2) cmake --build --config Release
rem   3) export detect.onnx / pose.onnx from weights/  (needs .venv, install.sh)
rem   4) bake cpp\models\stitch.lut from configs\pool_mesh.json (six-camera input)
rem TensorRT engines are NOT built here - the first run builds and caches them
rem (a few minutes), keyed by GPU arch + TRT version + flags.
rem
rem Override paths / rebuild behaviour:
rem   set SWIM_TRT_ROOT=<TensorRT-10.11 root>
rem   set SWIM_VCPKG=<vcpkg root>
rem   set SWIM_CUDA_ARCH=89        RTX40=89  RTX50=120  RTX30=86
rem   set SWIM_FRESH=1             delete cpp\build and reconfigure
rem   set SWIM_NO_PAUSE=1          do not wait for a keypress (used by dist.bat)

setlocal
rem This script lives in scripts\ but every path below is repo-root relative,
rem so cd to the parent of %~dp0. Double-click still works from anywhere.
cd /d "%~dp0.."
set "RC=0"

if not defined SWIM_TRT_ROOT set "SWIM_TRT_ROOT=D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33"
if not defined SWIM_VCPKG    set "SWIM_VCPKG=D:\BaiduNetdiskDownload\vcpkg-2025.12.12"
if not defined SWIM_CUDA_ARCH set "SWIM_CUDA_ARCH=89"

if not exist "%SWIM_TRT_ROOT%\include\NvInfer.h" (
  echo [error] TensorRT not found: "%SWIM_TRT_ROOT%\include\NvInfer.h"
  echo         set SWIM_TRT_ROOT to the TensorRT 10.11 root and retry.
  set "RC=1"
  goto :end
)
set "TOOLCHAIN=%SWIM_VCPKG%\scripts\buildsystems\vcpkg.cmake"
if not exist "%TOOLCHAIN%" (
  echo [error] vcpkg toolchain not found: "%TOOLCHAIN%"
  echo         set SWIM_VCPKG, or edit this file to pass -DOpenCV_DIR instead.
  set "RC=1"
  goto :end
)
"%SystemRoot%\System32\where.exe" cmake >nul 2>nul || (
  echo [error] cmake not on PATH - install CMake ^>=3.18 or use a VS Developer Prompt
  set "RC=1"
  goto :end
)

if defined SWIM_FRESH if exist cpp\build rmdir /s /q cpp\build

rem cmake re-runs configure by itself when CMakeLists.txt changes, so only the
rem very first time needs the full command line.
if not exist cpp\build\CMakeCache.txt (
  echo [build] configure  arch=%SWIM_CUDA_ARCH%
  cmake -S cpp -B cpp\build -G "Visual Studio 17 2022" ^
        -DTRT_ROOT="%SWIM_TRT_ROOT:\=/%" ^
        -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN:\=/%" ^
        -DCMAKE_CUDA_ARCHITECTURES=%SWIM_CUDA_ARCH% || goto :fail
)

echo [build] compile Release
cmake --build cpp\build --config Release || goto :fail

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [warn] %PY% missing - cannot export ONNX or bake the stitch LUT.
  echo        Run scripts/install.sh, then:
  echo          %PY% cpp\tools\export_onnx.py --out cpp\models
  echo          %PY% cpp\tools\build_stitch_lut.py
  goto :done
)
set PYTHONUTF8=1

if exist cpp\models\detect.onnx if exist cpp\models\pose.onnx (
  echo [build] onnx already exported - delete cpp\models\*.onnx to redo
  goto :lut
)
echo [build] export onnx from weights
"%PY%" cpp\tools\export_onnx.py --out cpp\models || goto :fail

:lut
rem Six-camera input (--cam-dir) needs this table; analysing an already-stitched
rem canvas (--input) does not. Skipped when newer than the calibration.
echo [build] bake stitch LUT from configs\pool_mesh.json
"%PY%" cpp\tools\build_stitch_lut.py || goto :fail

:done
echo.
echo [build] ok. Next: double-click scripts\preview.bat (live window)
echo         or scripts\analyse.bat (json / annotated mp4).
echo         The first run builds TensorRT engines - takes a few minutes.
goto :end

:fail
echo.
echo [error] build failed with code %ERRORLEVEL% - see the log above.
set "RC=1"

:end
echo.
rem "if cond a & b" would run b unconditionally (cmd splits on & before the if),
rem hence the goto. endlocal wipes RC, so expand it on the same line.
if defined SWIM_NO_PAUSE goto :quit
echo Press any key to close...
pause >nul
:quit
endlocal & exit /b %RC%
