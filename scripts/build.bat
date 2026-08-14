@echo off
rem swim_analyse - build the C++/CUDA realtime pipeline on Windows (double-click).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Notes live in cpp/README.md, not here.
rem
rem Does three things, each skipped when already done:
rem   1) cmake configure  (Visual Studio 2022 + CUDA + TensorRT + vcpkg OpenCV)
rem   2) cmake --build --config Release
rem   3) export detect.onnx / pose.onnx from weights/  (needs .venv, install.sh)
rem TensorRT engines are NOT built here - the first run builds and caches them
rem (a few minutes), keyed by GPU arch + TRT version + flags.
rem
rem Override paths / rebuild behaviour:
rem   set SWIM_TRT_ROOT=<TensorRT-10.11 root>
rem   set SWIM_VCPKG=<vcpkg root>
rem   set SWIM_CUDA_ARCH=89        RTX40=89  RTX50=120  RTX30=86
rem   set SWIM_FRESH=1             delete cpp\build and reconfigure

setlocal
rem This script lives in scripts\ but every path below is repo-root relative,
rem so cd to the parent of %~dp0. Double-click still works from anywhere.
cd /d "%~dp0.."

if not defined SWIM_TRT_ROOT set "SWIM_TRT_ROOT=D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33"
if not defined SWIM_VCPKG    set "SWIM_VCPKG=D:\BaiduNetdiskDownload\vcpkg-2025.12.12"
if not defined SWIM_CUDA_ARCH set "SWIM_CUDA_ARCH=89"

if not exist "%SWIM_TRT_ROOT%\include\NvInfer.h" (
  echo [error] TensorRT not found: "%SWIM_TRT_ROOT%\include\NvInfer.h"
  echo         set SWIM_TRT_ROOT to the TensorRT 10.11 root and retry.
  goto :end
)
set "TOOLCHAIN=%SWIM_VCPKG%\scripts\buildsystems\vcpkg.cmake"
if not exist "%TOOLCHAIN%" (
  echo [error] vcpkg toolchain not found: "%TOOLCHAIN%"
  echo         set SWIM_VCPKG, or edit this file to pass -DOpenCV_DIR instead.
  goto :end
)
"%SystemRoot%\System32\where.exe" cmake >nul 2>nul || (
  echo [error] cmake not on PATH - install CMake ^>=3.18 or use a VS Developer Prompt
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
if exist cpp\models\detect.onnx if exist cpp\models\pose.onnx (
  echo [build] onnx already exported - delete cpp\models\*.onnx to redo
  goto :done
)
if not exist "%PY%" (
  echo [warn] %PY% missing - cannot export ONNX. Run scripts/install.sh, then:
  echo        %PY% cpp\tools\export_onnx.py --out cpp\models
  goto :done
)
echo [build] export onnx from weights
set PYTHONUTF8=1
"%PY%" cpp\tools\export_onnx.py --out cpp\models || goto :fail

:done
echo.
echo [build] ok. Next: double-click scripts\preview.bat (live window)
echo         or scripts\analyse.bat (json / annotated mp4).
echo         The first run builds TensorRT engines - takes a few minutes.
goto :end

:fail
echo.
echo [error] build failed with code %ERRORLEVEL% - see the log above.

:end
echo.
echo Press any key to close...
pause >nul
endlocal
