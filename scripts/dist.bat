@echo off
rem swim_analyse - build and package for delivery (double-click).
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in CLAUDE.md; this file only orchestrates.
rem
rem Two levels, matching "first copy" vs "every update after that":
rem
rem   dist.bat            FULL     -> dist\swim_analyse\         about 3.0 GB
rem   dist.bat inc        UPDATE   -> dist\swim_analyse_update\  only what changed
rem   dist.bat zip        FULL + dist\swim_analyse.zip
rem   dist.bat inc zip    UPDATE + zip
rem
rem Both levels build first (cmake --build is a no-op when nothing changed), so
rem "double-click, wait, copy the folder" is the whole workflow.
rem
rem The FULL package needs nothing installed on the target machine except an
rem NVIDIA driver: CUDA runtime, VC runtime, ffmpeg, prebuilt engines, ONNX and
rem the TensorRT builder resource all travel with it. Different GPU architecture
rem is fine - the first run re-bakes the engines from the bundled ONNX.
rem
rem   set SWIM_DIST_LEAN=1   skip ONNX + builder resource (saves 1.8 GB, but the
rem                          package then only works on this GPU architecture)

setlocal
cd /d "%~dp0.."

set "MODE="
set "ZIP="
:args
if "%~1"=="" goto :parsed
if /I "%~1"=="inc" set "MODE=-Inc"
if /I "%~1"=="zip" set "ZIP=-Zip"
rem shift /1 leaves %0 alone. Plain "shift" moves %0 to %1, and then the
rem %~dp0 below resolves to the current directory instead of scripts\ -
rem "dist.bat inc" would fail to find build.bat while bare dist.bat worked.
shift /1
goto :args
:parsed

rem 1) build. Skips whatever is already done; SWIM_NO_PAUSE keeps it from
rem    stopping for a keypress in the middle of this script.
echo.
echo [dist] step 1/2  build
set "SWIM_NO_PAUSE=1"
call "%~dp0build.bat" || goto :fail
set "SWIM_NO_PAUSE="

rem 2) package. Windows PowerShell 5.1 ships with the OS, so packaging needs no
rem    extra tooling - no Git Bash, no coreutils. -ExecutionPolicy Bypass because
rem    the script is unsigned and the default policy on a fresh machine is
rem    Restricted; it applies to this one invocation only.
rem    TensorRT engines are baked on first run, not by build.bat, so a fresh
rem    clone needs one analyse run before the engines exist - dist.ps1 says so
rem    if they are missing.
echo.
if defined MODE (echo [dist] step 2/2  package  ^(incremental^)) else (echo [dist] step 2/2  package  ^(full^))
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0dist.ps1" %MODE% %ZIP% || goto :fail

echo.
echo [dist] ok.
set "RC=0"
if not defined MODE goto :say_full
rem dist.ps1 removes the update folder when nothing changed, so check
rem before telling anyone to copy it.
if not exist "dist\swim_analyse_update" goto :end
echo        Copy dist\swim_analyse_update\ to the target machine and
echo        double-click update.bat inside it.
goto :end
:say_full
echo        Copy dist\swim_analyse\ to the target machine and
echo        double-click run_6cam.bat inside it. Nothing to install there.
goto :end

:fail
echo.
echo [error] failed with code %ERRORLEVEL% - see the log above.
set "RC=1"

:end
echo.
echo Press any key to close...
pause >nul
endlocal & exit /b %RC%
