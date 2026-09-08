@echo off
rem Configure the six fixed ZCam units for the swim stitcher.
rem Target: 3840x2160, 29.97 fps, H.264 on stream0.
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.

setlocal EnableExtensions EnableDelayedExpansion
set "RC=0"

where curl.exe >nul 2>&1
if errorlevel 1 (
  echo [error] curl.exe was not found in PATH.
  echo         Windows 10/11 normally includes it.
  set "RC=1"
  goto :end
)

echo Configuring ZCam 192.168.3.101-106 ...
for %%I in (101 102 103 104 105 106) do call :setup 192.168.3.%%I

echo.
if "%RC%"=="0" (
  echo [ok] all six cameras are configured for 3840x2160 29.97fps H.264.
) else (
  echo [error] one or more cameras failed. Check the lines above and retry.
)

:end
echo.
echo Press any key to close...
pause >nul
endlocal & exit /b %RC%

:setup
set "IP=%~1"
set "CAM_FAIL=0"
echo.
echo [zcam] !IP!  3840x2160  29.97fps  H.264

rem movfmt is read-only while recording, so stop first.
call :optional_request "http://!IP!/ctrl/rec?action=stop" "stop recording"
timeout /t 1 /nobreak >nul
call :request "http://!IP!/ctrl/set?movfmt=4KP29.97" "set 4KP29.97"
call :request "http://!IP!/ctrl/stream_setting?index=stream0&venc=h264" "set stream0 H.264"
call :request "http://!IP!/ctrl/stream_setting?index=stream0&bitrate=50000000" "set stream0 bitrate"
call :request "http://!IP!/ctrl/set?send_stream=Stream0" "select stream0"

set "MOVFMT="
for /f "delims=" %%A in ('curl.exe -sS -f --noproxy "*" --connect-timeout 3 --max-time 8 "http://!IP!/ctrl/get?k=movfmt" 2^>nul') do set "MOVFMT=%%A"
if not defined MOVFMT (
  echo [error] !IP! could not read movfmt
  set "CAM_FAIL=1"
) else (
  echo !MOVFMT! | findstr /C:"4KP29.97" >nul
  if errorlevel 1 (
    echo [error] !IP! movfmt is not 4KP29.97: !MOVFMT!
    set "CAM_FAIL=1"
  ) else echo [ok] !IP! movfmt=4KP29.97
)

set "STREAM="
for /f "delims=" %%A in ('curl.exe -sS -f --noproxy "*" --connect-timeout 3 --max-time 8 "http://!IP!/ctrl/stream_setting?action=query^&index=stream0" 2^>nul') do set "STREAM=%%A"
if not defined STREAM (
  echo [error] !IP! could not read stream0 settings
  set "CAM_FAIL=1"
) else (
  echo !STREAM! | findstr /I /C:"h264" >nul
  if errorlevel 1 (
    echo [error] !IP! stream0 is not H.264: !STREAM!
    set "CAM_FAIL=1"
  ) else echo [ok] !IP! stream0=H.264
)

if not "!CAM_FAIL!"=="0" set "RC=1"
exit /b 0

:request
curl.exe -sS -f --noproxy "*" --connect-timeout 3 --max-time 8 "%~1" >nul 2>&1
if errorlevel 1 (
  echo [error] !IP! %~2 failed
  set "CAM_FAIL=1"
)
exit /b 0

:optional_request
curl.exe -sS -f --noproxy "*" --connect-timeout 3 --max-time 8 "%~1" >nul 2>&1
if errorlevel 1 echo [warn] !IP! %~2 request failed; continuing
exit /b 0
