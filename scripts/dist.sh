#!/usr/bin/env bash
# 打交付包：把「能在另一台同型号 GPU 机器上双击就跑」的东西收成一个目录。
#
#   bash scripts/dist.sh                  # 出 dist/swim_analyse/
#   bash scripts/dist.sh --zip            # 再打成 dist/swim_analyse.zip
#   bash scripts/dist.sh --out D:/tmp/x   # 换输出位置
#
# 包里有什么、为什么：
#   swim_analyse.exe + 运行期 DLL   OpenCV / FFmpeg(libav*) / TensorRT / x264
#   models/{detect,pose}.engine      **预构建**的 TRT engine，不带 ONNX
#   models/stitch.lut                拼接查找表（109 MB）
#   run_1cam.bat / run_6cam.bat      两个一键入口
#   cameras.txt                      相机清单模板，现场改 IP 即可
#
# 刻意**不带**的东西：
#   ONNX（140 MB）           只有重建 engine 才需要，交付机不重建
#   nvinfer_builder_resource 1.8 GB，只有构建 engine 用得到
#   ffmpeg.exe / ffprobe.exe 242 MB×2，`--out` 编码与 --input 解码要它，
#                            但装一次系统级更合适（见包内 README 的前置要求）
#
# **engine 与机器绑定**：它按 GPU 架构（sm89 = RTX 40 系）+ TensorRT 版本烘死。
# 换到 30 系（sm86）或 50 系（sm120）必须在目标机重跑 scripts/build.bat 重烘。
# 这条在包内 README 与运行时报错里都写了。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/dist/swim_analyse"
ZIP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip) ZIP=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "用法: bash scripts/dist.sh [--zip] [--out <目录>]" >&2; exit 1 ;;
  esac
done

BIN="$ROOT/cpp/build/Release"
TRT_LIB="${SWIM_TRT_LIB:-D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33/lib}"

die() { printf '\033[1;31m[dist]\033[0m %s\n' "$1" >&2; exit 1; }
say() { printf '\033[1;32m[dist]\033[0m %s\n' "$1"; }

[[ -x "$BIN/swim_analyse.exe" ]] || die "缺少 $BIN/swim_analyse.exe，先跑 scripts/build.bat"
[[ -f "$ROOT/cpp/models/stitch.lut" ]] || die "缺少 cpp/models/stitch.lut，先跑 scripts/build.bat"

# engine 的文件名带身份戳（sm/TRT/batch/精度/onnx 戳），包里统一改成裸名 ——
# 交付机没有 ONNX，程序会直接用裸名的那个（见 trt_engine.cpp 的两种来源）。
pick_engine() {   # $1=detect|pose  -> 打印路径
  local hit; hit="$(ls -t "$ROOT/cpp/models/$1".*.engine 2>/dev/null | head -1)"
  [[ -n "$hit" ]] || die "缺少 $1 的 engine，先在本机跑一次分析让它构建（scripts/analyse.bat）"
  echo "$hit"
}
DETECT="$(pick_engine detect)"
POSE="$(pick_engine pose)"

rm -rf "$OUT"
mkdir -p "$OUT/models"

# ── 1. 二进制与运行期 DLL ───────────────────────────────────────────────────
cp "$BIN/swim_analyse.exe" "$OUT/"
cp "$BIN"/*.dll "$OUT/"
for d in nvinfer_10.dll nvonnxparser_10.dll; do
  [[ -f "$TRT_LIB/$d" ]] || die "缺少 $TRT_LIB/$d（用 SWIM_TRT_LIB= 指定 TRT 的 lib 目录）"
  cp "$TRT_LIB/$d" "$OUT/"
done
say "二进制 + $(ls "$OUT"/*.dll | wc -l) 个 DLL"

# ── 2. 模型与查找表 ─────────────────────────────────────────────────────────
# engine 逐字节校验：它有一百多兆，一次静默的坏拷贝会让交付机报「反序列化失败 /
# 与本机架构不匹配」，把人引向完全错误的方向（实测踩过一次）。
copy_verified() {   # $1=源 $2=目标
  cp "$1" "$2"
  local a b
  a="$(md5sum "$1" | cut -d' ' -f1)"
  b="$(md5sum "$2" | cut -d' ' -f1)"
  [[ "$a" == "$b" ]] || die "拷贝校验失败: $1 -> $2（md5 $a vs $b，重跑一次）"
}
copy_verified "$DETECT" "$OUT/models/detect.engine"
copy_verified "$POSE"   "$OUT/models/pose.engine"
copy_verified "$ROOT/cpp/models/stitch.lut" "$OUT/models/stitch.lut"
cp "$ROOT/cpp/models/pose_meta.json" "$OUT/models/"
say "engine: $(basename "$DETECT") -> detect.engine"
say "engine: $(basename "$POSE") -> pose.engine"
say "查找表: stitch.lut $(du -h "$OUT/models/stitch.lut" | cut -f1)（均已 md5 校验）"

# ── 3. 相机清单模板 ─────────────────────────────────────────────────────────
# 相机名必须与 stitch.lut 里的一致（由 configs/pool_mesh.json 的 meshes 顺序定），
# 用「相机=地址」而不是按行序：现场 IP 尾数与 mesh 顺序不同，按行序写迟早错位。
# 含中文且给人在记事本里改，所以 UTF-8 带 BOM + CRLF（同 docs/windows.md 的约定）。
{
  printf '\xEF\xBB\xBF'
  cat <<'EOF'
# 六路 ZCam 相机清单。现场只改右边的 IP。
#
# 相机名（左边）不要改：它必须与 models/stitch.lut 里的一致，由泳池标定决定。
# 注释掉某一行 = 那台相机不接，画布上它的区域留黑（分批上线时用得上）。
#
# 相机侧要先配好：4K + 30fps（movfmt=4KP29.97）、h264。
# 用仓库的 scripts/cams.sh setup 下发，或手工：
#   curl "http://<ip>/ctrl/rec?action=stop"          # movfmt 在录制中只读
#   curl "http://<ip>/ctrl/set?movfmt=4KP29.97"
#   curl "http://<ip>/ctrl/stream_setting?index=stream0&venc=h264"
#   curl "http://<ip>/ctrl/get?k=movfmt"             # 回读校验，相机会静默拒绝

cam3=rtsp://192.168.3.101/live_stream
cam2=rtsp://192.168.3.102/live_stream
cam1=rtsp://192.168.3.103/live_stream
cam4=rtsp://192.168.3.104/live_stream
cam5=rtsp://192.168.3.105/live_stream
cam6=rtsp://192.168.3.106/live_stream
EOF
} | sed 's/$/\r/' > "$OUT/cameras.txt"
say "相机清单模板 cameras.txt"

# ── 4. 两个一键入口 ─────────────────────────────────────────────────────────
# .bat 必须 CRLF + 纯 ASCII（cmd.exe 按系统 ANSI 代码页解析），所以中文说明放在
# 同目录的 README.txt（UTF-8 带 BOM，记事本能正常打开）。
write_bat() {   # $1=文件名  stdin=内容（LF，本函数负责转 CRLF）
  sed 's/$/\r/' > "$OUT/$1"
}

write_bat run_1cam.bat <<'EOF'
@echo off
rem Local single-camera run: analyse ONE camera's raw 4K stream directly.
rem No stitching - the panorama needs all six. Use this to verify a camera,
rem the network, and the GPU pipeline before the full rig is up.
rem
rem   double-click                 use the URL below
rem   run_1cam.bat rtsp://1.2.3.4/live_stream    override it
rem   run_1cam.bat <file.mp4>      any video file works too
rem   run_1cam.bat --max-frames 300              flags pass through
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.

setlocal
cd /d "%~dp0"
rem Call the exe by its full path, not by bare name: some environments set
rem NoDefaultCurrentDirectoryInExePath=1 (Git Bash does), and then cmd refuses
rem to search the current directory - the launcher would fail with 9009.
set "EXE=%~dp0swim_analyse.exe"

rem Arg 1 is the source only when it is not a flag; the rest passes through.
rem Shift loop rather than %1..%9 so quoting survives and there is no 9-arg cap.
set "SRC="
set "ARGS="
set "A=%~1"
if defined A if not "%A:~0,1%"=="-" goto :take_src
goto :collect
:take_src
set "SRC=%~1"
shift
:collect
if "%~1"=="" goto :parsed
set "ARGS=%ARGS% %1"
shift
goto :collect
:parsed

if not defined SRC set "SRC=rtsp://192.168.1.199/live_stream"

if not exist "%EXE%" (
  echo [error] swim_analyse.exe missing - is this the unpacked dist folder?
  goto :end
)
if not exist "models\detect.engine" (
  echo [error] models\detect.engine missing - incomplete package.
  goto :end
)

echo.
echo   source : %SRC%
echo   mode   : single camera, no stitching, live preview
echo   every frame runs detect + pose + tracking; nothing is replayed
echo   press q or ESC on the preview window to stop
echo.
rem Warn early: without ffprobe the exe silently falls back to OpenCV decoding,
rem which is 4x+ slower (5.7 vs 77.7 fps measured) and misreports the frame rate.
where ffprobe >nul 2>nul || echo [warn] ffprobe not on PATH - decoding falls back to OpenCV, 4x+ slower. See README.txt.
echo.

"%EXE%" --input "%SRC%" --models models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] exited with code %RC% - see README.txt

:end
echo.
echo Press any key to close...
pause >nul
endlocal
EOF

write_bat run_6cam.bat <<'EOF'
@echo off
rem Production six-camera run: pull six 4K streams, stitch on the GPU, analyse.
rem Nothing touches the CPU between decode and inference.
rem
rem   double-click                 read cameras.txt, live preview window
rem   run_6cam.bat --json out.json      also write per-swimmer results
rem   run_6cam.bat --out out.mp4        also write an annotated video (needs ffmpeg)
rem   run_6cam.bat <list.txt>      use a different camera list
rem
rem Edit cameras.txt to match the venue's IPs. Camera names on the left must
rem stay as they are - they are tied to the pool calibration in stitch.lut.
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.

setlocal
cd /d "%~dp0"
rem Call the exe by its full path, not by bare name: some environments set
rem NoDefaultCurrentDirectoryInExePath=1 (Git Bash does), and then cmd refuses
rem to search the current directory - the launcher would fail with 9009.
set "EXE=%~dp0swim_analyse.exe"

rem Arg 1 is the camera list only when it is not a flag; the rest passes through.
set "LIST="
set "ARGS="
set "A=%~1"
if defined A if not "%A:~0,1%"=="-" goto :take_list
goto :collect
:take_list
set "LIST=%~1"
shift
:collect
if "%~1"=="" goto :parsed
set "ARGS=%ARGS% %1"
shift
goto :collect
:parsed

if not defined LIST set "LIST=cameras.txt"

if not exist "%EXE%" (
  echo [error] swim_analyse.exe missing - is this the unpacked dist folder?
  goto :end
)
if not exist "models\stitch.lut" (
  echo [error] models\stitch.lut missing - incomplete package.
  goto :end
)
if not exist "%LIST%" (
  echo [error] camera list not found: %LIST%
  echo         edit cameras.txt, or pass a list file as the first argument.
  goto :end
)

echo.
echo   cameras : %LIST%
echo   mode    : six 4K streams -^> NVDEC -^> GPU stitch -^> detect + pose
echo   first run builds nothing - engines are prebuilt in this package
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" --cam-dir "%LIST%" --models models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] exited with code %RC% - see README.txt

:end
echo.
echo Press any key to close...
pause >nul
endlocal
EOF
say "入口 run_1cam.bat / run_6cam.bat"

# ── 5. 包内 README（UTF-8 带 BOM，老记事本才不乱码）─────────────────────────
{
  printf '\xEF\xBB\xBF'
  cat <<EOF
游泳动作分析 —— 交付包
========================

两个入口，双击即可：

  run_1cam.bat    单相机联调。直接分析一路 4K 原始画面，不拼接。
                  用来验相机、验网络、验这台机器的 GPU 链路。
  run_6cam.bat    六路上线。拉六路 4K，在 GPU 上拼成全景后分析，
                  中间不落文件、不回 CPU。

上线前只需要改一个文件：**cameras.txt**，把右边的 IP 换成现场的。
左边的相机名不要动 —— 它与泳池标定（models/stitch.lut）绑定，改了会接缝错位。
某台相机还没接就注释掉那一行，画布上它的区域留黑，其余五路照常工作。

前置要求
--------
* NVIDIA 显卡，驱动 >= 550。**架构必须与打包机一致**（本包 = $(basename "$DETECT" | sed 's/.*\.\(sm[0-9]*\)-.*/\1/')，
  即 RTX 40 系）。换 30 系或 50 系必须用源码仓库重新烘 engine。
* CUDA 运行库：装 CUDA Toolkit 12.x，或把 cudart64_12.dll 放到本目录。
* ffmpeg / ffprobe 在 PATH 里。**run_1cam.bat 必须有它**：单相机路（--input）
  用 ffmpeg 管道解码，缺了会静默回退 OpenCV/MSMF —— 同一台 4K60 相机实测
  5.7~15 fps（有 ffmpeg 时 66~78 fps），且帧率报成 30.00（真值 59.94）。
  --out 写标注视频、分析已拼好的视频文件也要它。
  run_6cam.bat 不需要（NVDEC 在进程内解码，不经 ffmpeg）。
  装法：winget install Gyan.FFmpeg，装完**重开一个命令行窗口**（PATH 才生效）

相机侧配置（4K + 30fps）
------------------------
movfmt 决定主流的分辨率与帧率，**它在录制中是只读的** —— 查询会看到 "ro":1，
写入返回 code:-1 且不报原因。所以要先停录：

  curl "http://<ip>/ctrl/rec?action=stop"
  curl "http://<ip>/ctrl/set?movfmt=4KP29.97"
  curl "http://<ip>/ctrl/stream_setting?index=stream0&venc=h264"
  curl "http://<ip>/ctrl/get?k=movfmt"          回读校验，必须看到 4KP29.97

RTSP 只有一个挂载点 rtsp://<ip>/live_stream，它给出的是相机当前 send_stream
选中的那一路（默认 Stream0，即 4K 主流）。同一台相机最多约 4 个并发会话。

常用参数
--------
  run_6cam.bat --json out.json      每个泳者的划水次数与末速
  run_6cam.bat --out out.mp4        标注视频（需要 ffmpeg，编码会吃掉一半帧率）
  run_6cam.bat --max-frames 300     只跑前 300 帧
  run_6cam.bat --preview-scale 0.4  预览窗口缩放比
  swim_analyse.exe --help           全部参数

看不懂报错时
------------
"0xC0000135" / "-1073741515" / 启动即退，什么都不打印
    缺 DLL，八成是 cudart64_12.dll。装 CUDA Toolkit 12.x，或把该文件从别的
    机器拷到本目录。这是最常见的一条。

"engine 反序列化失败 … 与本机的 GPU 架构或 TensorRT 版本不匹配"
    换了显卡架构。必须在目标机用源码仓库重烘 engine，本包无法自愈
    （包里不带 ONNX 与 TRT 的构建资源，那是 2 GB 的东西）。
    也可能是包在传输中损坏 —— 先比一下 models/ 下三个文件的大小。

"9009" / "'swim_analyse.exe' 不是内部或外部命令"
    不该出现（入口用绝对路径调用）。若出现说明包被拆散了，exe 与 .bat 不同目录。

"ffprobe 探测失败，回退 OpenCV 解码"
    PATH 里没有 ffprobe。程序还能跑，但慢四倍以上（实测 5.7 fps vs 77.7），
    且帧率显示不对（30.00 vs 真值 59.94）。装 ffmpeg 后**重开命令行**再试。

"读取失败(-138)，重试中"
    相机流断了或网络抖动。单次是正常的，持续出现要查网线与相机状态。

"直播流累计丢弃 N 帧"
    正常。这台机器追不上相机帧率时会丢旧帧保实时；N 持续暴涨说明该降负载。

性能参考（RTX 4080 Laptop）
---------------------------
  单相机直分析      约 66~78 fps（PATH 里有 ffmpeg；没有则掉到 6~15 fps）
  六路现拼 + 分析   约 38 fps（瓶颈是 NVDEC，六路 4K 已把解码器跑满）
  加 --out 落盘     约 27~35 fps（libx264 与解码抢 CPU）
EOF
} > "$OUT/README.txt"
sed -i 's/$/\r/' "$OUT/README.txt"
say "README.txt"

# ── 6. 汇总 ────────────────────────────────────────────────────────────────
say "完成: $OUT  $(du -sh "$OUT" | cut -f1)"
# find 的 %s 是字节；按大小降序列出，让人一眼看到大头在哪
find "$OUT" -maxdepth 2 -type f -printf '%s %P\n' |
  sort -rn | awk '{ printf "  %8.1f MB  %s\n", $1/1048576, $2 }'

if [[ $ZIP -eq 1 ]]; then
  command -v zip >/dev/null || die "没有 zip 命令（Git Bash 自带；或手工压缩）"
  ZIPPATH="$(dirname "$OUT")/$(basename "$OUT").zip"
  rm -f "$ZIPPATH"
  (cd "$(dirname "$OUT")" && zip -qr "$(basename "$ZIPPATH")" "$(basename "$OUT")")
  say "打包: $ZIPPATH  $(du -h "$ZIPPATH" | cut -f1)"
fi
