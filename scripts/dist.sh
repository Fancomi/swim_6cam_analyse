#!/usr/bin/env bash
# 打交付包。两级，对应「第一次拷过去」与「之后每次更新」：
#
#   bash scripts/dist.sh            全能包  dist/swim_analyse/         约 3.0 GB
#   bash scripts/dist.sh --inc      增量包  dist/swim_analyse_update/  只含变了的文件
#   bash scripts/dist.sh --zip      顺手压一个同名 .zip
#   bash scripts/dist.sh --out D:/x 换输出位置
#
# Windows 上双击 scripts/dist.bat 更省事：它先跑 build.bat 再调本脚本
#（`dist.bat inc` / `dist.bat zip` 对应上面的两个开关）。
#
# **全能包 = 目标机零安装**：exe、全部运行期 DLL、CUDA 运行库、VC 运行库、
# ffmpeg/ffprobe、预烘 engine、ONNX + TRT 构建资源、拼接查找表、两个双击入口。
# 目标机只需要 NVIDIA 卡 + 驱动 >= 550，别的什么都不用装。
#
# 带上 ONNX（133 MB）与 nvinfer_builder_resource（1.7 GB）是刻意的：engine 与
# 「GPU 架构 + TRT 版本」烘死，换代机器上预烘的那份用不了；带着这两样它能自己现烘
# （几分钟，之后秒开），不必回来重新打包。要省这 1.8 GB 就 SWIM_DIST_LEAN=1，
# 代价是换架构即失效（trt_engine.cpp 会给出「请在本机重新构建后再打包」）。
#
# **增量包 = 相对最近一次全能包的累积差异** + 一个 update.bat（拷进目标目录）。
# 通常只有 swim_analyse.exe 变（0.4 MB），改了权重才会带上 engine/ONNX。
# 按 dist/<名>.manifest 判断「上次全能包里是什么」，所以必须先打过一次全能包。
# 累积而非逐次差分：中间漏掉几个增量包也没关系，只应用最新的那个就对了。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/dist/swim_analyse"
ZIP=0
INC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inc) INC=1; shift ;;
    --zip) ZIP=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "用法: bash scripts/dist.sh [--inc] [--zip] [--out <目录>]" >&2; exit 1 ;;
  esac
done

BIN="$ROOT/cpp/build/Release"
TRT_LIB="${SWIM_TRT_LIB:-D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33/lib}"
LEAN="${SWIM_DIST_LEAN:-0}"
MANIFEST="$OUT.manifest"        # 全能包写、增量包读：上次全能包里有哪些文件与其 md5

die() { printf '\033[1;31m[dist]\033[0m %s\n' "$1" >&2; exit 1; }
say() { printf '\033[1;32m[dist]\033[0m %s\n' "$1"; }

# 生成的文本文件（清单模板 / 入口 / README）先落到这里，再与拷贝的文件走同一条
# 「md5 -> 比清单 -> 入包」的路，增量包才能自动带上改过的入口脚本。
GEN="$(mktemp -d)"
trap 'rm -rf "$GEN"' EXIT

# ── 依赖定位 ────────────────────────────────────────────────────────────────
# 这三样都不在仓库里，得去系统里找。找不到就停 —— 少一个都会让目标机在启动瞬间
# 以 -1073741515（找不到 DLL）死掉，什么都不打印，是最难被现场描述清楚的故障。
first_of() {   # 打印第一个存在的路径；都不存在则返回 1
  local p; for p in "$@"; do [[ -e "$p" ]] && { echo "$p"; return 0; }; done
  return 1
}

CUDART="$(first_of "${SWIM_CUDA_BIN:-/nonexistent}/cudart64_12.dll" \
    ${CUDA_PATH:+"$(cygpath -u "$CUDA_PATH" 2>/dev/null)/bin/cudart64_12.dll"} \
    /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.*/bin/cudart64_12.dll)" \
  || die "找不到 cudart64_12.dll（装 CUDA Toolkit 12.x，或用 SWIM_CUDA_BIN= 指定其 bin 目录）"

# 只带 exe 真正 import 的四个（dumpbin /dependents 核过），不整目录搬 VC143.CRT
VCRT_DIR="$(dirname "$(first_of "${SWIM_VCRT:-/nonexistent}/msvcp140.dll" \
    /c/Program\ Files/Microsoft\ Visual\ Studio/*/*/VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT/msvcp140.dll)")" \
  || die "找不到 VC 运行库 msvcp140.dll（用 SWIM_VCRT= 指定 Microsoft.VC143.CRT 目录）"

# ffmpeg/ffprobe：winget 装的是软链接，cp 会跟过去取到真身（231 MB×2）
FFMPEG="$(command -v ffmpeg)"  || die "PATH 里没有 ffmpeg（winget install Gyan.FFmpeg）"
FFPROBE="$(command -v ffprobe)" || die "PATH 里没有 ffprobe（同上）"

[[ -x "$BIN/swim_analyse.exe" ]] || die "缺少 $BIN/swim_analyse.exe，先跑 scripts/build.bat"
[[ -f "$ROOT/cpp/models/stitch.lut" ]] || die "缺少 cpp/models/stitch.lut，先跑 scripts/build.bat"

# engine 的文件名带身份戳（sm/TRT/batch/精度/onnx 戳），包里统一改成裸名 ——
# 目标机的身份戳必然与本机不同（ONNX 的 mtime 变了），裸名才是它能认的那条路
# （见 trt_engine.cpp 的三种来源：身份戳名 / 裸名 / 现烘）。
pick_engine() {   # $1=detect|pose  -> 打印路径
  local hit; hit="$(ls -t "$ROOT/cpp/models/$1".*.engine 2>/dev/null | head -1)"
  [[ -n "$hit" ]] || die "缺少 $1 的 engine，先在本机跑一次分析让它构建（scripts/analyse.bat）"
  echo "$hit"
}

# ── 清单：先把「包里该有什么」列全，再决定拷哪些 ─────────────────────────────
# 一份清单同时喂全能包与增量包，两者对「包里该有什么」的理解不可能分叉。
# 顺手拿到的 md5 又当了拷贝校验：engine 有一百多兆，一次静默的坏拷贝会让目标机报
# 「反序列化失败 / 与本机架构不匹配」，把人引向完全错误的方向（实测踩过一次）。
PLAN="$GEN/plan"                # 每行：<md5> <包内相对路径> <源文件绝对路径>
add() {   # $1=源  $2=包内相对路径（省略则取源的 basename）
  local src="$1" rel="${2:-$(basename "$1")}"
  [[ -f "$src" ]] || die "清单里的文件不存在: $src"
  printf '%s %s %s\n' "$(md5sum "$src" | cut -d' ' -f1)" "$rel" "$src" >> "$PLAN"
}

add "$BIN/swim_analyse.exe"
for f in "$BIN"/*.dll; do add "$f"; done
for d in nvinfer_10.dll nvonnxparser_10.dll; do
  [[ -f "$TRT_LIB/$d" ]] || die "缺少 $TRT_LIB/$d（用 SWIM_TRT_LIB= 指定 TRT 的 lib 目录）"
  add "$TRT_LIB/$d"
done
add "$CUDART"
for d in msvcp140.dll vcruntime140.dll vcruntime140_1.dll concrt140.dll; do
  add "$VCRT_DIR/$d"
done
add "$FFMPEG" ffmpeg.exe
add "$FFPROBE" ffprobe.exe
add "$(pick_engine detect)" models/detect.engine
add "$(pick_engine pose)"   models/pose.engine
add "$ROOT/cpp/models/stitch.lut"     models/stitch.lut
add "$ROOT/cpp/models/pose_meta.json" models/pose_meta.json

# ONNX + 构建资源：让目标机在架构不符时能自己现烘 engine（见文件头的理由）
if [[ "$LEAN" == 1 ]]; then
  say "精简模式：不带 ONNX 与 TRT 构建资源，换 GPU 架构即失效"
else
  for m in detect pose; do add "$ROOT/cpp/models/$m.onnx" "models/$m.onnx"; done
  BR="$TRT_LIB/nvinfer_builder_resource_10.dll"
  [[ -f "$BR" ]] || die "缺少 $BR（现烘 engine 要它；不想带就 SWIM_DIST_LEAN=1）"
  add "$BR"
fi

# ── 生成的文本文件 ──────────────────────────────────────────────────────────
# 相机名必须与 stitch.lut 里的一致（由 configs/pool_mesh.json 的 meshes 顺序定），
# 用「相机=地址」而不是按行序：现场 IP 尾数与 mesh 顺序不同，按行序写迟早错位。
# 含中文且给人在记事本里改，所以 UTF-8 带 BOM + CRLF（同 docs/windows.md 的约定）。
# 程序会剥掉首行 BOM（见 stitch_source.cpp 的 uris_from_list）。
bom_crlf() { { printf '\xEF\xBB\xBF'; cat; } | sed 's/$/\r/' > "$GEN/$1"; }
# .bat 必须 CRLF + 纯 ASCII（cmd.exe 按系统 ANSI 代码页解析），所以中文说明放在
# 同目录的 README.txt（UTF-8 带 BOM，记事本能正常打开）。
crlf()     { sed 's/$/\r/' > "$GEN/$1"; }

bom_crlf cameras.txt <<'EOF'
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
add "$GEN/cameras.txt" cameras.txt

# 两个入口只差三处：源的默认值、必需的模型文件、传 --input 还是 --cam-dir。
# 所以共同骨架（cd、绝对路径调 exe、shift 循环解析参数、退出码提示）由这里拼，
# 差异靠参数传进去 —— 两份 .bat 各自完整可读，但骨架只维护一份。
launcher() {   # $1=文件名 $2=默认源 $3=必需文件 $4=传给 exe 的参数名  stdin=头部说明
  { cat
    cat <<EOF

setlocal
cd /d "%~dp0"
rem Call the exe by its full path, not by bare name: some environments set
rem NoDefaultCurrentDirectoryInExePath=1 (Git Bash does), and then cmd refuses
rem to search the current directory - the launcher would fail with 9009.
set "EXE=%~dp0swim_analyse.exe"
rem Bundled ffmpeg/ffprobe and CUDA runtime live right here, so prepend this
rem folder to PATH: the target machine needs nothing installed.
set "PATH=%~dp0;%PATH%"

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

if not defined SRC set "SRC=$2"

if not exist "%EXE%" (
  echo [error] swim_analyse.exe missing - is this the unpacked dist folder?
  goto :end
)
if not exist "$3" (
  echo [error] $3 missing - incomplete package.
  goto :end
)
EOF
    [[ "$4" == --cam-dir ]] && cat <<'EOF'
if not exist "%SRC%" (
  echo [error] camera list not found: %SRC%
  echo         edit cameras.txt, or pass a list file as the first argument.
  goto :end
)
EOF
    cat <<EOF

echo.
echo   source : %SRC%
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" $4 "%SRC%" --models models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] exited with code %RC% - see README.txt

:end
echo.
echo Press any key to close...
pause >nul
endlocal
EOF
  } | crlf "$1"
  add "$GEN/$1" "$1"
}

launcher run_1cam.bat "rtsp://192.168.1.199/live_stream" "models\detect.engine" --input <<'EOF'
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
rem Every frame runs detect + pose + tracking; nothing is replayed.
rem Decoding goes through the bundled ffmpeg (about 78 fps here).
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.
EOF

launcher run_6cam.bat "cameras.txt" "models\stitch.lut" --cam-dir <<'EOF'
@echo off
rem Production six-camera run: pull six 4K streams, stitch on the GPU, analyse.
rem Nothing touches the CPU between decode and inference.
rem
rem   double-click                 read cameras.txt, live preview window
rem   run_6cam.bat --json out.json      also write per-swimmer results
rem   run_6cam.bat --out out.mp4        also write an annotated video
rem   run_6cam.bat <list.txt>      use a different camera list
rem
rem Edit cameras.txt to match the venue's IPs. Camera names on the left must
rem stay as they are - they are tied to the pool calibration in stitch.lut.
rem Six 4K streams -> NVDEC -> GPU stitch -> detect + pose, about 38 fps.
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.
EOF

ARCH="$(basename "$(pick_engine detect)" | sed 's/.*\.\(sm[0-9]*\)-.*/\1/')"
bom_crlf README.txt <<EOF
游泳动作分析 —— 交付包
========================

**目标机什么都不用装**（除了 NVIDIA 驱动）。CUDA 运行库、VC 运行库、ffmpeg
都在本目录里，入口脚本会把本目录加到 PATH。整个目录拷到哪都能跑，别拆散。

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
* NVIDIA 显卡，驱动 >= 550。就这一条。
* 换了显卡架构也能跑：包里预烘的 engine 是 $ARCH（RTX 40 系），架构不符时
  程序会用包内的 ONNX 现场重烘，**首次运行多花几分钟**，之后秒开。
  重烘的 engine 落在 models/ 下，带 sm/TRT 版本戳，不会覆盖原来那份。

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
  run_6cam.bat --out out.mp4        标注视频（编码会吃掉一半帧率）
  run_6cam.bat --max-frames 300     只跑前 300 帧
  run_6cam.bat --preview-scale 0.4  预览窗口缩放比
  swim_analyse.exe --help           全部参数

怎么更新
--------
开发侧双击 scripts\dist.bat inc 出一个增量包（通常只有几百 KB），把整个
swim_analyse_update 文件夹拷进本目录，双击里面的 update.bat 即可。
它只覆盖不删除，也不动 cameras.txt（现场 IP 在里面）。不要手工挑文件拷。

看不懂报错时
------------
"0xC0000135" / "-1073741515" / 启动即退，什么都不打印
    缺 DLL。本包自带了全部依赖，出现这个说明目录被拆散了或拷贝不完整
    —— 重新整目录拷一遍。

"[TRT] … 不可用（…），改用 ONNX 现烘"
    正常，换机器/换显卡后的第一次运行就这样，等几分钟。
    若紧跟着报"新烘的 engine 仍不可用"，才是真出了问题。

"预构建 engine … 不可用：…请在本机重新构建"
    只有精简包（SWIM_DIST_LEAN=1 打的）会这样，它不带 ONNX 无从重烘。
    找开发侧要一个全能包。

"9009" / "'swim_analyse.exe' 不是内部或外部命令"
    不该出现（入口用绝对路径调用）。若出现说明包被拆散了，exe 与 .bat 不同目录。

"读取失败(-138)，重试中"
    相机流断了或网络抖动。单次是正常的，持续出现要查网线与相机状态。

"直播流累计丢弃 N 帧"
    正常。这台机器追不上相机帧率时会丢旧帧保实时；N 持续暴涨说明该降负载。

性能参考（RTX 4080 Laptop）
---------------------------
  单相机直分析      约 78 fps
  六路现拼 + 分析   约 38 fps（瓶颈是 NVDEC，六路 4K 已把解码器跑满）
  加 --out 落盘     约 27~35 fps（libx264 与解码抢 CPU）
EOF
add "$GEN/README.txt" README.txt

# ── 出包 ────────────────────────────────────────────────────────────────────
# 拷完逐字节复核。目标是 engine/lut 这些一百多兆的大文件：一次静默的坏拷贝在
# 目标机上表现为「反序列化失败」，跟「换了显卡架构」的症状一模一样（实测踩过）。
copy_verified() {   # $1=清单里的 md5  $2=源  $3=目标
  mkdir -p "$(dirname "$3")"
  cp "$2" "$3"
  local b; b="$(md5sum "$3" | cut -d' ' -f1)"
  [[ "$1" == "$b" ]] || die "拷贝校验失败: $2 -> $3（md5 $1 vs $b，重跑一次）"
}

if [[ $INC -eq 1 ]]; then
  OUT="${OUT}_update"
  [[ -f "$MANIFEST" ]] ||
    die "找不到 $MANIFEST：增量是相对上一次全能包的，先跑一次 bash scripts/dist.sh"
fi

rm -rf "$OUT"
mkdir -p "$OUT"

n=0
total=0
while read -r md5 rel src; do
  # 增量包：md5 与上次全能包里一致就跳过。grep -x 整行匹配，避免「md5 相同但
  # 路径不同」或「路径相同但 md5 不同」被当成命中。
  if [[ $INC -eq 1 ]] && grep -qxF "$md5 $rel" "$MANIFEST"; then continue; fi
  copy_verified "$md5" "$src" "$OUT/$rel"
  n=$((n + 1)); total=$((total + $(stat -c %s "$src")))
done < "$PLAN"

if [[ $INC -eq 0 ]]; then
  # 清单只在全能包时写：增量包是相对「上次完整拷过去的那份」，不该移动这个基准
  cut -d' ' -f1,2 "$PLAN" > "$MANIFEST"
  say "全能包 $n 个文件 $(du -sh "$OUT" | cut -f1)$([[ "$LEAN" == 1 ]] && echo "（精简）")"
  say "engine: $(basename "$(pick_engine detect)") -> models/detect.engine"
  say "目标机零安装：自带 cudart / VC 运行库 / ffmpeg$([[ "$LEAN" == 1 ]] || echo " / ONNX + TRT 构建资源")"
elif [[ $n -eq 0 ]]; then
  rmdir "$OUT"
  say "增量包：与上次全能包无差异，什么都不用拷"
  exit 0
else
  # 覆盖脚本。刻意不删目标目录里的旧文件：增量只做覆盖，删文件要人来判断
  # （目标机上可能有现场改过的 cameras.txt、跑出来的 json、重烘的 engine）。
  cat > "$GEN/raw_update" <<'EOF'
@echo off
rem Apply this update to an unpacked swim_analyse folder.
rem Put this whole update folder inside the swim_analyse folder and double-click,
rem or: update.bat D:\path\to\swim_analyse
rem
rem Only copies; never deletes. cameras.txt is skipped even when it is in here:
rem the venue IPs live in that file, and clobbering them silently would take the
rem rig offline. Diff it by hand if this update mentions it.
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.

setlocal
cd /d "%~dp0"
set "DST=%~1"
if not defined DST set "DST=%~dp0.."

if not exist "%DST%\swim_analyse.exe" (
  echo [error] "%DST%" does not look like an unpacked swim_analyse folder.
  echo         Pass the target folder: update.bat D:\path\to\swim_analyse
  set "RC=1"
  goto :end
)

echo Updating "%DST%" ...
set "RC=0"
rem Loose files, then subdirectories (models\). Skip ourselves and cameras.txt.
rem "cmd && echo || set" rather than "if errorlevel": cmd splits a line on & or |
rem before evaluating an if, so "if errorlevel 1 set RC=1 & goto :eof" would jump
rem unconditionally. Failures only mark RC - a goto out of a called subroutine
rem would land in :end and pause once per file.
for %%f in (*) do call :one "%%~nxf"
for /d %%d in (*) do xcopy /y /i /q "%%d" "%DST%\%%d\" >nul && echo   %%d\ || set "RC=1"
if not "%RC%"=="0" goto :copyfail
echo.
echo Done. Launch run_6cam.bat in "%DST%".
goto :end

:one
if /I "%~1"=="update.bat" goto :eof
if /I "%~1"=="cameras.txt" (
  echo   cameras.txt  SKIPPED - your venue IPs are in it. Diff by hand.
  goto :eof
)
copy /y "%~1" "%DST%\" >nul && echo   %~1 || set "RC=1"
goto :eof

:copyfail
echo.
echo [error] copy failed - is the target folder in use or read-only?
echo         Close any running swim_analyse.exe and retry.

:end
echo.
echo Press any key to close...
pause >nul
endlocal & exit /b %RC%
EOF
  crlf update.bat < "$GEN/raw_update"
  cp "$GEN/update.bat" "$OUT/update.bat"
  say "增量包 $n 个文件 $(du -sh "$OUT" | cut -f1) + update.bat"
  say "用法：整个 $(basename "$OUT")/ 拷到目标机，双击里面的 update.bat"
fi

# find 的 %s 是字节；按大小降序列出，让人一眼看到大头在哪
find "$OUT" -type f -printf '%s %P\n' | sort -rn | head -8 |
  awk '{ printf "  %8.1f MB  %s\n", $1/1048576, $2 }'

if [[ $ZIP -eq 1 ]]; then
  # Git Bash 有时不带 zip（本机就没有），但 Windows 10+ 自带 bsdtar，
  # System32\tar.exe -a 按扩展名选压缩格式，能出标准 zip。两者都在 $OUT 的父目录
  # 里执行，压出来的顶层就是 <包名>/，解压即得一个完整目录。
  ZIPNAME="$(basename "$OUT").zip"
  ZIPPATH="$(dirname "$OUT")/$ZIPNAME"
  rm -f "$ZIPPATH"
  if command -v zip >/dev/null; then
    (cd "$(dirname "$OUT")" && zip -qr "$ZIPNAME" "$(basename "$OUT")")
  elif [[ -x /c/Windows/System32/tar.exe ]]; then
    (cd "$(dirname "$OUT")" &&
      /c/Windows/System32/tar.exe -a -c -f "$ZIPNAME" "$(basename "$OUT")")
  else
    die "既没有 zip 也没有 Windows 自带的 tar.exe，请手工压缩 $OUT"
  fi
  [[ -f "$ZIPPATH" ]] || die "压缩失败，没生成 $ZIPPATH"
  say "打包: $ZIPPATH  $(du -h "$ZIPPATH" | cut -f1)"
fi
