<#
打交付包。两级，对应「第一次拷过去」与「之后每次更新」：

  scripts\dist.bat            全能包  dist\swim_analyse\         约 3.0 GB
  scripts\dist.bat inc        增量包  dist\swim_analyse_update\  只含变了的文件
  scripts\dist.bat zip        顺手压一个同名 .zip
  powershell -File scripts\dist.ps1 -Out D:\x    换输出位置

平常双击 scripts\dist.bat 即可（它先跑 build.bat 再调本脚本）。本脚本用
PowerShell 而非 bash：交付链路只在 Windows 上跑，不该拖上 Git Bash 这个依赖
（cmd 调起的 bash 是非登录 shell，连 dirname/md5sum 都找不到）。

**全能包 = 目标机零安装**：exe、全部运行期 DLL、CUDA 运行库、VC 运行库、
ffmpeg/ffprobe、预烘 engine、ONNX + TRT 构建资源、拼接查找表、两个双击入口。
目标机只需要 NVIDIA 卡 + 驱动 >= 550，别的什么都不用装。

带上 ONNX（133 MB）与 nvinfer_builder_resource（1.7 GB）是刻意的：engine 与
「GPU 架构 + TRT 版本」烘死，换代机器上预烘的那份用不了；带着这两样它能自己现烘
（几分钟，之后秒开），不必回来重新打包。要省这 1.8 GB 就 SWIM_DIST_LEAN=1，
代价是换架构即失效（trt_engine.cpp 会给出「请在本机重新构建后再打包」）。

**换代显卡分两件事，别混**：engine 靠上面的 ONNX 现烘解决；**我们自己的 CUDA
kernel 只能靠编进 exe 的 cubin**，ONNX 帮不上。所以 build.bat 默认按 89;120
（RTX40 + RTX50/Blackwell）双架构编译，本脚本用 cuobjdump 核一遍并写进 README.txt。

**增量包 = 相对最近一次全能包的累积差异** + 一个 update.bat（拷进目标目录）。
通常只有 swim_analyse.exe 变（0.4 MB），改了权重才会带上 engine/ONNX。
按 dist\<名>.manifest 判断「上次全能包里是什么」，所以必须先打过一次全能包。
累积而非逐次差分：中间漏掉几个增量包也没关系，只应用最新的那个就对了。
#>
[CmdletBinding()]
param(
  [switch]$Inc,
  [switch]$Zip,
  [string]$Out
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 本脚本在 scripts\ 下，但所有路径都以仓库根为基准
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
if (-not $Out) { $Out = Join-Path $Root 'dist\swim_analyse' }

$Bin      = Join-Path $Root 'cpp\build\Release'
$TrtLib   = if ($env:SWIM_TRT_LIB) { $env:SWIM_TRT_LIB }
            else { 'D:\WindowsProject\workspace\TRT\TensorRT-10.11.0.33\lib' }
$Lean     = $env:SWIM_DIST_LEAN -eq '1'
$Manifest = "$Out.manifest"   # 全能包写、增量包读：上次包里有哪些文件与其 md5

# 生成的文本文件（清单模板 / 入口 / README）先落到这里，再与拷贝的文件走同一条
# 「md5 -> 比清单 -> 入包」的路，增量包才能自动带上改过的入口脚本。
$Gen = Join-Path $env:TEMP ('swimdist_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $Gen -Force | Out-Null

function Remove-Gen { Remove-Item -LiteralPath $Gen -Recurse -Force -EA SilentlyContinue }
function Say ($m) { Write-Host "[dist] $m" -ForegroundColor Green }
function Die ($m) { Write-Host "[dist] $m" -ForegroundColor Red; Remove-Gen; exit 1 }

# ── 依赖定位 ────────────────────────────────────────────────────────────────
# 这三样都不在仓库里，得去系统里找。找不到就停 —— 少一个都会让目标机在启动瞬间
# 以 -1073741515（找不到 DLL）死掉，什么都不打印，是最难被现场描述清楚的故障。
function Find-First {   # 返回第一个存在的路径（支持通配），都不存在则 $null
  foreach ($p in $args) {
    if (-not $p) { continue }
    $hit = Get-Item -Path $p -EA SilentlyContinue | Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  return $null
}

$cand = @()
if ($env:SWIM_CUDA_BIN) { $cand += Join-Path $env:SWIM_CUDA_BIN 'cudart64_12.dll' }
if ($env:CUDA_PATH)     { $cand += Join-Path $env:CUDA_PATH 'bin\cudart64_12.dll' }
$cand += 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin\cudart64_12.dll'
$Cudart = Find-First @cand
if (-not $Cudart) { Die '找不到 cudart64_12.dll（装 CUDA Toolkit 12.x，或用 SWIM_CUDA_BIN= 指定其 bin 目录）' }

# 只带 exe 真正 import 的四个（dumpbin /dependents 核过），不整目录搬 VC143.CRT
$cand = @()
if ($env:SWIM_VCRT) { $cand += Join-Path $env:SWIM_VCRT 'msvcp140.dll' }
$cand += 'C:\Program Files\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT\msvcp140.dll'
$vcrt = Find-First @cand
if (-not $vcrt) { Die '找不到 VC 运行库 msvcp140.dll（用 SWIM_VCRT= 指定 Microsoft.VC143.CRT 目录）' }
$VcrtDir = Split-Path -Parent $vcrt

# ffmpeg/ffprobe：winget 装的是软链接，Copy-Item 会跟过去取到真身（231 MB×2）
$Ffmpeg  = (Get-Command ffmpeg.exe  -EA SilentlyContinue).Source
$Ffprobe = (Get-Command ffprobe.exe -EA SilentlyContinue).Source
if (-not $Ffmpeg)  { Die 'PATH 里没有 ffmpeg（winget install Gyan.FFmpeg）' }
if (-not $Ffprobe) { Die 'PATH 里没有 ffprobe（同上）' }

if (-not (Test-Path -LiteralPath (Join-Path $Bin 'swim_analyse.exe'))) {
  Die "缺少 $Bin\swim_analyse.exe，先跑 scripts\build.bat"
}
if (-not (Test-Path -LiteralPath (Join-Path $Root 'cpp\models\stitch.lut'))) {
  Die '缺少 cpp\models\stitch.lut，先跑 scripts\build.bat'
}

# exe 里带了哪些 GPU 架构的机器码。engine 能靠包内 ONNX 在目标机现烘，**我们自己的
# CUDA kernel 不能** —— 它只有编进 exe 的那几档 cubin，缺了就退化成驱动 JIT 那份
# PTX（首次启动多等几十秒，且是没验证过的路径）。build.bat 默认 89;120 覆盖
# RTX40/50，这里只是拦住「用单架构构建目录打包」的情况，不改行为。
# 优先问 cuobjdump（查的是真二进制），没装 CUDA 工具链就退到 CMakeCache 的配置值。
$Arches = @()
if (Get-Command cuobjdump.exe -EA SilentlyContinue) {
  $Arches = (cuobjdump.exe --list-elf (Join-Path $Bin 'swim_analyse.exe') 2>$null |
             Select-String -Pattern 'sm_\d+a?' -AllMatches).Matches.Value | Sort-Object -Unique
}
if (-not $Arches) {
  $cache = Join-Path $Root 'cpp\build\CMakeCache.txt'
  if (Test-Path -LiteralPath $cache) {
    $line = Select-String -LiteralPath $cache -Pattern '^CMAKE_CUDA_ARCHITECTURES:.*=(.*)$' |
            Select-Object -First 1
    if ($line) { $Arches = $line.Matches[0].Groups[1].Value -split '[;, ]+' |
                           Where-Object { $_ } | ForEach-Object { "sm_$_" } }
  }
}
$missing = @('sm_89', 'sm_120') | Where-Object { $Arches -notcontains $_ }
# 按数字排序而不是字符串（否则 sm_120 排在 sm_89 前面，读起来像漏了低档）
$Arches = $Arches | Sort-Object { [int]($_ -replace '\D', '') }
if ($Arches -and $missing) {
  Say "警告：exe 只含 $($Arches -join ' ')，缺 $($missing -join ' ')"
  Say '     那些卡上只能靠驱动 JIT。要原生支持就 set SWIM_CUDA_ARCH=89;120 后重跑 build.bat'
}

# engine 的文件名带身份戳（sm/TRT/batch/精度/onnx 戳），包里统一改成裸名 ——
# 目标机的身份戳必然与本机不同（ONNX 的 mtime 变了），裸名才是它能认的那条路
# （见 trt_engine.cpp 的三种来源：身份戳名 / 裸名 / 现烘）。
function Get-Engine ($what) {   # detect|pose -> 最新那份的完整路径
  $hit = Get-ChildItem -Path (Join-Path $Root "cpp\models\$what.*.engine") -EA SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $hit) { Die "缺少 $what 的 engine，先在本机跑一次分析让它构建（scripts\analyse.bat）" }
  return $hit.FullName
}

# ── 清单：先把「包里该有什么」列全，再决定拷哪些 ─────────────────────────────
# 一份清单同时喂全能包与增量包，两者对「包里该有什么」的理解不可能分叉。
# 顺手拿到的 md5 又当了拷贝校验：engine 有一百多兆，一次静默的坏拷贝会让目标机报
# 「反序列化失败 / 与本机架构不匹配」，把人引向完全错误的方向（实测踩过一次）。
$Plan = New-Object Collections.ArrayList   # 每项：@{ Md5; Rel; Src }
function Get-Md5 ($p) { (Get-FileHash -LiteralPath $p -Algorithm MD5).Hash.ToLower() }
function Add-File ($src, $rel) {   # $rel 省略则取源的文件名
  if (-not $rel) { $rel = Split-Path -Leaf $src }
  if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { Die "清单里的文件不存在: $src" }
  $Plan.Add(@{ Md5 = (Get-Md5 $src); Rel = $rel; Src = $src }) | Out-Null
}

Add-File (Join-Path $Bin 'swim_analyse.exe')
Get-ChildItem -Path (Join-Path $Bin '*.dll') | ForEach-Object { Add-File $_.FullName }
foreach ($d in 'nvinfer_10.dll', 'nvonnxparser_10.dll') {
  $p = Join-Path $TrtLib $d
  if (-not (Test-Path -LiteralPath $p)) { Die "缺少 $p（用 SWIM_TRT_LIB= 指定 TRT 的 lib 目录）" }
  Add-File $p
}
Add-File $Cudart
foreach ($d in 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll') {
  Add-File (Join-Path $VcrtDir $d)
}
Add-File $Ffmpeg  'ffmpeg.exe'
Add-File $Ffprobe 'ffprobe.exe'
$DetectEngine = Get-Engine detect
Add-File $DetectEngine 'models\detect.engine'
Add-File (Get-Engine pose) 'models\pose.engine'
Add-File (Join-Path $Root 'cpp\models\stitch.lut')     'models\stitch.lut'
Add-File (Join-Path $Root 'cpp\models\pose_meta.json') 'models\pose_meta.json'

# ONNX + 构建资源：让目标机在架构不符时能自己现烘 engine（见文件头的理由）
if ($Lean) {
  Say '精简模式：不带 ONNX 与 TRT 构建资源，换 GPU 架构即失效'
} else {
  foreach ($m in 'detect', 'pose') {
    Add-File (Join-Path $Root "cpp\models\$m.onnx") "models\$m.onnx"
  }
  $br = Join-Path $TrtLib 'nvinfer_builder_resource_10.dll'
  if (-not (Test-Path -LiteralPath $br)) { Die "缺少 $br（现烘 engine 要它；不想带就 SWIM_DIST_LEAN=1）" }
  Add-File $br
}

# ── 生成的文本文件 ──────────────────────────────────────────────────────────
# 两种编码，各有硬理由，别混（同 docs\windows.md 的约定）：
#   .txt 给人在记事本里改，含中文 -> UTF-8 带 BOM + CRLF（老记事本无 BOM 按 ANSI 解）
#   .bat 由 cmd.exe 解析        -> UTF-8 无 BOM + CRLF + 纯 ASCII（BOM 会被当成
#        第一条命令的一部分，中文按系统 ANSI 代码页解，换台机器就是乱码）
# 程序会剥掉 cameras.txt 首行的 BOM（见 stitch_source.cpp 的 uris_from_list）。
function Write-Text ($path, $text, $bom) {
  # 统一成 CRLF 并补上末行换行：这些文件要被 cmd.exe 解析或给人在记事本里改，
  # 缺末尾换行时最后一条命令在某些 cmd 版本下会被吞掉。
  $text = (($text -replace "`r`n", "`n").TrimEnd("`n") + "`n") -replace "`n", "`r`n"
  [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $bom))
}
function Write-Gen ($name, $text, $bom) {
  Write-Text (Join-Path $Gen $name) $text $bom
  Add-File (Join-Path $Gen $name) $name
}

# 相机名必须与 stitch.lut 里的一致（由 configs\pool_mesh.json 的 meshes 顺序定），
# 用「相机=地址」而不是按行序：现场 IP 尾数与 mesh 顺序不同，按行序写迟早错位。
Write-Gen cameras.txt @'
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
'@ $true

# 两个入口只差三处：源的默认值、必需的模型文件、传 --input 还是 --cam-dir。
# 所以共同骨架（cd、绝对路径调 exe、shift 循环解析参数、退出码提示）由这里拼，
# 差异靠参数传进去 —— 两份 .bat 各自完整可读，但骨架只维护一份。
function Write-Launcher ($name, $head, $default, $need, $flag) {
  # 只有 --cam-dir 要查源：它必须是一份清单文件，缺了就得把人指回 cameras.txt。
  # --input 的源可以是 rtsp:// 也可以是视频文件，交给 exe 自己判断。
  $srcCheck = if ($flag -eq '--cam-dir') { @'
if not exist "%SRC%" (
  echo [error] camera list not found: %SRC%
  echo         edit cameras.txt, or pass a list file as the first argument.
  goto :end
)

'@ } else { '' }
  Write-Gen $name @"
$head

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
rem Two separate ifs, NOT `if defined A if not "%A:~0,1%"=="-"`: cmd expands the
rem whole line before running any of it, so ":~0,1" is applied even when A is
rem empty, expands to the bare text ~0,1 and eats the quotes - the line ends up
rem unbalanced and cmd aborts with "The syntax of the command is incorrect."
rem That is the no-argument path, i.e. exactly what a double-click does.
set "SRC="
set "ARGS="
set "A=%~1"
if not defined A goto :collect
if "%A:~0,1%"=="-" goto :collect
set "SRC=%~1"
shift
:collect
if "%~1"=="" goto :parsed
set "ARGS=%ARGS% %1"
shift
goto :collect
:parsed

if not defined SRC set "SRC=$default"

if not exist "%EXE%" (
  echo [error] swim_analyse.exe missing - is this the unpacked dist folder?
  goto :end
)
if not exist "$need" (
  echo [error] $need missing - incomplete package.
  goto :end
)
$srcCheck
echo.
echo   source : %SRC%
echo   press q or ESC on the preview window to stop
echo.

"%EXE%" $flag "%SRC%" --models models --preview --show-fps%ARGS%
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [error] exited with code %RC% - see README.txt

:end
echo.
echo Press any key to close...
pause >nul
endlocal
"@ $false
}

Write-Launcher run_1cam.bat @'
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
rem In the preview window: 1 = keypoints, 2 = boxes/labels, 3 = metre grid,
rem q/ESC = quit. Resize or maximize it freely - the picture is scaled to
rem fit and stays centred.
rem
rem Every frame runs detect + pose + tracking; nothing is replayed.
rem Decoding goes through the bundled ffmpeg (about 78 fps here).
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.
'@ 'rtsp://192.168.1.199/live_stream' 'models\detect.engine' '--input'

Write-Launcher run_6cam.bat @'
@echo off
rem Production six-camera run: pull six 4K streams, stitch on the GPU, analyse.
rem Nothing touches the CPU between decode and inference.
rem
rem   double-click                 read cameras.txt, live preview window
rem   run_6cam.bat --json out.json      also write per-swimmer results
rem   run_6cam.bat --out out.mp4        also write an annotated video
rem   run_6cam.bat --fps 30        force the time axis to 30fps
rem   run_6cam.bat <list.txt>      use a different camera list
rem
rem In the preview window: 1 = keypoints, 2 = boxes/labels, 3 = metre grid,
rem q/ESC = quit. Toggles also apply to what --out writes. Resize or
rem maximize the window freely - the picture is scaled to fit and centred.
rem
rem Edit cameras.txt to match the venue's IPs. Camera names on the left must
rem stay as they are - they are tied to the pool calibration in stitch.lut.
rem Six 4K streams -> NVDEC -> GPU stitch -> detect + pose, about 38 fps.
rem
rem ASCII only on purpose: cmd.exe parses .bat with the system ANSI codepage.
rem Chinese notes are in README.txt.
'@ 'cameras.txt' 'models\stitch.lut' '--cam-dir'

$Arch = if ((Split-Path -Leaf $DetectEngine) -match '\.(sm\d+)-') { $Matches[1] } else { '未知架构' }
$ArchList = if ($Arches) { $Arches -join ' ' } else { '未探测' }
$Readme = @"
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
* 支持的显卡代次：本包的 exe 编进了 $ArchList 的机器码
  （sm_89 = RTX 40 系，sm_120 = RTX 50 系 / Blackwell）。这两代开箱即用。
  更老的卡（30 系 sm_86 等）也能跑，但 CUDA kernel 要由驱动现场 JIT，
  首次启动多等几十秒。
* 换了显卡代次时 TensorRT engine 会自动重烘：包里预烘的那份是 $Arch，
  与本机不符时程序用包内的 ONNX 现场重建，**首次运行多花几分钟**，之后秒开。
  重烘的 engine 落在 models/ 下，带 sm/TRT 版本戳，不会覆盖原来那份。

相机侧配置（4K + 30fps）
------------------------
movfmt 决定主流的分辨率与帧率，**它在录制中是只读的** —— 查询会看到 "ro":1，
写入返回 code:-1 且不报原因。所以要先停录：

  curl "http://<ip>/ctrl/rec?action=stop"
  curl "http://<ip>/ctrl/set?movfmt=4KP29.97"
  curl "http://<ip>/ctrl/stream_setting?index=stream0&venc=h264"
  curl "http://<ip>/ctrl/get?k=movfmt"          回读校验，必须看到 4KP29.97

回读只说明相机认了；流里自报的帧率**可能仍是 59.94**（实测该字段跟不上 movfmt
切换）。时间轴按错帧率走会让速度整体翻倍，这时加 --fps 30 把它摆正 ——
它只改时间轴，不改解码。

RTSP 只有一个挂载点 rtsp://<ip>/live_stream，它给出的是相机当前 send_stream
选中的那一路（默认 Stream0，即 4K 主流）。同一台相机最多约 4 个并发会话。

常用参数
--------
  run_6cam.bat --json out.json      每个泳者的划水次数与末速
  run_6cam.bat --out out.mp4        标注视频（编码会吃掉一半帧率）
  run_6cam.bat --max-frames 300     只跑前 300 帧
  run_6cam.bat --preview-scale 0.4  预览窗口缩放比
  run_6cam.bat --fps 30             按 30fps 算时间轴（流报错帧率时用）
  swim_analyse.exe --help           全部参数

预览窗口的热键（焦点要在窗口上）
--------------------------------
  1     人体关键点（骨架）开关
  2     分析框与 ID/划水/速度 标签开关
  3     米制标尺网格开关（每 1 m 一条，5 m 加粗标米数）
  q/ESC 退出

窗口可随意拖拽缩放或最大化：画面按窗口等比放到最大并居中，四周补黑边，
不会只占左上一角。--preview-scale 只决定初始窗口大小。

窗口里看到的就是 --out 写进 mp4 的：热键切换会同时作用于落盘。
想一开始就是某个状态，用 --no-kpts / --no-boxes / --grid。

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
"@
Write-Gen README.txt $Readme $true

# ── 出包 ────────────────────────────────────────────────────────────────────
# 拷完逐字节复核。目标是 engine/lut 这些一百多兆的大文件：一次静默的坏拷贝在
# 目标机上表现为「反序列化失败」，跟「换了显卡架构」的症状一模一样（实测踩过）。
function Copy-Verified ($md5, $src, $dst) {
  $dir = Split-Path -Parent $dst
  if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Copy-Item -LiteralPath $src -Destination $dst -Force   # 会跟随 winget 的软链接取到真身
  $b = Get-Md5 $dst
  if ($b -ne $md5) { Die "拷贝校验失败: $src -> $dst（md5 $md5 vs $b，重跑一次）" }
}

if ($Inc) {
  $Out = "${Out}_update"
  if (-not (Test-Path -LiteralPath $Manifest)) {
    Die "找不到 $Manifest：增量是相对上一次全能包的，先跑一次 scripts\dist.bat"
  }
}

Remove-Item -LiteralPath $Out -Recurse -Force -EA SilentlyContinue
New-Item -ItemType Directory -Path $Out -Force | Out-Null

# 增量包：md5 与上次全能包里一致就跳过。整行比对（md5 + 包内路径），避免
# 「md5 相同但路径不同」或「路径相同但 md5 不同」被当成命中。
# @() 包一层：清单只剩一行时 Get-Content 返回单个字符串，直接转 HashSet[string]
# 会按 IEnumerable<char> 拆成一堆单字符。
$Base = if ($Inc) {
  [Collections.Generic.HashSet[string]] @(Get-Content -LiteralPath $Manifest)
} else { $null }
$n = 0
foreach ($it in $Plan) {
  if ($Inc -and $Base.Contains("$($it.Md5) $($it.Rel)")) { continue }
  Copy-Verified $it.Md5 $it.Src (Join-Path $Out $it.Rel)
  $n++
}

function Show-Size ($p) {   # 目录或文件的总字节 -> 人话
  $b = (Get-ChildItem -LiteralPath $p -Recurse -File | Measure-Object -Sum Length).Sum
  if (-not $b) { $b = (Get-Item -LiteralPath $p).Length }
  if ($b -ge 1GB) { '{0:N1} GB' -f ($b / 1GB) } else { '{0:N1} MB' -f ($b / 1MB) }
}

if (-not $Inc) {
  # 清单只在全能包时写：增量包是相对「上次完整拷过去的那份」，不该移动这个基准
  Write-Text $Manifest (($Plan | ForEach-Object { "$($_.Md5) $($_.Rel)" }) -join "`n") $false
  Say "全能包 $n 个文件 $(Show-Size $Out)$(if ($Lean) { '（精简）' })"
  Say "engine: $(Split-Path -Leaf $DetectEngine) -> models\detect.engine"
  Say "目标机零安装：自带 cudart / VC 运行库 / ffmpeg$(if (-not $Lean) { ' / ONNX + TRT 构建资源' })"
} elseif ($n -eq 0) {
  Remove-Item -LiteralPath $Out -Force
  Say '增量包：与上次全能包无差异，什么都不用拷'
  Remove-Gen
  exit 0
} else {
  # 覆盖脚本。刻意不删目标目录里的旧文件：增量只做覆盖，删文件要人来判断
  # （目标机上可能有现场改过的 cameras.txt、跑出来的 json、重烘的 engine）。
  $upd = @'
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
'@
  Write-Text (Join-Path $Out 'update.bat') $upd $false
  Say "增量包 $n 个文件 $(Show-Size $Out) + update.bat"
  Say "用法：整个 $(Split-Path -Leaf $Out)\ 拷到目标机，双击里面的 update.bat"
}

# 按大小降序列出，让人一眼看到大头在哪
Get-ChildItem -LiteralPath $Out -Recurse -File | Sort-Object Length -Descending |
  Select-Object -First 8 | ForEach-Object {
    '{0,10:N1} MB  {1}' -f ($_.Length / 1MB), $_.FullName.Substring($Out.Length + 1)
  }

if ($Zip) {
  # Compress-Archive（5.1 自带）有 2 GB 单条目上限，全能包里的
  # nvinfer_builder_resource 1.7 GB 已经贴脸，所以优先用 Windows 自带的 bsdtar
  # （System32\tar.exe -a 按扩展名选格式，出标准 zip），它没有这个限制。
  # 两者都在 $Out 的父目录里执行，压出来的顶层就是 <包名>\，解压即得完整目录。
  $zipName = (Split-Path -Leaf $Out) + '.zip'
  $parent  = Split-Path -Parent $Out
  $zipPath = Join-Path $parent $zipName
  Remove-Item -LiteralPath $zipPath -Force -EA SilentlyContinue
  $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
  if (Test-Path -LiteralPath $tar) {
    Push-Location -LiteralPath $parent
    try { & $tar -a -c -f $zipName (Split-Path -Leaf $Out) } finally { Pop-Location }
  } else {
    Compress-Archive -Path $Out -DestinationPath $zipPath
  }
  if (-not (Test-Path -LiteralPath $zipPath)) { Die "压缩失败，没生成 $zipPath" }
  Say "打包: $zipPath  $(Show-Size $zipPath)"
}

Remove-Gen

