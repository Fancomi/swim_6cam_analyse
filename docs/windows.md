# Windows 环境说明与踩过的坑

面向在 Windows 上构建/运行本项目的人。**Linux 上不需要读这页。**
构建与运行命令本身在 `README.md`（Python）与 `cpp/README.md`（C++），这里只写
Windows 特有的环境要求、编码规则与已知坑；每条坑的修复都已提交在代码里，
列在这里是为了别再踩一次。

## 1. 环境要求（C++ 链路）

| 组件 | 版本 | 说明 |
|---|---|---|
| Windows | 10/11 x64 | 实测 Win11 26200 |
| Visual Studio | 2022 | 需 "使用 C++ 的桌面开发" 工作负载 |
| CUDA Toolkit | 12.x | 与驱动匹配（实测 12.8）；**50 系卡需 ≥12.8**（Blackwell） |
| TensorRT | **10.11.0.33** | 官方 Windows zip，解压即可。TRT 11 移除了 `BuilderFlag::kFP16`，本代码只针对 10.x |
| OpenCV | 4.x | 实测 vcpkg `opencv4:x64-windows`（4.10）。`--preview` 需要 `highgui`。**它不含 ffmpeg 特性**，见第 3 节 |
| FFmpeg libav*（可选） | 含 `h264_cuvid` | 只有六路现拼 `--cam-dir` 需要。vcpkg 装 `ffmpeg[core,avcodec,avformat,nvcodec]:x64-windows`（实测 7.1.2），`find_package(FFMPEG)` 自动找到。缺了只是这一路不编 |
| CMake | ≥3.18 | |
| ffmpeg CLI | 任意近期版本 | 在 `PATH` 里。**`--input` 的默认解码路径与 `--out` 编码都要它** |

`scripts/build.bat` 里三个可覆盖的环境变量：`SWIM_TRT_ROOT`、`SWIM_VCPKG`、`SWIM_CUDA_ARCH`
（RTX40=`89`，RTX50=`120`，RTX30=`86`；不要用 CMake 默认的 `86 89 90`，多架构编译慢很多）。
运行期还要让 TRT 的 DLL 可见 —— `scripts/env.bat` 会把 `SWIM_TRT_LIB`（默认
`<TRT_ROOT>\lib`）加进 `PATH`，三个 `.bat` 都 `call` 它。
vcpkg 的 DLL（OpenCV / libav*）由 CMake 自动拷到 exe 旁，不必手动加 `PATH`。

**库名差异**：Windows 的 TRT 导入库带版本后缀（`nvinfer_10.lib` / `nvonnxparser_10.lib`），
Linux 是 `libnvinfer.so`。`cpp/CMakeLists.txt` 两套名字都找。

**engine 与 GPU + TRT 版本绑定**：Linux 上构建的 `.engine` 在 Windows 无效，
首次运行会自动重建（几分钟）。`.onnx` 也不入库，`scripts/build.bat` 会从 `weights/` 导出。

## 2. 脚本编码规则（改脚本前必读）

根因是 Windows 按**系统 ANSI 代码页**（中文机 = 936）解无 BOM 的文件。

| 类型 | 要求 | 不遵守的后果 |
|---|---|---|
| `.bat` / `.cmd` | UTF-8 **无 BOM** + CRLF + **内容纯 ASCII** | BOM 会被 `cmd.exe` 当命令（`@echo off` 变乱码）；中文在非 936 机器上乱码 |
| `.ps1`（含中文） | UTF-8 **带 BOM** + CRLF | PowerShell 5.1 按 ACP 解码，中文字节被解成全角字符吞掉引号 → `Missing closing '}'`，报错行还在别处 |
| `.txt`（给人读） | UTF-8 带 BOM + CRLF | 老记事本按 ANSI 解无 BOM 文件 |
| `.sh` / `.py` / `.cpp` / `.md` | UTF-8 无 BOM + LF | `.gitattributes` 已按类型钉死行尾，别绕过它 |

**改这些文件只做单字节级精确替换，禁止「整文件读出 → 改字符串 → 写回」**：
实测这条链路会写坏 UTF-8 字节且难以定位。需要大改先 `git checkout` 还原。
验证 `.ps1` 只信 Windows PowerShell 5.1：

```powershell
powershell.exe -NoProfile -Command "[System.Management.Automation.Language.Parser]::ParseFile('<绝对路径>',[ref]$null,[ref]$err)|Out-Null; if($err){$err}else{'PARSE OK'}"
```

源码侧同源的两个坑：MSVC 默认按 ACP 解 UTF-8 无 BOM 的 `.cpp`，会把中文注释的字节
吃进语法 → CMake 已给 `/utf-8`（CXX）与 `-Xcompiler=/utf-8`（CUDA），
运行期 `main()` 调 `SetConsoleOutputCP(CP_UTF8)`。
但 **`/utf-8` 救不了 nvcc**：它的 EDG 前端在 ACP=936 下按 ANSI 解 `.cu`，中文**字面量**
会被吞掉右引号（加 BOM 也无效），所以 `kernels.cu` 里字面量一律 ASCII、只有注释用中文。

## 3. 已知坑（都已修，列出来是为了别回退）

1. **`#include <dlfcn.h>` 在 MSVC 不存在** —— `frame_source.cpp` 原有一段
   `dlopen("libnvcuvid.so")` 的 NVDEC 探测，已整段删除。探到 dll 也不代表能解本画布
   （宽 5002 > CUVID 的 4096 上限），判据本身是错的。
2. **`windows.h` 的 `min`/`max` 宏**会打断 `std::max` 与 OpenCV 模板调用 ——
   include 之前必须 `#define NOMINMAX`。现在全仓库只有 `cpp/src/proc.cpp` 一个 TU
   include `windows.h`（子进程与管道都收在 `Proc` 里），其余文件不要再引它。
3. **vcpkg 的 opencv4 没有 ffmpeg 特性** —— videoio 只剩 MSMF/DSHOW，而 MSMF 的 H.264
   编码器在 1080p/4K 能开，**在画布 5002×2102 直接 `isOpened()==false`**（读没问题，写不了）。
   所以 `Writer` 的回退（起 `ffmpeg` 子进程喂 rawvideo，`-c:v libx264 -preset veryfast -crf 23`）
   在 Windows 是**必经路径**，`--out` 要求 `PATH` 里有 ffmpeg。
   读侧也已默认走 ffmpeg 管道：不只是快（13.0 vs 16.9 ms/帧），MSMF 的 YUV→BGR 换算
   与 swscale 不一致（逐像素平均差约 4.5/255），只有 ffmpeg 路径与 Python 的 `cv2` 逐字节相同。
   `--decoder cpu` 仅作兜底排障，它的数字是另一组合法结果，不是回归。
   注意两条链路的编码参数不同：C++ 的 `Writer` 用 `veryfast/crf23`（实时优先），
   Python 的 `render()` 用 `medium/crf20`（离线画质优先），产物体积不可直接比。
4. **本机 NVENC 用不了** —— 驱动 571.96 只提供 NVENC API 13.0，ffmpeg 8.1 要求 13.1
   （报 `Driver does not support the required nvenc API version`）。故编码器定为 libx264；
   换新驱动后把 `Writer` 的命令改 `-c:v h264_nvenc` 即可提速。
5. **装 torch 会把 numpy 顶到 2.x** —— `import torch` 报
   `Failed to initialize NumPy: _ARRAY_API not found`（torch 的 C 扩展按 numpy1 ABI 编译）。
   必须在装 mmcv/mmpose **之前** `pip install numpy==1.26.4 "setuptools<81"`。`scripts/install.sh` 已按序处理。
6. **`python3` 可能是 WindowsApps 空壳**（只会弹应用商店）—— `scripts/install.sh` 逐个候选试
   3.10 并在最后回退 `py -3.10`。
7. **`export_onnx.py` 会在 `weights/` 下留一个中间 `yolo_swim_detect.onnx`** ——
   ultralytics 先写在 `.pt` 旁边再 `os.replace`，可手动删掉。
8. **`av_hwdevice_ctx_create` 的顺序坑（六路现拼）** —— 传
   `AV_CUDA_USE_PRIMARY_CONTEXT` 时它要设置 primary context 的 flags，若 CUDA runtime
   API（`cudaSetDevice`/`cudaMalloc`）已经激活过它，就返回 `-129`
   （`Primary context already active with incompatible flags`）。所以
   `StitchFrameSource` 的构造顺序是「先建 hwdevice，再 `cudaMalloc` 查找表」，不能调换。
   不传这个 flag 也能解码（FFmpeg 自建 context，靠 UVA 让裸指针可寻址），但那样
   TensorRT 与 kernel 与解码器不在同一 context，跨 context 的裸指针访问不受保证。
9. **h264_cuvid 的 `hw_frames_ctx` 可能为 NULL** —— 它自己管 surface 池，不走 FFmpeg 的
   frames context。读 `ctx->hw_frames_ctx->data` 前必须判空，否则段错误
   （调试时表现为「解出第一帧就崩」）。`sw_format` 要从 `frame->hw_frames_ctx` 拿。
   另外 `surfaces` 选项已废弃，用 `AVCodecContext::extra_hw_frames` 加余量：
   解码帧要一直扣到拼接读完才能 free，给不够会在 `avcodec_receive_frame` 处卡死。
10. **奇数画布让 libx264 直接开不了** —— 拼接的 logical 画布是 5001×2101，`--out` 会在
    第 3 帧短写、`moov atom not found`。所以烘表时就把画布补到偶数 5002×2102，
    多出的一列一行没有 lane 覆盖、自然写黑，与上游 `encoded_width/height` 的约定一致。

## 4. Python 环境（参考实现用，Windows 实测顺序）

**Python 必须 3.10**（mmcv 预编译 wheel 只提供 cp310）。`scripts/install.sh` 在 Git Bash 里可直接跑，
它会自动选 `Scripts/python.exe`。想手动装就照这个顺序（**顺序不能乱**，第 3 步是关键）：

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip.exe install numpy==1.26.4 "setuptools<81"
.venv\Scripts\pip.exe install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install --no-deps mmpose==1.3.1
.venv\Scripts\pip.exe install --no-deps -e .
```

实测组合：Python 3.10.11 / torch 2.1.0+cu121 / opencv-python 4.10.0 / mmcv 2.1.0 /
mmpose 1.3.1 / mmdet 3.2.0。自检 `bash scripts/test.sh`。

## 5. 调试技巧（Windows 特有）

- **中文输出乱码**：先查是不是漏了 `/utf-8`、`SetConsoleOutputCP`，或 Python 侧忘了
  `PYTHONUTF8=1`（`scripts/` 下的三个 `.sh` 都已导出）。
- **显存量**：用运行前后 `nvidia-smi --query-gpu=memory.used` 的差值。
  `--query-compute-apps=used_memory` 在 Windows WDDM 下返回 `[N/A]`，别用它。
  六路现拼还要看 `--query-gpu=utilization.decoder`：它是 NVDEC 的独立利用率，
  跑满 100% 就说明瓶颈在解码器而不是 SM（`utilization.gpu` 那时只有 23%）。
- **渲染产物校验**：`ffprobe` 看帧数/分辨率，再 `ffmpeg -i out.mp4 -f null -` 全量解码
  确认无坏帧（走 ffmpeg 管道时若中途断流，这里会暴露）。
- **路径**：`.bat` 用 `\`；bash 脚本用 Git Bash 跑。注意 Git Bash 的 `/tmp` 对
  Windows 版 Python 的 `open`/`subprocess` 不可见。往 `PATH` 里加目录也要注意：
  Git Bash 的 `PATH` 以 `:` 分隔，`"D:/x"` 会被拆成 `D` 和 `/x` 两项而**静默失效**
  （症状是 exe 报找不到 DLL）。先 `cygpath -u` 转成 `/d/x` 再加，`scripts/test.sh` 已这么做。
- **拼接对照**：`--dump-canvas f0.png` 把首帧画布原样落盘（未画标注），与离线参考逐像素
  比。这比比对分析数字灵敏得多 —— 0.34 灰阶的画布差就能让 track 数从 63 变 88。
- **对照与基线**：见 `CLAUDE.md` 的「怎么验证一处改动」与「数字的口径」。
