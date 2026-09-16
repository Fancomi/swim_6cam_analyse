# Linux 加速链路部署踩坑（从裸机克隆到跑通 C++/TRT）

> 面向对象：在一台**全新的 Linux 开发/加速机**（非 Windows 交付机）上，从
> `git clone` 到跑通 C++ Plan C（画布 detect + RTMPose，TensorRT 加速）。
> Windows 工程侧的坑另见 `docs/windows.md`；本页只记 Linux 加速链路。
>
> 本页记的都是**实测踩过、且 CLAUDE.md 主线没单独展开**的点。按顺序照做即可复现。

## 参考环境（已验证）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| OS | Ubuntu 22.04 | |
| GPU | H800 80G（sm_90） | `-DCMAKE_CUDA_ARCHITECTURES=90` |
| nvcc | 12.9 | ≥12.8 才支持 sm_120，本机不涉及 |
| gcc | 11.4 | |
| cmake | 3.18 | 与 `CMakeLists.txt` 的 `cmake_minimum_required` 相同，勿更低 |
| TensorRT | 10.11.0.33 | 与 Windows 侧钉死同一版，换版本 engine 全失效 |
| OpenCV | 4.5.4（系统 apt 包） | cmake 命中 `/usr/lib/x86_64-linux-gnu/cmake/opencv4` |
| ffmpeg CLI | 7.0.2 static | 画布 `--input` 主链路只用 CLI |
| libav*（dev） | 58.x（apt） | 六路 `--cam-dir` 拼接需要，且必须带 `h264_cuvid` |
| torch / ultralytics / mmpose / mmcv | 2.3.1+cu121 / 8.4.62 / 1.3.2 / 2.1.0 | 仅导出 ONNX 时用 |

## 坑 1：权重是 Git LFS 指针，且机器常常没装 git-lfs

`weights/*.pt|*.pth` 走 LFS。裸克隆只会拿到 **134 字节的指针文本**，Python/导出会以
"不是合法 checkpoint" 失败。而不少精简的训练镜像**根本没装 git-lfs**
（`git lfs` 报 `'lfs' is not a git command`）。

```bash
sudo apt-get install -y git-lfs          # 镜像没预装时
git lfs install --local
git lfs pull                             # 或只取加速链路要的两个：
git lfs pull --include="weights/yolo_swim_detect.pt,weights/plans/planC_rtmpose_m_canvas.pth"
```

加速链路（C++ Plan C）只需 **detect** 与 **planC pose** 两个权重；planB、原相机
rtmpose 是 Python 侧 Plan A/B 用的，可不拉。

## 坑 2：trt_local 的 TensorRT 可能缺 `NvOnnxParser.h`（最隐蔽）

手工拷来的 `TensorRT-10.11.0.33` 目录里，`lib/libnvonnxparser.so` 在，但
`include/NvOnnxParser.h` **缺失**（只有 `NvOnnxConfig.h`）。现象是 `trt_engine.cpp`：

```
fatal error: NvOnnxParser.h: No such file or directory
```

lib 有、头没有，很容易误以为是 `-DTRT_ROOT` 指错。修复：补一个**版本匹配**的头
（onnx-tensorrt 的 `release/10.11-GA` 分支，与 `libnvonnxparser.so.10` 同版）：

```bash
curl -fsSL https://raw.githubusercontent.com/onnx/onnx-tensorrt/release/10.11-GA/NvOnnxParser.h \
     -o "$TRT_ROOT/include/NvOnnxParser.h"
```

根治办法是换用 NVIDIA 官方完整 TensorRT tarball（头与库齐全）。切记头的版本要与
`.so` 一致，跨版本的 `IParser` 接口不保证兼容。

## 坑 3：六路拼接源在 Linux 默认被静默跳过

`CMakeLists.txt` 用 `find_package(FFMPEG)` 找 libav*，但 **FFmpeg 没有官方 CMake
模块**——只有 vcpkg 的 ffmpeg port 提供 config（Windows 命中）。Linux 上系统/自编
ffmpeg 命不中，configure 只打印：

```
拼接源: 跳过（未找到 FFmpeg libav*，--input-6cam 不可用）
```

于是 `--cam-dir` 六路现拼这条路悄悄没编进去，画布 `--input` 主链路照常。

已在 `CMakeLists.txt` 加了 **pkg-config 回退**：找不到 FFMPEG config 时用
`pkg_check_modules(FFMPEG libavcodec libavformat libavutil libswscale)`。因此 Linux
只要装了 libav* 的 dev 包（`pkg-config --exists libavcodec` 为真）就会自动编入，
configure 打印 `拼接源: 启用`。运行期还需要 `libnvcuvid.so` 与带 `h264_cuvid`
的 libavcodec（`ffprobe -decoders | grep cuvid` 能看到即可）。

## 坑 4：首次运行的 engine 构建很慢，别把超时设太短

首次跑会从 ONNX 现烘两个 engine：**detect（yolo26x）约 7 min，pose 约 4 min**，
期间无画面输出。把整体超时设成 10 min 会在 pose 构建中途被杀（detect 已占掉大半）。
烘好后按身份戳缓存，之后秒开。engine 命名形如
`detect.sm90-trt101100-b1-fp16-<stamp>.engine`，换 GPU 架构或 TRT 版本会自动重烘。

## 坑 5：六路数据的相机→mesh 对应是定死的

`20260730-4k-raw` 的六个文件名是 `..._cam{1..6}.mp4`，但拼接的 mesh 顺序是
`cam3 cam2 cam1 cam4 cam5 cam6`（`build_stitch_lut.py` 的 `POOL_CAMERA_IDS`）。
用默认参数烘表即正确，不要重排：

```bash
python3 cpp/tools/build_stitch_lut.py --force     # -> cpp/models/stitch.lut，画布 5002x2102
```

（`data/20260629/` 那批是另一组对应 `cam4 cam3 cam2 cam5 cam6 cam1`，别混用。）

## 一键复现（本机已验证的完整序列）

```bash
cd swim_6cam_analyse
TRT=/root/paddlejob/workspace/env_run/penghaotian/sport_project/trt_local/TensorRT-10.11.0.33

# 0) 权重（坑 1）
apt-get install -y git-lfs && git lfs install --local && git lfs pull

# 1) 导出 ONNX（detect 必须 imgsz=640）
python3 cpp/tools/export_onnx.py --out cpp/models
python3 cpp/tools/build_stitch_lut.py --force          # 六路拼接表（坑 5）

# 2) 补 TRT 头（坑 2，仅当 trt_local 缺 NvOnnxParser.h）
[ -f "$TRT/include/NvOnnxParser.h" ] || curl -fsSL \
  https://raw.githubusercontent.com/onnx/onnx-tensorrt/release/10.11-GA/NvOnnxParser.h \
  -o "$TRT/include/NvOnnxParser.h"

# 3) 构建（sm_90；pkg-config 回退让六路拼接编入，坑 3）
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release -DTRT_ROOT="$TRT" -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build cpp/build -j"$(nproc)"

# 4) 跑（首次烘 engine 慢，坑 4）
CAM=/root/paddlejob/workspace/env_run/penghaotian/sport_project/swim_datas/20260730-4k-raw
cpp/build/swim_analyse --cam-dir "$CAM" --models cpp/models --max-frames 3000 --json out.json
```
