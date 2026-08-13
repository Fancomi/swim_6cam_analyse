# C++ 实时推理（Plan C，全程 GPU 驻留）

把 Python 版 Plan C（画布 detect + 画布 RTMPose）重写为 C++/CUDA，目标是台式机上
实时处理 live 画布流。核心约束：**图像进 GPU 后不再回 CPU**，全链路只回读关键点。

## 数据流

```
FrameSource          解码 → GpuFrame（BGR uint8，显存）
  ↓ 全程显存
preprocess (CUDA)    BGR→RGB + 双线性 + letterbox → detect 输入 fp16
detect     (TRT)     yolo26 end2end，输出 [1,300,6]，NMS 已内置
filter+dedup (CUDA)  conf 筛选 + 坐标还原 + 包含率去重
crop_affine (CUDA)   复刻 GetBBoxCenterScale 语义，batch 拼接 → pose 输入
pose       (TRT)     RTMPose-m，动态 batch 1..40
simcc_decode (CUDA)  双轴 argmax + 二次插值 → 画布坐标
  ↓ 唯一 D2H：框 + 关键点，约 5 KB/帧
CPU                  IoU 跟踪 → 划水计数 → 速度 → 回调
```

渲染（`--out`）才额外付一次整帧 D2H（30 MB/帧，实测 0.67 ms）。

## 依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| CUDA | 12.x | Linux 机 12.9，Windows 机 12.8 |
| **TensorRT** | **10.11.0.33** | cuda-12.0~12.9 变体。TRT 11 移除了 `BuilderFlag::kFP16`（强类型恒开），本代码只针对 10.x。Windows 导入库名带后缀：`nvinfer_10.lib` |
| OpenCV | 4.x | 解码与渲染；apt / vcpkg 版都未编 `cudacodec`，故 NVDEC 暂走 CPU 回退 |
| ffmpeg CLI | 任意近期版本 | 仅 `--out` 需要，且仅当 OpenCV 开不了写出器时（Windows 必然走这条，见「渲染写出」） |
| cmake | ≥3.18 | |

## 构建

```bash
# Linux
cmake -B build -DTRT_ROOT=/opt/trt/TensorRT-10.11.0.33 -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j

# Windows（VS 2022 + CUDA + TRT 官方 zip + vcpkg 的 OpenCV）
cmake -B build -G "Visual Studio 17 2022" -DTRT_ROOT=C:/TensorRT-10.11.0.33 ^
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake ^
      -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

`CMAKE_CUDA_ARCHITECTURES`：H800=90，RTX40xx=89，RTX30xx=86，RTX50xx=120。

## 准备模型

```bash
# 从 .pt/.pth 导出 fp16 ONNX（detect 必须用训练时的 imgsz=640）
python cpp/tools/export_onnx.py --out cpp/models
```

engine 由程序首次运行时自动构建并缓存到 `cpp/models/*.engine`。
**engine 与 GPU 型号 + TRT 版本绑定，不可跨机复制**——换机器删掉 `.engine` 重建即可
（ONNX 更新时也会自动重建，靠 mtime 比较）。

## 运行

```bash
export LD_LIBRARY_PATH=/opt/trt/TensorRT-10.11.0.33/lib

# 离线视频，纯分析（最快，只回读关键点）
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --json out.json

# 加标注视频输出
./build/swim_analyse --input data/xxx.mp4 --models cpp/models \
    --out out.mp4 --json out.json --show-fps

# live 流
./build/swim_analyse --input rtsp://... --models cpp/models --show-fps
```

`--help` 看全部参数。

## 输入源

`FrameSource` 三种实现共用 `next() -> GpuFrame` 接口，下游不感知差异：

| 实现 | 用途 |
| --- | --- |
| `CpuSource` | 文件与 rtsp/rtmp 共用（OpenCV `VideoCapture`）。带解码线程预取，让解码与推理重叠 |
| `NvdecSource` | 待补。运行时 `dlopen("libnvcuvid.so")` 探测，缺失自动回退 CPU |
| `RawSource` | 拼接程序把已在显存的画布直接 `push()` 进来，零解码零拷贝 |

NVDEC 位置已留好：装了带 CUDA 的 OpenCV 后在 `FrameSource::open()` 里接
`cv::cudacodec::createVideoReader` 即可。若 live 场景走 `RawSource`，则完全绕过解码。
（探测已跨平台：Linux `dlopen("libnvcuvid.so")`，Windows `LoadLibraryA("nvcuvid.dll")`。）

## 渲染写出（`--out`）

`Writer` 两级：先试 `cv::VideoWriter`（`avc1` → `mp4v`），都开不了则起 `ffmpeg` 子进程，
用管道喂 rawvideo（`-c:v libx264 -preset veryfast -crf 23`）。

回退是 Windows 必需的：vcpkg 的 opencv4 没有 ffmpeg 特性，videoio 只剩 MSMF，
而 MSMF 的 H.264 编码器在 1080p/4K 能开，**在 5002×2102 直接 `isOpened()==false`**。
NVENC 本可以更快，但实测机（驱动 571.96）只提供 NVENC API 13.0，ffmpeg 8.1 要求 13.1，
所以编码器定为 libx264；驱动升级后把命令里的编码器换成 `h264_nvenc` 即可。
走独立进程的附带好处：编码与本进程的 GPU 推理天然并行。

## 显存

按 `--max-persons`（默认 40）一次性预留，运行期零 `cudaMalloc`，因此人数波动不影响
延迟稳定性。实测占用：

| 项 | 大小 |
| --- | --- |
| 画布帧 BGR ×4（环形） | 120 MB |
| detect 输入 fp16 (640²) | 2.4 MB |
| detect engine 工作区 | 56.3 MB |
| pose 输入 fp16 ×40 | 11.2 MB |
| pose engine 工作区 | 94.4 MB |
| 关键点等小缓冲 | <2 MB |

超过 40 人的框会被丢弃以保持显存恒定（实测该数据每帧最多 13 人）。

## 性能

3000 帧 5002×2102 画布，约 10 人/帧。

### H800（Linux，CUDA 12.9）

| 阶段 | 每帧 |
| --- | --- |
| `1_pre+detect` | 4.49 ms |
| `2_crop+pose+decode` | 1.66 ms |
| `7_track_metrics` | 0.08 ms |
| **GPU 段合计** | **6.2 ms** |
| 纯分析端到端 | 12.4 ms（**80 fps**） |
| 加渲染端到端 | 41.6 ms（24 fps，瓶颈在 CPU H.264 编码） |

### RTX 4080 Laptop（Windows 11，CUDA 12.8，驱动 571.96）

| 阶段 | 纯分析 | 渲染 |
| --- | --- | --- |
| `1_pre+detect` | 8.27 ms | 9.55 ms |
| `2_crop+pose+decode` | 3.58 ms | 4.32 ms |
| `8_frame_d2h` | — | 3.24 ms |
| `7_track_metrics` | 0.11 ms | 0.15 ms |
| **端到端** | **19.8 ms（50.5 fps）** | **32.8 ms（30.5 fps）** |

两机结果一致性：24249 人次、64 track、592 次划水；纯分析与渲染两种模式的 JSON 逐字节相同。
显存（Windows 实测）：整进程约 1.0 GB，含 CUDA context；engine 工作区 detect 61.1 MB /
pose 153.4 MB（比 H800 的 56/94 MB 大，TRT 按 GPU 选 kernel，属正常差异）。

与 Python 版 Plan C 的 55 ms（detect+pose）相比，GPU 段快 **8.8×**（H800）；pose 部分从
34 ms 降到 1.7 ms，因为裁切的仿射采样与 SimCC 解码都进了 kernel。

纯分析模式下剩余开销是 CPU 解码残留（5002×2102 约 19 ms/帧，已被预取部分掩盖）。
**当前瓶颈在 CPU 解码，不在 GPU 推理**——NVDEC 或 `RawSource` 可去掉这部分。

## 与 Python 版的对齐

| 项 | 状态 |
| --- | --- |
| letterbox（scale/pad） | 与 ultralytics 逐值一致 |
| 检出数 | 10.7/帧 vs Python 10.85/帧 |
| track 数（1000 帧） | 35 vs Python 36 |
| 划水计数 | 算法同源，但改为滑动窗口增量式（live 不能等全序列），计数不逐帧相同 |

划水的差异是设计选择：Python 用全序列 `find_peaks`，C++ 用 8 秒滑动窗口逐帧判定
"倒数第 3 点是否局部极小" + 局部自适应阈值。
