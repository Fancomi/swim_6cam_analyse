# C++ 实时推理（Plan C，全程 GPU 驻留）

把 Python 版 Plan C（画布 detect + 画布 RTMPose）重写为 C++/CUDA，目标是台式机上
实时处理 live 画布流。核心约束：**图像进 GPU 后不再回 CPU**，全链路只回读关键点。

本页是 C++ 链路的唯一权威文档（数据流、依赖、构建、显存、性能、与 Python 的对齐）。
项目总览与 Python 链路见 [`../README.md`](../README.md)；改动前的导航与同步契约见
[`../CLAUDE.md`](../CLAUDE.md)；Windows 环境坑见 [`../docs/windows.md`](../docs/windows.md)。

## 数据流

输入有两条路，产出同一个 `GpuFrame`（BGR uint8，显存），下游完全不感知差异：

```
--input   已拼好的画布 mp4/流  ─ffmpeg 管道解码 + H2D─┐
--cam-dir 六路 4K 原片         ─NVDEC + CUDA 拼接────┤（全程显存，零 H2D）
                                                     ↓
preprocess (CUDA)    BGR→RGB + 双线性 + letterbox → detect 输入 fp16
detect     (TRT)     yolo26 end2end，输出 [1,300,6]，NMS 已内置
filter+dedup (CUDA)  conf 筛选 + 坐标还原 + 包含率去重
crop_affine (CUDA)   复刻 GetBBoxCenterScale 语义，batch 拼接 → pose 输入
pose       (TRT)     RTMPose-m，动态 batch 1..40
simcc_decode (CUDA)  双轴 argmax + 二次插值 → 画布坐标
  ↓ 唯一 D2H：框 + 关键点合并成一块，约 9 KB/帧
CPU                  IoU 跟踪 → 划水计数 → 速度 → 回调
```

渲染（`--out`）才额外付一次整帧 D2H（31.5 MB/帧），但它被 `cudaEvent` 完全重叠掉，
计时里只剩 0.02 ms/帧。`--preview` 走同一次 D2H，因此两者同开不会拷两遍。

跨帧重叠：关键点与整帧的 D2H 都只记一个 `cudaEvent` 就返回，由后处理线程在真正读
之前 `cudaEventSynchronize`，所以「GPU 算第 n+1 帧」与「CPU 跟踪第 n 帧」并行。
每帧唯一必须当帧等待的是框数（4 字节，pose 的 batch 取决于它）。

## 上游拼接（`--cam-dir`，六路 4K → 画布，全程显存）

把六路原相机片段直接喂进来，画布在 GPU 上现拼，**没有中间的画布 mp4**：

```
六路 4K H.264 ──NVDEC(h264_cuvid)──▶ AV_PIX_FMT_CUDA / NV12（显存，每路一线程）
                                        │  唯一 D2H 是几百字节的 lane 描述
                     stitch (CUDA) ─────┤  逐画布像素 gather：查表取源坐标 + 权重
                                        ▼  NV12 双线性 → bt601 → 加权累加 → BGR
                                     GpuFrame（画布 5002×2102）
```

省掉了「先拼成 mp4、再解码画布」的一次 H.264 编码与一次 5002 宽画布解码 ——
后者本身就要 13 ms/帧、占原链路 49% 的耗时。

**NVDEC 在这里可用、解画布时不可用**：CUVID 的 H.264 上限 4096×4096，成品画布宽
5002 超限，而单路 4K 只有 3840 宽正好装得下（实测六路并发解码 25.8 ms/帧，
NVDEC 利用率 100%，即已跑满解码器）。

**几何不在 C++ 里算**。`cpp/tools/build_stitch_lut.py` 把 `configs/pool_mesh.json`
烘成逐像素查找表 `cpp/models/stitch.lut`（每覆盖像素一条：源坐标 f32×2 + 权重 u16，
109 MB，1.041 条/像素），kernel 只做 gather。理由：「哪个画布像素属于哪个三角形」的
判定语义来自 OpenCV 的 `fillConvexPoly` + `getAffineTransform`，在 CUDA 里重写是整条
链路唯一容易**静默**出错的地方（差一个边界像素、仿射差半像素，表现为接缝错位而不
报错）。烘表脚本直接调那两个函数，于是 C++ 侧零光栅化代码。改了标定就重跑它
（`scripts\build.bat` 已包含这一步）。

kernel 的三处口径是逐值拟合出来的，改动前先看清代价：

| 口径 | 取值 | 依据 |
| --- | --- | --- |
| YUV→BGR 矩阵 | **bt601 limited → full** | `cv2.VideoCapture` 与 `ffmpeg -vf scale=in_color_matrix=bt601:in_range=tv` **逐字节相同**；按文件标签的 bt709 会整体偏 mean 3.6 灰阶 |
| 定点取整 | `floor(x - 0.5)` | 拟合 swscale 的 16 位定点查表（4K 整帧 mean\|d\| 0.36，floor 0.62，round 1.12） |
| 插值顺序 | 4 个 tap 各自转 BGR 后再双线性 | 反过来（先插 chroma 再转色）在 chroma 边缘偏差翻倍（独占像素 mean\|d\| 1.69 vs 1.08） |

权重烘表时已全局归一化到和为 1（实测偏差 ±1.5e-5 = u16 的 1 个 LSB），所以 kernel
不必再除 alpha —— 那次除法只用来兜量化残差与边界像素。累加顺序固定为 lane 顺序、
无原子操作，因此逐位可复现。

对齐结果：CUDA 画布 vs 离线 CPU 参考（同一份标定，用上游 `compose.py` 的语义独立
算出）**mean|d| 0.34 灰阶、最大 3、66.8% 逐字节相同、100% 在 2 灰阶内、PSNR 52.7 dB**。
残差是无偏舍入噪声（bias +0.13），不是几何错位。

## 依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| CUDA | 12.x | Linux 机 12.9，Windows 机 12.8 |
| **TensorRT** | **10.11.0.33** | cuda-12.0~12.9 变体。TRT 11 移除了 `BuilderFlag::kFP16`（强类型恒开），本代码只针对 10.x。Windows 导入库名带后缀：`nvinfer_10.lib` |
| OpenCV | 4.x | 解码、渲染与 `--preview` 窗口（需 `highgui`）。apt / vcpkg 版都未编 `cudacodec` |
| ffmpeg CLI | 任意近期版本 | `--input` 的默认解码路径要它（`ffmpeg` + `ffprobe`），`--out` 的编码也要（Windows 必然走这条，见「渲染写出」）。找不到则回退 OpenCV |
| FFmpeg libav*（可选） | 含 `h264_cuvid` | 只有 `--cam-dir` 需要：`avcodec`/`avformat`/`avutil`。vcpkg 装 `ffmpeg[core,avcodec,avformat,nvcodec]:x64-windows`（实测 7.1.2）。找不到就跳过这一路，其余功能不变 |
| cmake | ≥3.18 | |

## 构建

**Windows 双击 `scripts\build.bat`** 即可：它做 configure + 编译 + 导出 ONNX，
路径用环境变量覆盖（`SWIM_TRT_ROOT` / `SWIM_VCPKG` / `SWIM_CUDA_ARCH`，`SWIM_FRESH=1` 重来）。
下面是它实际执行的命令，Linux 或想手工控制时用：

```bash
# Linux
cmake -B build -DTRT_ROOT=/opt/trt/TensorRT-10.11.0.33 -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j

# Windows（VS 2022 + CUDA + TRT 官方 zip + vcpkg 的 OpenCV）
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" \
      -DTRT_ROOT=D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33 \
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build cpp/build --config Release
```

`CMAKE_CUDA_ARCHITECTURES`：RTX30xx=86，RTX40xx=89，H800=90，RTX50xx(Blackwell)=120
（`120` 要 nvcc ≥ 12.8，CMakeLists 会在更老的 toolkit 上自动剔掉它）。
`scripts/build.bat` 默认 **`89;120`** —— 交付包要同时覆盖开发机（40 系）与现场
可能是 50 系的机器，而**我们自己的 CUDA kernel 只能靠编进 exe 的 cubin**：
TRT engine 可以在目标机用 ONNX 现烘，kernel 不行，缺了那一档就退化成驱动 JIT
（首次启动多等几十秒）。只在本机跑、想省一半 CUDA 编译时间就 `set SWIM_CUDA_ARCH=89`。
产物带了哪几档用 `cuobjdump --list-elf cpp/build/Release/swim_analyse.exe` 核，
`scripts/dist.ps1` 打包时也会核一遍并写进交付包的 `README.txt`。
OpenCV 由 vcpkg 提供（`vcpkg install opencv4:x64-windows`）时传它的 toolchain file 即可，
`find_package(OpenCV)` 会自动解析；用官方预编译包则改传 `-DOpenCV_DIR=C:/opencv/build`。
多配置生成器（VS）**必须加 `--config Release`**，否则拿到的是慢一个量级的 Debug 二进制。

## 准备模型与拼接表

```bash
# 从 .pt/.pth 导出 fp16 ONNX（detect 必须用训练时的 imgsz=640）
python cpp/tools/export_onnx.py --out cpp/models
# 烘拼接查找表（只有 --cam-dir 需要；改了 configs/pool_mesh.json 要重跑）
python cpp/tools/build_stitch_lut.py
```

Windows 上 `scripts\build.bat` 已包含这两步（用 `.venv\Scripts\python.exe`，
需先跑 `scripts/install.sh`）。
`.onnx` / `.engine` / `.lut` 都不入库；权重来源见 [`../docs/权重来源与复现.md`](../docs/权重来源与复现.md)。

engine 由程序首次运行时自动构建并缓存。文件名带**身份戳**：
`pose.engine` → `pose.sm89-trt101100-b40-fp16-<onnx mtime>-<size>.engine`。
戳里含 GPU 架构、运行期 TRT 版本、`--max-persons`、精度与 ONNX 的 mtime/size，
所以换机器 / 换 `--max-persons` / 换 `--fp32` 各存一份，既不会互相覆盖，也不会
静默复用不匹配的 engine，无需手动删文件。写盘走临时文件（名字带 pid）+ rename，
中途崩溃或两个进程同时首次构建都不会留下半个 engine 冒充缓存；反序列化失败会
自动删除并重建一次。

首次构建要几分钟且期间不出画面，所以它会先打一行「要等多久」，再由
`IProgressMonitor` 原地刷 `[TRT] 构建中 <阶段> NN%  已用 NNN s`（限频 10 Hz ——
一次构建有 240 万次回调，逐次 printf 会把构建本身拖慢），结束时报总耗时。
「试着反序列化交付包自带的裸名 engine」失败是**预期路径**（换代显卡必然不匹配），
TRT 那条 `Error Code 6` 被静音后并进我们自己的 `不可用（…）` 一行，
免得现场把「正在按计划现烘」看成崩溃。

输入形状不进身份戳 —— 它由 ONNX 写死，而 ONNX 的 mtime/size 已在戳里。因此
`--detect-size` 与 ONNX 不符时不会（也不该）触发重建，而是在启动时直接报错：
kernel 按该参数写、engine 按 ONNX 分配，不校验就是越界写。detect 输出的 `max_det`
与 `kMaxDet`、pose 两个输出的 simcc 轴长也一并核对，都在 `Pipeline` 构造里一次挡掉。

## 运行

Windows 双击即可，不必记命令。**脚本选「结果去哪」，第一个参数选「输入从哪来」**：

| | `scripts\preview.bat`（开窗口） | `scripts\analyse.bat`（写文件） |
| --- | --- | --- |
| 省略 / `canvas` | 实时看已拼画布 | 批处理已拼画布 → json |
| `6cam` | 实时看六路现拼 | 批处理六路现拼 → json |
| 视频文件（可拖） | 实时看那段视频 | 批处理那段视频 |
| 目录（可拖） | 实时看那批六路片段 | 批处理那批六路片段 |
| 相机清单文件 | 实时看六路 ZCam 现拼 | 批处理（流没有结尾，需配 `--max-frames`） |
| `rtsp://…` | 实时看单路直播画布 | 报错（流没有结尾） |

两者跑的是同一套算法、同一份 exe，其余参数原样透传
（`scripts\analyse.bat 6cam --out o.mp4 --max-frames 300`）。命令行解析、源解析与
前置检查都在 `scripts\env.bat` 里做一次，两个 launcher 各只剩十几行。
默认源用 `SWIM_CANVAS` / `SWIM_CAM_DIR` 覆盖。

Linux 或要自定义参数时直接调二进制：

```bash
export LD_LIBRARY_PATH=/opt/trt/TensorRT-10.11.0.33/lib   # Windows 是把该目录加进 PATH

# 离线视频，纯分析（最快，只回读关键点）
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --json out.json

# 六路 4K 原片，GPU 上现拼再分析（没有中间画布 mp4）
./build/swim_analyse --cam-dir /path/to/20260730-4k-raw --models cpp/models --json out.json

# 六路现场 ZCam：相机清单，每行 <相机>=rtsp://…（bash scripts/cams.sh list 生成）
./build/swim_analyse --cam-dir configs/cameras.txt --models cpp/models --preview --show-fps

# 实时预览窗口（不落盘，按 q/ESC 提前结束，1/2/3 切关键点/分析/米制网格）
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --preview --show-fps

# 加标注视频输出
./build/swim_analyse --input data/xxx.mp4 --models cpp/models \
    --out out.mp4 --json out.json --show-fps

# live 流
./build/swim_analyse --input rtsp://... --models cpp/models --preview --show-fps

# 与 Python 逐帧对照：导出每个框与 17 个关键点
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --max-frames 200 --dump d.csv

# 拼接对照：把首帧画布原样落 PNG（未画标注），与离线拼接逐像素比
./build/swim_analyse --cam-dir /path/to/clips --models cpp/models \
    --max-frames 1 --dump-canvas f0.png
```

`--input` 与 `--cam-dir` **必须且只能给一个**（前者是已拼画布，后者是六路输入）。
`--cam-dir` 给**目录**时按 `stitch.lut` 里的相机 id 找 `*_<相机>.mp4`，一个相机匹配到
0 个或 2 个以上都直接报错 —— 静默挑一个等于把错误的相机贴到网格上，症状是接缝
错位而不是报错。给**文件**时按相机清单读（每行 `<相机>=<地址>`，`#` 起注释），
地址可以是路径也可以是 `rtsp://`，缺哪台相机会明确报出来。
用 `相机=地址` 而不是按行序对应，是因为现场相机的 IP 尾数与 mesh 顺序不同
（`20260730` 那批实测是 `cam3 cam2 cam1 cam4 cam5 cam6`），按行序写迟早错位。
`--stitch-lut` 可换表（默认 `<models>/stitch.lut`）。

`--help` 看全部参数（帮助里的默认值直接取自 `PipelineOptions{}`，不会与代码走偏）。
算法阈值与 Python CLI 同名同义：`--conf` / `--kpt-thr` / `--containment` /
`--track-iou` / `--track-max-lost` / `--stroke-type` / `--signal` / `--ppm`；
`--queue-depth` 只影响流水线深度，不改结果。
`--json` 的格式是 `{tid: {"strokes": n, "speed": v}}`，`speed` 是该 track **最后一次**
速度采样，面向实时上报；Python 的 `result.json` 存整条速度序列，两者不可直接 diff。

## 可视化（`--preview` / `--out`）

两条路各自独立，可单开也可同用。**两者显示/写出的都是当帧刚算出来的结果** ——
没有任何结果缓存，每一帧都现场跑 detect + pose + 跟踪 + 划水/速度，
`--preview` 只是把这帧的框与骨架画到窗口上（Windows 上双击 `scripts\preview.bat`
即可启动）。

| 开关 | 做什么 | 实测（RTX 4080 Laptop） |
| --- | --- | --- |
| 无 | 纯分析，不拷图像 | 12.8–15.5 ms/帧（65–78 fps） |
| `--preview` | 缩放后开窗实时显示 | 14.3–15.5 ms/帧（65–70 fps） |
| `--out` | 标注视频落盘 | 29.0–37.6 ms/帧（27–35 fps） |

预览几乎免费（约 +1 ms/帧）：整帧 D2H 已被 `cudaEvent` 重叠到 0.01 ms，
真正的开销只是 `cv::resize` + `imshow`。落盘贵是因为 libx264 与解码抢 CPU。

画布 5002×2102 装不进任何屏幕，所以**先缩再画**：默认自适应到 1600×900 以内
（本数据 ×0.320 → 1600×672），`--preview-scale 0.05~1` 可手动指定（给了它就
等于开了 `--preview`）。缩放在画标注之前做，因此框线与文字是原生清晰度而非
被一起缩糊；字号与线宽按比例降但有下限，小窗仍可读。

窗口**可自由拖拽缩放与最大化**，画面按当前客户区等比放到最大并居中（letterbox，
黑边补齐）。所以 `--preview-scale` 只决定**初始**窗口大小，之后每帧按实际客户区
重算缩放比。实测 2560×1494 客户区下画面 2552×1077、上下黑边 208/209 px。
自己做 letterbox 而不用 OpenCV 的 `WINDOW_KEEPRATIO`：后者在 Win32 与 Qt 后端
行为不一致。**不能用 `WINDOW_AUTOSIZE`** —— 它把窗口钉在图像尺寸上，最大化后
画面仍以原尺寸缩在左上角、`resizeWindow` 对它无效（曾经的表现）。
黑边画板只在客户区尺寸变化时重建一次，每帧仍只有一次 `cv::resize`、零额外拷贝。

窗口在**后处理线程**里惰性构造 —— Win32 消息队列按线程分，`waitKey` 就是
pump，跨线程创建会不刷新甚至卡死。按 q/ESC 触发 `Pipeline::request_stop()`：
置停止位 + `close()` 唤醒可能阻塞在 push 的推理线程，已入队的几帧仍会处理完，
所以提前退出不会丢结果也不会死锁。

预览不改变任何分析结果：3000 帧带 `--preview` 与纯分析的 JSON 逐字节相同
（md5 `5b143434…`）。

### 叠加层的三个开关（窗口里按 1/2/3）

叠加层分三层，各自独立开关。**`--input` 与 `--cam-dir` 两条路完全一样** ——
开关状态存在一个 `Overlay` 结构里，按引用传进 `Preview::show()`，`--out` 落盘那条
路读的是同一份，所以「窗口里看到的」就是「写进 mp4 的」，不存在第二套标志。

| 层 | 热键 | 启动开关 | 内容 |
| --- | --- | --- | --- |
| 关键点 | `1` | `--no-kpts` | COCO17 骨架与关节点（按 `--kpt-thr` 过滤） |
| 分析 | `2` | `--no-boxes` | 检测框 + `ID:n S:n v m/s` 标签 |
| 米制网格 | `3` | `--grid` | 按 `--ppm` 每 1 m 一条线，5 m 加粗并标米数 |

米制网格**与标定无关**：它只用 `--ppm`（默认 100 px/m）和画面尺寸换算，用途是
目视量距离、核对速度口径（速度 = 框中心位移 / ppm）。所以画布与六路现拼两条路
画出来的是同一套刻度，5002×2102 的画布左下角会标出 `50.0 x 21.0 m`。
图内文字一律英文；线太密（缩放后 1 m 不足 4 px）时自动不画。

热键只有 `--preview` 那条路有（`--out` 没有键盘），且必须**焦点在窗口上** ——
`cv::waitKey` 是唯一能读到按键的地方，而它只在创建窗口的那个线程里有效。

### 时间轴帧率（`--fps`）

不传 `--fps` 时帧率由输入自己决定（容器/流自报值；六路现拼取各路的最大值），
这是默认行为。传了就以参数为准，两条路都会打一行「覆盖源自报的 X」。

它**只改时间轴与离线路限速，不碰解码** —— 划水间隔与速度都按帧率折算成秒，
所以帧率报错时这两个数会整体偏。实测 ZCam 相机配成 `4KP29.97` 后 rtsp 里
仍可能报 59.94，`bash scripts/cams.sh run` 因此默认按 `CAM_FMT` 下发
`--fps`（`CAM_FPS=0` 关掉）。同一段跑 `--fps 30` 与不传，人次/track 数完全相同。

## 输入源

`FrameSource` 的三种实现共用 `next() -> GpuFrame` 接口，下游不感知差异：

| 实现 | `backend()` | 用途 |
| --- | --- | --- |
| ffmpeg 管道 | `ffmpeg-pipe(prefetch)` | `--input` 默认。`ffprobe` 取元信息 + `ffmpeg … -f rawvideo -pix_fmt bgr24 -` 直读进锁页内存，实测 13.0 ms/帧 |
| OpenCV | `opencv(prefetch)` | 回退与 `--decoder cpu`。文件与 rtsp/rtmp 共用（`VideoCapture`），16.9 ms/帧 |
| 六路 NVDEC + 拼接 | `nvdec-stitch` | `--cam-dir`。六路 4K 硬解进显存 + CUDA 拼接。离线片段 20.3 ms/帧（NVDEC 已跑满）；现场 RTSP 实测 30.3 fps，见「上游拼接」 |

前两者带解码线程预取；第三种是每路一个解码线程 + 画布环。
**接现场 ZCam 相机**就走第三种：`--cam-dir` 接受一个相机清单文件（每行
`<相机>=rtsp://…`），`libavformat` 对文件与流是同一套 API，`NvdecLane` 内部不区分。
相机侧的配置口径与实测约束见 [`../docs/cameras.md`](../docs/cameras.md)。

**直播路是「先探后放，用时再拉」**：构造时逐路打开、打印
`lane i = camX 3840x2160 29.97 fps h264_cuvid 直播流`，随即 `park()` 关掉 demux 与
解码器（只留 hw device context 与已探到的参数），解码线程推迟到第一次 `next()` 才
起、在自己的 `loop()` 里补开连接。因为 `make_source()` 返回后主线程还要烘 TRT
engine（换代显卡首次几分钟），六路 RTSP 挂着没人读就会 TCP 缓冲堆满 → 全部
`-138` 超时 + 重连刷屏（现场日志实测）。补开失败直接进重连那条路，所以某路未上电
不会让整条链路起不来。

直播流与离线片段的处理有四处**刻意不同**（都由实测逼出来，改动前先读那份文档）：

| | 离线片段 | 直播流 |
| --- | --- | --- |
| 队列满时 | 背压等待（丢帧会与基线对不上） | **丢最旧的一帧**（相机不等人，停读 socket 会让它超时） |
| 读失败 | 即 EOF | 连续 3 次才判定断流（单次通常只是 socket 超时） |
| 判定断流后 | 该路结束 | **本路后台重连，其余五路照常出画** |
| 解码速度 | 不限速（批处理越快越好） | 由相机定速；**混合来源时离线路按帧率限速**，否则会饿死直播路 |

**单路掉线不停画布**（只有直播路有这条）：某路连续读失败即在**它自己的解码线程**里
重开 demux + 解码器，hw device context 与 surface 池不动，所以一次成功的重连约 1 秒。
掉线期间 `pop()` 返回「本路无新帧」，`next()` 用该路上一帧的 `av_frame_ref` 顶住
（同一张 surface，不拷像素），其余路正常拼进画布；整帧都没有新画面时按帧率 sleep
一帧，**绝不空转**（否则占满 CPU 且 q/ESC 停不下来）。退出时打印
`掉线重连 N 次，整帧沿用旧画面 M 帧` —— 后者乘帧周期就是累计掉线时长。
实测（一台真机经本机 TCP 转发制造 10 s 断流）：重连 1 次、顶住 228 帧，
帧率 30.3 → 25.8 fps 后自行恢复，画布全程未停。重连后分辨率变了直接报错，
因为 `stitch.lut` 是按单一源尺寸烘的。

ffmpeg 管道的 13.0 ms/帧已经贴住 ffmpeg CLI 自身的地板：同一条解码命令落 `NUL`
实测 10.0–13.5 ms/帧（纯解码不出像素 7.0–10.6 ms/帧），管道搬运几乎没有余量。
区间来自系统页缓存：同一文件冷热两次差 25%，所以这条地板要连着测量前提一起引用。
地板是靠自建 `CreatePipe`（128 MB 缓冲）拿到的 —— `_popen` 的匿名管道只有几 KB，
ffmpeg 每写满就得等读取方，实测退化到 16.7 ms/帧。

**两条路径解出的像素值不同**，不只是快慢之别：vcpkg 的 `opencv4:x64-windows` 没编
ffmpeg 特性，videoio 只剩 MSMF，它的 YUV→BGR 换算与 swscale 不一致（逐像素平均差
约 4.5/255），下游 conf 与 track 数都会变（见下表 24107 vs 24245 人次）。
ffmpeg 管道与 Python 的 `cv2.VideoCapture`（自带 ffmpeg）逐字节相同，
所以**要与 Python 逐值比对必须走默认路径**，`--decoder cpu` 只作兜底与排障。

**解画布时 NVDEC 用不上，不是"待补"**：CUVID 的 H.264 8bit 规格上限 4096×4096，
画布宽 5002 硬解开不了（HEVC 上限 8192，但源是 H.264）。所以 `DecoderPref` 只有
`Auto`/`Cpu` 两个取值；`--decoder nvdec` 仅为兼容旧命令行而接受，会打印被忽略的原因
并降级为 `auto`，其它非法取值直接报错退出。
`--cam-dir` 那条路能用 NVDEC，正是因为它解的是单路 3840 宽而不是成品画布。

## 渲染写出（`--out`）

`Writer` 两级：先试 `cv::VideoWriter`（`avc1` → `mp4v`），都开不了则起 `ffmpeg` 子进程，
用管道喂 rawvideo（`-c:v libx264 -preset veryfast -crf 23`）。子进程与管道走
`Proc`（`include/swim/proc.h`）—— 解码（ffmpeg/ffprobe）与编码共用这一个类，
Windows 上自建 `CreatePipe` 而非 `_popen`，理由见「输入源」的缓冲区一段。

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
| 画布帧 BGR ×4（device 环，`ring`） | 126 MB |
| detect 输入 fp16 (640²) | 2.4 MB |
| detect engine 工作区 | 56.2 MB |
| pose 输入 fp16 ×40 | 11.2 MB |
| pose engine 工作区 | 153.4 MB |
| 关键点等小缓冲 | <2 MB |

host 侧另有锁页缓冲：解码环 4×31.5 MB、关键点回读环 5×9 KB，`--out`/`--preview` 时
再加整帧回读环 5×31.5 MB。回读环深度 = `queue_depth + 2`（队列 + 消费者手上 1 +
生产者正在写 1），有了环，推理线程排完 D2H 就能去排下一帧，不必等拷贝落地。

超过 40 人的框会被丢弃以保持显存恒定（实测该数据每帧最多 13 人）。

`--cam-dir` 那条路多占约 1.8 GB（整进程 2.8 GB，对比画布路的 1.0 GB）：

| 项 | 大小 |
| --- | --- |
| 拼接查找表（一次 `cudaMalloc`，只读） | 109 MB |
| 六路 NVDEC surface 池（4K NV12 ×12 MB × 每路约 14 张） | 约 1.0 GB |
| 画布环 4×31.5 MB（拼接输出，替代解码环） | 126 MB |
| 六路解出但尚未消费的帧（预取 3 + 环里扣 4） | 计入上面的 surface 池 |

surface 数由 `kExtraHwFrames`（`stitch_source.cpp`）控制。它必须 ≥「预取深度 + 画布
环深」：解码帧要一直扣到拼接 kernel 读完才能 `av_frame_free`（free 就把 surface 还给
解码器，提前还等于边解码边覆写正在被读的显存），给不够会在 `receive_frame` 处卡死。
直播路每路再多扣 1 张（掉线时顶住的「上一帧」，同一张 surface 的第二个引用），
默认 16 的余量已经含它。

## 性能

3000 帧 5002×2102 画布，约 8 人/帧，`--max-persons 40`。

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
| `0_src_wait`（等解码） | 6.0–8.0 ms | 18.6–25.2 ms |
| `1_pre+detect` | 5.1–5.6 ms | 7.4–9.0 ms |
| `2_crop+pose+decode` | 1.7–2.0 ms | 2.8–3.5 ms |
| `7_track_metrics` | 0.22–0.35 ms | 0.22–0.47 ms |
| `8_frame_d2h` | — | 0.02 ms |
| **端到端** | **12.8–15.5 ms（65–78 fps）** | **29.0–37.6 ms（27–35 fps）** |

### 六路 4K 现拼（`--cam-dir`，RTX 4080 Laptop，3000 帧）

| 阶段 | 每帧 |
| --- | --- |
| `0_src_wait`（六路 NVDEC + CUDA 拼接） | 20.3 ms（77.6%） |
| `1_pre+detect` | 4.54 ms |
| `2_crop+pose+decode` | 1.34 ms |
| `7_track_metrics` | 0.53 ms |
| **端到端** | **26.2 ms（38.2 fps）** |

比读已拼画布慢一倍（13.3 vs 26.2 ms/帧），瓶颈仍在解码侧但性质不同：
**NVDEC 利用率已经 100%**，六路 4K 并发的裸解码就要 25.8 ms/帧（单独测 ffmpeg
h264_cuvid 六路并发 900 帧 23.2 s），拼接 kernel 只占其中约 1 ms。所以这条路的上限
就是本卡的 NVDEC 吞吐，比它更快只能减路数、降分辨率、或换多解码器的卡。
GPU 段（`1_`+`2_`）反而比画布路更快（5.9 vs 7.5 ms），因为不再和 CPU 解码抢内存带宽。

接实际 zcam 流时这一项会消失：那时帧本来就在显存里，不必解码。

给区间而非单值：这是笔记本，同一条命令连跑三次差 10% 属常态（散热与后台进程），
渲染模式波动更大因为它多起一个 libx264 进程抢 CPU。GPU 段（`1_`+`2_`）稳定在 6.8–8 ms。

一致性（`--input` 画布路）：24107 人次、63 track、376 次划水；**纯分析与渲染两种模式的
JSON 与 `--dump` CSV 逐字节相同**，同一路径重复跑也逐字节可复现。
`--decoder cpu` 是另一组数（24245 人次、63 track），差异来自 MSMF 的
像素换算，不是随机性，见「输入源」。
显存（Windows 实测）：整进程约 1.0 GB，含 CUDA context；engine 工作区 detect 56.2 MB /
pose 153.4 MB（比 H800 的 56/94 MB 大，TRT 按 GPU 选 kernel，属正常差异）。

`--cam-dir` 是**另一组合法数字**（3000 帧 25078 人次、91 track、379 次划水），不是回归：
它的画布与 `merged_3000f.mp4` 差 0.34 灰阶（见「上游拼接」），而 track 数对亚灰阶差异
敏感 —— 91 vs 63 主要是短 track 变多（划水为 0 的 track 63 vs 34），有划水的 track
28 vs 29、划水合计 379 vs 376（+0.8%）。逐帧比对 600 帧：框数相同的帧 62.3%，
配对框中心距离中位 1.57 px、91% 在 5 px 内。**两条路的数字不可直接 diff**，
各自对自己的基线。

与 Python 版 Plan C 的 55 ms（detect+pose）相比，GPU 段快 **8.8×**（H800）；pose 部分从
34 ms 降到 1.7 ms，因为裁切的仿射采样与 SimCC 解码都进了 kernel。

**`--input` 那条路的瓶颈在 CPU 解码**：纯分析模式下 `0_src_wait` 占 49%，而它的下限
就是 ffmpeg 自身的 10.0–13.5 ms/帧（见「输入源」），预取只能把它与 GPU 段重叠、消不掉。
NVDEC 对成品画布帮不上（4096 上限），但**上游那条 `--cam-dir` 就是这个问题的正解**：
它绕开画布编解码，直接从六路原片拼，代价是受本卡 NVDEC 吞吐限制（38 fps）。
接 zcam 流后两处解码都没有了。
另一个曾考虑的选项 —— 让 ffmpeg 输出 yuv420p 把搬运字节数减半（31.5 → 15.8 MB/帧）
并在 kernel 里转 BGR —— 会丢掉与 Python 的逐字节一致性（swscale 用 BT.601 limited
range，自行重建后平均差约 0.6/255），故未采用。

渲染模式下 `0_src_wait` 涨到 18.6–25.2 ms 不是解码变慢，而是 libx264 编码进程与解码
进程抢 CPU；`8_frame_d2h` 只有 0.02 ms，说明整帧回读已被 `cudaEvent` 完全重叠掉。
参照系：单用 ffmpeg 把同一段做一次「解码 + libx264 重编码」（不做任何分析）就要
32.8 ms/帧，我们连分析带渲染是 29–37.6 ms/帧 —— 也就是全部 GPU 推理、跟踪、CPU 绘制
基本被 ffmpeg 的编解码掩盖掉了。想快只能换 NVENC。

## 与 Python 版的对齐

| 项 | 状态 |
| --- | --- |
| letterbox（scale/pad） | 与 ultralytics 逐值一致（半像素约定，同 `cv2.resize`） |
| 解码像素 | 默认 ffmpeg 管道与 Python `cv2.VideoCapture` **逐字节相同**；`--decoder cpu`(MSMF) 不同，见「输入源」 |
| 检出数 | 8.0/帧（3000 帧 24107 人次）。Python 侧记录过 10.85/帧，但那份数字的统计口径已无法复现（本机无 torch，未重测），暂不作为对齐依据 |
| track 数（1000 帧） | 34 vs Python 36 |
| 包含率去重 | 复刻 `filter_contained_boxes` 的顺序语义（含 `break`），保序 |
| 裁切/解码坐标 | 两者都用**索引约定**（输出下标直接代入逆映射，无半像素偏移），与 `cv2.warpAffine` 及 mmpose 的 `keypoints/input_size*scale + center - 0.5*scale` 严格互逆。注意这与 letterbox 的半像素约定不同，是两套 OpenCV 语义各自对应的正确写法 |
| 划水信号 | `elbow_angle` / `wrist_x_head` 两路，同 `metrics.py` 的 `SIGNALS`（`--signal`） |
| 平滑 | 缺帧线性插值 → 中值(5) → Savitzky-Golay(11,3)，常数表实现。与 scipy 的 `median_filter`+`savgol_filter` 在支撑完整区间逐点差 < 1e-14 rad |
| 波谷判据 | 局部均值 + **拓扑** prominence（同 scipy 定义），窗宽 `max(distance*2, fps*6)` |
| 过零滞回 | 门限 = 累计全程幅度 × 0.15，同 Python 的全序列 `np.ptp`（不用局部窗，否则开局幅度偏小会多计） |
| 左右分侧 | 自由泳/仰泳分开计数取 min，其余取均值（`--stroke-type`） |
| 速度 | 每 0.2 s 一个采样点、窗口 2 s、帧间前向填充。与 Python 逐 track 相对差 < 5.4e-4（纯 float32 vs float64 舍入，复算可逐位吻合） |

**关键点数值本身不可能逐字节对齐**（上表只保证下游算法逻辑一致）：Python Plan C 的
`configs/rtmpose-m_canvas-192x256.py` 开了 `flip_test=True`，做左右翻转取平均，而
`export_onnx.py` 导出的是裸 backbone+head，C++ 没有翻转平均；另外 C++ 的 SimCC 解码多做
一次二次插值，mmpose 侧 `use_dark=False`。所以两边关键点存在系统性小差，做 `--dump`
对照时应比对下游统计量（划水次数、速度）而非关键点原值。

**实测对齐幅度**（同一份 `--dump` 关键点喂两边，3000 帧 / 63 track）：
`elbow_angle` C++ 376 次 vs Python 379 次（−0.8%），逐 track 平均绝对差 0.52、最大 4；
`wrist_x_head` C++ 234 vs Python 232（+0.9%），平均绝对差 0.13、最大 1。
差异集中在长 track（≥300 帧），全部来自下面这条在线约束。

刻意保留的差异只有一条，源自在线约束（live 不能等全序列结束）：Python 用
`scipy.find_peaks` 扫全序列并按峰高优先解 `distance` 冲突；C++ 是滑窗内逐帧因果判定，
最小间隔以"上一次已计数的事件"为基准，后来的更高峰不会挤掉先前已计入的峰。
局部统计与 prominence 的搜索范围同样限制在最近 6 秒窗口内（Python 是以候选点为中心的
±3 s 截断窗口）：位置偏过去，且窗内可用的平滑样本比 Python 少 2×7 个。
判定点滞后窗口末端 7 帧（中值 2 + SG 5 的邻域），换取该点平滑值无边界偏差；
序列首尾各有一段判不到的盲区，这是 elbow 差异里最大的单项（离线消融量化为 −12 次）。
`wrist_x_head` 另有一处无法消除的差异：方向符号在线只能用"首个可信鼻子 x → 当前"判定，
Python 用全序列首末，开局几帧方向可能翻转（实测造成 2 个 track 各多计 1 次）。
