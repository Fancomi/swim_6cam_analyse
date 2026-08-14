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
  ↓ 唯一 D2H：框 + 关键点合并成一块，约 9 KB/帧
CPU                  IoU 跟踪 → 划水计数 → 速度 → 回调
```

渲染（`--out`）才额外付一次整帧 D2H（31.5 MB/帧），但它被 `cudaEvent` 完全重叠掉，
计时里只剩 0.02 ms/帧。`--preview` 走同一次 D2H，因此两者同开不会拷两遍。

跨帧重叠：关键点与整帧的 D2H 都只记一个 `cudaEvent` 就返回，由后处理线程在真正读
之前 `cudaEventSynchronize`，所以「GPU 算第 n+1 帧」与「CPU 跟踪第 n 帧」并行。
每帧唯一必须当帧等待的是框数（4 字节，pose 的 batch 取决于它）。

## 依赖

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| CUDA | 12.x | Linux 机 12.9，Windows 机 12.8 |
| **TensorRT** | **10.11.0.33** | cuda-12.0~12.9 变体。TRT 11 移除了 `BuilderFlag::kFP16`（强类型恒开），本代码只针对 10.x。Windows 导入库名带后缀：`nvinfer_10.lib` |
| OpenCV | 4.x | 解码、渲染与 `--preview` 窗口（需 `highgui`）。apt / vcpkg 版都未编 `cudacodec`；不过本画布（5002 宽）超出 CUVID H.264 的 4096 上限，硬解本就不可用，见「输入源」 |
| ffmpeg CLI | 任意近期版本 | **默认解码路径就要它**（`ffmpeg` + `ffprobe`），`--out` 的编码也要（Windows 必然走这条，见「渲染写出」）。找不到则回退 OpenCV |
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

engine 由程序首次运行时自动构建并缓存。文件名带**身份戳**：
`pose.engine` → `pose.sm89-trt101100-b40-fp16-<onnx mtime>-<size>.engine`。
戳里含 GPU 架构、运行期 TRT 版本、`--max-persons`、精度与 ONNX 的 mtime/size，
所以换机器 / 换 `--max-persons` / 换 `--fp32` 各存一份，既不会互相覆盖，也不会
静默复用不匹配的 engine，无需手动删文件。写盘走临时文件（名字带 pid）+ rename，
中途崩溃或两个进程同时首次构建都不会留下半个 engine 冒充缓存；反序列化失败会
自动删除并重建一次。

输入形状不进身份戳 —— 它由 ONNX 写死，而 ONNX 的 mtime/size 已在戳里。因此
`--detect-size` 与 ONNX 不符时不会（也不该）触发重建，而是在启动时直接报错：
kernel 按该参数写、engine 按 ONNX 分配，不校验就是越界写。detect 输出的 `max_det`
与 `kMaxDet`、pose 两个输出的 simcc 轴长也一并核对，都在 `Pipeline` 构造里一次挡掉。

## 运行

```bash
export LD_LIBRARY_PATH=/opt/trt/TensorRT-10.11.0.33/lib

# 离线视频，纯分析（最快，只回读关键点）
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --json out.json

# 实时预览窗口（不落盘，按 q/ESC 提前结束）
./build/swim_analyse --input data/xxx.mp4 --models cpp/models --preview --show-fps

# 加标注视频输出
./build/swim_analyse --input data/xxx.mp4 --models cpp/models \
    --out out.mp4 --json out.json --show-fps

# live 流
./build/swim_analyse --input rtsp://... --models cpp/models --preview --show-fps
```

`--help` 看全部参数。

## 可视化（`--preview` / `--out`）

两条路各自独立，可单开也可同用。**两者显示/写出的都是当帧刚算出来的结果** ——
没有任何结果缓存，每一帧都现场跑 detect + pose + 跟踪 + 划水/速度，
`--preview` 只是把这帧的框与骨架画到窗口上（Windows 上双击 `run_preview.bat`
即可启动，见仓库根目录）。

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

窗口在**后处理线程**里惰性构造 —— Win32 消息队列按线程分，`waitKey` 就是
pump，跨线程创建会不刷新甚至卡死。按 q/ESC 触发 `Pipeline::request_stop()`：
置停止位 + `close()` 唤醒可能阻塞在 push 的推理线程，已入队的几帧仍会处理完，
所以提前退出不会丢结果也不会死锁。

预览不改变任何分析结果：3000 帧带 `--preview` 与纯分析的 JSON 逐字节相同
（md5 `5b143434…`）。

## 输入源

`FrameSource` 三种实现共用 `next() -> GpuFrame` 接口，下游不感知差异：

| 实现 | `backend()` | 用途 |
| --- | --- | --- |
| ffmpeg 管道 | `ffmpeg-pipe(prefetch)` | 默认。`ffprobe` 取元信息 + `ffmpeg … -f rawvideo -pix_fmt bgr24 -` 直读进锁页内存，实测 13.0 ms/帧 |
| OpenCV | `opencv(prefetch)` | 回退与 `--decoder cpu`。文件与 rtsp/rtmp 共用（`VideoCapture`），16.9 ms/帧 |
| `RawSource` | `raw` | 拼接程序把已在显存的画布 `push()` 进来，零解码、一次 D2D 拷贝 |

前两者都带解码线程预取，解码与推理重叠。

ffmpeg 管道的 13.0 ms/帧已经贴住 ffmpeg CLI 自身的地板：同一条命令落 `NUL`
实测 12.9 ms/帧（纯解码不出像素 10.6 ms/帧），管道搬运只剩 0.1 ms 的余量。
这条地板是靠自建 `CreatePipe`（128 MB 缓冲）拿到的 —— `_popen` 的匿名管道只有几 KB，
ffmpeg 每写满就得等读取方，实测退化到 16.7 ms/帧。

**两条路径解出的像素值不同**，不只是快慢之别：vcpkg 的 `opencv4:x64-windows` 没编
ffmpeg 特性，videoio 只剩 MSMF，它的 YUV→BGR 换算与 swscale 不一致（逐像素平均差
约 4.5/255），下游 conf 与 track 数都会变（见下表 24107 vs 24245 人次）。
ffmpeg 管道与 Python 的 `cv2.VideoCapture`（自带 ffmpeg）逐字节相同，
所以**要与 Python 逐值比对必须走默认路径**，`--decoder cpu` 只作兜底与排障。

**NVDEC 用不上，不是"待补"**：CUVID 的 H.264 8bit 规格上限 4096×4096，画布宽 5002
硬解开不了（HEVC 上限 8192，但源是 H.264）。`--decoder nvdec` 保留只为兼容旧命令行，
行为等价 `auto`，并会打印被忽略的原因。live 场景要彻底绕开解码就用 `RawSource`。

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

给区间而非单值：这是笔记本，同一条命令连跑三次差 10% 属常态（散热与后台进程），
渲染模式波动更大因为它多起一个 libx264 进程抢 CPU。GPU 段（`1_`+`2_`）稳定在 6.8–8 ms。

一致性：24107 人次、63 track、376 次划水；**纯分析与渲染两种模式的 JSON 与 `--dump` CSV
逐字节相同**，同一路径重复跑也逐字节可复现。
`--decoder cpu` 是另一组数（24245 人次、63 track），差异来自 MSMF 的
像素换算，不是随机性，见「输入源」。
显存（Windows 实测）：整进程约 1.0 GB，含 CUDA context；engine 工作区 detect 56.2 MB /
pose 153.4 MB（比 H800 的 56/94 MB 大，TRT 按 GPU 选 kernel，属正常差异）。

与 Python 版 Plan C 的 55 ms（detect+pose）相比，GPU 段快 **8.8×**（H800）；pose 部分从
34 ms 降到 1.7 ms，因为裁切的仿射采样与 SimCC 解码都进了 kernel。

**瓶颈已经从 GPU 转到 CPU 解码**：纯分析模式下 `0_src_wait` 占 49%，而它的下限就是
ffmpeg 自身的 12.9 ms/帧（见「输入源」），预取只能把它与 GPU 段重叠、消不掉。
NVDEC 在这里帮不上：CUVID 的 H.264 8bit 上限 4096×4096，画布宽 5002 硬解开不了
（HEVC 上限 8192，但源是 H.264）。要进一步提速只有两条路：live 场景走 `RawSource`
让拼接程序直接给显存帧（零解码），或让 ffmpeg 输出 yuv420p 把搬运字节数减半
（31.5 → 15.8 MB/帧）并在 kernel 里转 BGR —— 后者会丢掉与 Python 的逐字节一致性
（swscale 用 BT.601 limited range，自行重建后平均差约 0.6/255），故未采用。

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
