# swim_6cam_analyse

六路相机拼接全景视频的游泳动作分析：**逐人划水次数 + 瞬时速度 + 标注视频**。

输入六机位的泳池俯视画面，输出每个泳者的划水次数、逐帧速度曲线，以及叠加了检测框 /
ID / 划水数 / 速度 / 关键点骨架的视频。输入可以是**已拼好的全景视频**（5002×2102），
也可以是**六路 4K 原片** —— 后者的画布在 GPU 上现拼，不落中间文件。

仓库里有**两条独立实现**同一套算法：

| | 用途 | 入口 | 速度 |
| --- | --- | --- | --- |
| **C++/CUDA** | 实时、上线、看效果 | 双击 `scripts/preview.bat` | 画布 65–78 fps / 六路现拼 38 fps |
| **Python** | 换模型、Plan A/B/C 对比、评测 | `bash scripts/run.sh C` | 约 13 倍实时 |

C++ 只实现 Plan C，是 Python Plan C 的重写（同一份权重），两边靠数值对照保持一致；
上游的六路拼接只有 C++ 有。
**Agent / 新人先读 [`CLAUDE.md`](CLAUDE.md)** —— 它说明该走哪条线、只读哪些文件、
以及改动时必须同步的几处契约。

## 快速开始

**Windows（推荐，C++ 实时链路）** —— 双击 `scripts\` 里的 `.bat` 即可，无需命令行。
**脚本选「结果去哪」，参数选「输入从哪来」**，两个维度独立：

```
scripts\build.bat            构建 + 导出 ONNX + 烘拼接查找表（首次一次）

scripts\preview.bat          开窗口实时看 —— 已拼全景视频（默认）
scripts\preview.bat 6cam     开窗口实时看 —— 六路 4K 原片，GPU 上现拼
scripts\analyse.bat          跑完写 json  —— 已拼全景视频（默认）
scripts\analyse.bat 6cam     跑完写 json  —— 六路 4K 原片，GPU 上现拼
```

两个 `.bat` 跑的是**同一套算法**（逐帧 detect + pose + 跟踪 + 划水/速度，无缓存），
区别只有结果去哪：`preview` 显示，`analyse` 落盘。第一个参数还可以是**视频文件**
（= 指定一段全景视频）、**目录**（= 指定六路片段目录）、`rtsp://…`（直播流，只有
preview 能用），都支持直接拖到 `.bat` 上。其余参数原样透传，例如
`scripts\analyse.bat 6cam --out o.mp4 --max-frames 300`。

首次运行会构建 TensorRT engine（几分钟），之后秒开。
环境要求与踩过的坑见 [`docs/windows.md`](docs/windows.md)。

**Python 参考链路（Linux 或 Git Bash）**：

```bash
bash scripts/install.sh                # 建 .venv、装依赖、跑自检
bash scripts/run.sh C                  # Plan C（推荐），默认数据
bash scripts/run.sh C --max-frames 300 # 先跑 300 帧确认链路
bash scripts/run.sh A                  # Plan A（交接原版，需六路原相机视频）
bash scripts/test.sh                   # 秒级自检；--full 加 30 帧 GPU 冒烟
```

Python 结果落在 `output/<视频名>_plan<X>/`：

| 文件 | 内容 |
| --- | --- |
| `<视频名>_plan<X>.mp4` | 标注视频（H.264） |
| `result.json` | 每个 ID 的划水次数与速度采样序列 `{tid: {strokes, speed_mps: [[t,v],…]}}` |
| `signal_plots/id*.png` | 每人的划水信号图（原始 + 平滑 + 事件标记） |
| `cache.pkl` | Stage1/2 中间结果，同 plan 重跑时自动复用 |

C++ 的 `--json` 是另一种更薄的格式（`{tid: {strokes, speed}}`，speed 为最后一次采样值），
面向实时上报；两者不可直接 diff。

## 三套关键点方案

三者只在"如何从画布得到每人的框与关键点"上不同，下游的划水计数、速度、渲染完全共用，
因此指标可直接对比。用 `--plan` 选择：

| Plan | 关键点在哪算 | 需六路视频 | 检测 AP50-95 | 检测召回 | 关键点 PCK@5% | 延迟(12人) |
| --- | --- | --- | --- | --- | --- | --- |
| **A** 交接原版 | 原相机（mesh 映射） | 是 | 0.518 | 0.972 | — | 228 ms |
| **B** 一体 | 画布（yolo26-pose） | 否 | 0.477 | 0.917 | 67.3% | 54.4 ms |
| **C** 两阶段 | 画布（RTMPose 按框） | 否 | **0.518** | **0.972** | **88.3%** | 55.0 ms |

- **A** 关键点分辨率最高（画布框 224×78 映射回相机后 365×85，面积 ×1.7），但每帧要解
  4.65 路 4K 帧、做 416 万次几何点查询，且 RTMPose 的 batch 被切碎（19.3 人分 4.65 组，
  18.2 ms 固定开销重复付出）。
- **B** 一次前向出框+点，整图并行，耗时几乎与人数无关（17 人 35 ms、68 人 17 ms）；
  代价是一个骨干、一个尺度同时服务两个任务，检测与关键点精度均最低。
- **C** 检测与关键点解耦，各自用专门模型；相比 A 省掉 4K 解码与 mesh，batch 不再切碎。
  **推荐用这套**，也是 C++ 实时链路唯一实现的方案。

> 上表口径：`data/20260629`（19.3 人/帧）、H800、人工标注 val（36 帧/639 框）。
> 关键点用 PCK（误差按人体框高归一化）而非 COCO OKS —— 本数据目标 area 中位 17880，
> OKS 容差达 17–29 px 而人体框高仅 78 px，模型差异会被度量淹没。
> **引用任何性能数字都要带上数据集 + 机器 + Plan**，两个数据集的人数密度差一倍以上。

## 工作原理（以 Plan A 为例）

> 完整的链路说明、公式推导与实测耗时拆解见 [`docs/pipeline.html`](docs/pipeline.html)（浏览器直接打开）。

```
拼接全景视频 ──Stage1──▶ 每帧每人的画布框 (YOLO 检测 + IoU 跟踪)
                            │
              Stage2  mesh 几何反投影：画布框 ─▶ 投影面积最大（最清晰）那路相机的原图框
                            │  按 (帧, 相机) 分组，批量 RTMPose top-down 回归 17 关键点
                            │  整体置信度不达标 ─▶ 换次清晰相机补检一次
                            ▼
              Stage3  关键点插值 ─▶ 划水信号 ─▶ 事件计数 ；框中心轨迹 ─▶ 瞬时速度
                            ▼
              Stage4  重读拼接视频，叠加框 / 标签 / 骨架（关键点反投影回画布坐标）
```

Plan B/C 把 Stage1+2 换成画布单遍或画布两阶段，Stage3/4 不变。

**为什么 Plan A 要绕回原始相机**：拼接画布是六路 4K 拉伸重采样后的产物，单个泳者在画布上
只有一两百像素宽。画布框经 mesh 几何映射回原始相机后能拿到几倍的有效分辨率，因此检测跟踪
在画布上做（全局、ID 连续），关键点在原图上做（清晰）。Plan C 用"按框裁切到 192×256"
达到同样目的，无需 mesh。

**划水信号**（`--signal`）：

- `elbow_angle`（默认）肩-肘-腕夹角的**波谷**计数。波谷是"高肘抓水"姿态，
  三点构型立体、角度对关键点噪声不敏感；波峰是手臂伸直、三点接近共线，此时夹角对
  坐标误差极敏感，所以检谷不检峰。阈值按 6 秒滑动窗口的局部统计量自适应，避免动作
  幅度变浅的时段被全局阈值整段淹没。
- `wrist_x_head` 手腕沿泳道方向相对鼻子的带符号位移，滞回过零计数。前进方向由该
  目标鼻子的首末净位移自动判定，与游动朝向无关。

自由泳 / 仰泳左右手交替，两侧分别计数后取 `min`（更抗单侧漏检）；其余泳姿双臂同步，
左右信号取均值后统一计数。

**速度**取检测框中心在画布坐标下的位移：画布是等比米制（`--ppm` 像素/米），像素距离
除以 `ppm` 即得米，不需要反投影。相比用单个关键点（如鼻子），框中心来自跟踪器、
帧间更稳，不会因个别关键点抖动产生速度突刺。

## 目录结构

```
CLAUDE.md                        导航：该走哪条线、同步契约、验证方法（先读这个）
scripts/                         全部用户入口，脚本自己 cd 到仓库根，从哪调都一样
  build.bat                      构建 + 导出 ONNX + 烘拼接表（双击）
  preview.bat                    开窗口实时看（双击；第一个参数选输入源）
  analyse.bat                    跑完写 json / mp4（双击；同一套参数）
  env.bat                        两个 bat 共用：解析命令行 + 定位输入源 + 前置检查，不单独跑
  install.sh run.sh test.sh      Python 入口（Linux / Git Bash）
configs/
  pool_mesh.json                 泳池 mesh 标定（六路相机三角面片 + UV），Plan A 与 GPU 拼接共用
  rtmpose-m_swim-256x192.py      Plan A 的原相机 RTMPose 推理配置
  rtmpose-m_canvas-192x256.py    Plan C 的画布 RTMPose 推理配置
weights/                         成品权重，来源与复现见 docs/权重来源与复现.md
src/swim_analyse/                Python 参考实现
  cli.py         统一入口：--plan A/B/C，Stage3/4 共用
  plans.py       三套方案的实现，统一接口 run() -> (all_boxes, raw_seq)
  metrics.py     划水信号与计数、瞬时速度、信号图
  geometry.py    canvas <-> source 双向映射（网格索引加速），Plan A 用
  tracking.py    检测框去重 + IoU 贪心跟踪
  pose.py        RTMPose 封装、COCO17 定义、关键点插值
  draw.py        骨架 / 标签绘制
  video.py       多路视频的帧级随机读取（带帧缓存），Plan A 用
cpp/                             C++/CUDA 实时实现（Plan C），见 cpp/README.md
  src/ include/swim/             实现与头文件（含六路 NVDEC 拼接 stitch.cu）
  tools/export_onnx.py           weights/ -> ONNX（scripts/build.bat 会调）
  tools/build_stitch_lut.py      pool_mesh.json -> 逐像素拼接查找表（同上）
  models/                        ONNX / TRT engine / stitch.lut（除 pose_meta.json 外不入库）
tests/test_core.py               纯逻辑单元测试（无需 GPU 和数据）
docs/
  windows.md                     Windows 环境要求、脚本编码规则、踩过的坑
  pipeline.html                  Plan A 的公式推导与逐段耗时拆解
  权重来源与复现.md              权重来源与训练复现
```

三套方案共享同一份数据约定，因此新增方案只需在 `plans.py` 里实现一个子类，
覆盖 `_setup()`（加载模型）与 `_frame(fi, frame)`（处理一帧，结果写进
`self.all_boxes` / `self.raw_seq`）两个钩子 —— 开视频、逐帧驱动、收尾都在基类：

```python
all_boxes  {frame_idx: [(track_id, x1, y1, x2, y2, conf)]}    画布坐标
raw_seq    {track_id: [(frame_idx, kpts(17,2), scores(17,), cam_idx)]}
           cam_idx = -1 表示关键点已在画布坐标系
```

## 数据准备

默认数据集 `data/20260730/`（C++ 与 Python Plan B/C 共用，两条线的数字才可比）：

```
merged_3000f.mp4      六路拼接后的全景视频（5002x2102，3000 帧）
```

**六路原片（仅 C++ `--cam-dir`）**：一个目录，每台相机一个 `*_<相机>.mp4`（3840×2160
H.264），相机名取自 `cpp/models/stitch.lut`。实测目录
`D:\WindowsProject\workspace\SWIM\20260730-4k-raw`，与上面的画布是同一场录制：

```
20260730_170731_cam1.mp4 ... _cam6.mp4    六路 4K 原片（各约 2.4 GB）
```

Plan A 额外需要六路原始相机视频，只有旧数据集 `data/20260629/` 有：

```
merged_3000f.mp4      画布
cam1.mp4 ... cam6.mp4 六路原始相机视频（3840x2160）
```

用其他数据时通过环境变量指定：

```bash
CANVAS=/path/to/merged.mp4 bash scripts/run.sh C   # Plan B/C 只要画布
DATA_DIR=/path/to/20260629 bash scripts/run.sh A   # Plan A 还要同目录下的 cam*.mp4
```

> **相机顺序（仅 Plan A）**：`--camera-videos` 必须按 `pool_mesh.json` 里 `meshes` 数组的
> 顺序传入，即 `cam4 cam3 cam2 cam5 cam6 cam1`。mesh 的 `texture_basename` 依次是
> `camera_3/2/1/4/5/6`，与文件名并非同序 —— 这个对应关系在标定时确定，不要改动。
> **`20260730-4k-raw` 那批文件的对应是另一组** `cam3 cam2 cam1 cam4 cam5 cam6`
> （实测认出来的，写在 `cpp/tools/build_stitch_lut.py` 的 `POOL_CAMERA_IDS`）。

## 常用参数

```bash
bash scripts/run.sh --help                        # 全部参数
bash scripts/run.sh C --signal wrist_x_head       # 换划水信号
bash scripts/run.sh C --stroke-type breaststroke  # 换泳姿
bash scripts/run.sh C --kpt-thr 0.5               # 提高单点置信度门槛（影响信号/插值/绘制）
bash scripts/run.sh C --pose-score-thr 0.5        # 仅 Plan A：17 点均值低于此值才换相机补检
bash scripts/run.sh C --codec mp4v                # 换编码（默认 h264）
DRAW_KEYPOINTS= bash scripts/run.sh C             # 关掉骨架叠加（渲染更快）
```

`--kpt-thr` 与 `--pose-score-thr` 是**两个不同的东西**，只是默认值都是 0.4：前者是单个
关键点的可信门槛，后者是 Plan A 判断"这一次推理整体够不够好、要不要换相机再推一次"的门槛。

调参时不必重跑 GPU：`output_dir/cache.pkl` 存在时会跳过 Stage1/2，直接从 Stage3 起算
（缓存键是「影响 Stage1/2 的参数 + 输入视频与权重文件的大小/mtime」哈希，换 plan、
换权重、改检测阈值都会自动失效；只改 `--signal`/`--stroke-type`/绘制选项则照旧命中）。
需要强制重跑推理时删掉该文件。

## 环境说明

Python 侧版本被锁在 **Python 3.10 + torch 2.1.0(cu121) + mmcv 2.1.0 + mmpose 1.3.1**，
原因是 mmcv 的 CUDA 算子只有预编译 wheel 可用，而 OpenMMLab 只为特定 torch/CUDA
组合发布 wheel；`cu121/torch2.1.0` 是同时满足 mmpose 1.3.1 与 mmdet 3.2.0
（两者都要求 `mmcv<2.2.0`）的组合。内网环境下 `scripts/install.sh` 需要代理才能取到 mmcv wheel：

```bash
export https_proxy=http://<proxy>:<port> http_proxy=http://<proxy>:<port>
bash scripts/install.sh
```

C++ 侧只需 CUDA 12.x + TensorRT 10.11 + OpenCV 4.x + ffmpeg CLI，不依赖上面这套 Python
（只有导出 ONNX 与烘拼接表用到）。六路现拼（`--cam-dir`）还要 FFmpeg 的 libav* 开发库
（含 `h264_cuvid`），缺了只是这一路不编，其余功能不变。Windows 细节见
[`docs/windows.md`](docs/windows.md)。

## 性能参考

### Python Plan A：H800，`data/20260629`，3000 帧（约 100 秒视频，19.3 人/帧，81 个 track）

| 阶段 | 耗时 | 占比 | 主要成本 |
| --- | --- | --- | --- |
| Stage1 检测跟踪 | 146s | 11% | 3000 次画布整帧 YOLO |
| Stage2 关键点 | 683s | 53% | 推理 58% / 4K 解码 24% / 几何 14% / 补检 4% |
| Stage3 指标 | <1s | ~0% | 纯 CPU 信号处理 |
| Stage4 渲染 | 463s | 36% | 逐帧编码（编码本身占 85%） |

合计约 21.5 分钟，即约 13 倍实时。两个值得知道的细分结论：

- Stage2 日志里的 `inference` 有 **64% 实际花在 CPU 上**（把 4K 整帧按 bbox 仿射采样到
  192x256，逐 bbox 串行），GPU 只占 36%。所以关掉 `flip_test` 只省 14% 而非一半。
- Stage4 的 463 秒里**绘制只占约 4%**，85% 是视频编码——与算法无关。

有 `cache.pkl` 时跳过 Stage1+2，可省掉全程的 64%。

### C++ Plan C：`data/20260730`，3000 帧（8.0 人/帧，63 个 track，划水合计 376 次）

| 机器 | 纯分析 | 实时预览 `--preview` | 渲染落盘 `--out` |
| --- | --- | --- | --- |
| RTX 4080 Laptop | 12.8–15.5 ms/帧（**65–78 fps**） | 14.3–15.5 ms（65–70 fps） | 29.0–37.6 ms（27–35 fps） |
| H800 | 12.4 ms/帧（**80 fps**） | — | 41.6 ms（24 fps） |

全程 GPU 驻留，只回读关键点（约 9 KB/帧）。瓶颈已从 GPU 推理转到 CPU 解码与编码 ——
GPU 段稳定在 6.2 ms（H800）/ 6.8–8 ms（4080 Laptop）。

### C++ 六路现拼：`20260730-4k-raw` 六路 4K，3000 帧，RTX 4080 Laptop

26.2 ms/帧（**38.2 fps**），88 个 track、划水合计 387 次。慢一倍是因为
**NVDEC 已跑满 100%**：六路 4K 并发的裸解码就要 25.8 ms/帧，拼接 kernel 只占约 1 ms。
画布与离线 CPU 参考差 mean|d| 0.34 灰阶（最大 3，100% 在 2 灰阶内）。
接实际 zcam 流后解码这一项会消失。

拼接的语义与口径、显存拆解、逐段耗时见 [`cpp/README.md`](cpp/README.md)。
