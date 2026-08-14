# 给 Agent 的导航（先读这页，再决定读哪些文件）

本仓库有**两条独立实现**同一套游泳分析算法。**先确认你在哪条线上，只读那条线的文件。**
跨线改动几乎总是错的：两边刻意各自维护，靠数值对照保证一致，不共享代码。

| 你的任务 | 走哪条线 | 只需读 |
| --- | --- | --- |
| 实时 / 上线 / 提速 / 显存 / TensorRT / CUDA kernel | **C++**（`cpp/`） | `cpp/README.md` + `cpp/include/swim/*.h` |
| 换检测或关键点模型、Plan A/B/C 对比、离线批处理、评测口径 | **Python**（`src/swim_analyse/`） | `README.md` +『Python 侧速查』 |
| 改划水计数 / 速度算法 | **两条都要改**（见『同步契约』） | `src/swim_analyse/metrics.py` + `cpp/src/metrics.cpp` |
| 双击运行、装环境、CI | 入口脚本 | 本页『入口脚本』一节 |
| 权重从哪来、怎么复现训练 | — | `docs/权重来源与复现.md` |
| Windows 环境坑（编码、TRT、ffmpeg） | — | `docs/windows.md` |

两条线的关系：C++ 只实现 **Plan C**（画布 detect + 画布 RTMPose），是 Python Plan C 的
重写，用同一份权重（`weights/` → `cpp/tools/export_onnx.py` → ONNX → TRT engine）。
Python 侧还有 Plan A（跨相机 mesh）与 Plan B（一体模型），C++ 没有也不打算有。

## 入口脚本（改这些之前先看 `docs/windows.md` 的编码规则）

| 入口 | 干什么 | 平台 |
| --- | --- | --- |
| `build.bat` | 构建 C++ + 导出 ONNX（双击） | Windows |
| `run_preview.bat` | C++ 实时预览窗口（双击 / 拖视频 / 传 rtsp URL） | Windows |
| `run_analyse.bat` | C++ 批处理出 json，加 `--out` 出标注视频（双击） | Windows |
| `install.sh` | 建 `.venv` 装 Python 依赖 + 自检 | Linux / Git Bash |
| `run.sh` | Python 链路，`bash run.sh [A\|B\|C] [参数]` | Linux / Git Bash |
| `test.sh` | `bash test.sh` 秒级自检；`--full` 加 30 帧 GPU 冒烟 | Linux / Git Bash |

三个 `.bat` 共用 `scripts/env.bat`（定位 TensorRT、检查 exe/onnx/ffmpeg）。
C++ 侧没有 shell 入口，Linux 上直接调 `cpp/build/swim_analyse`（`cpp/README.md` 有命令）。

## Python 侧速查

| 关心什么 | 文件 |
| --- | --- |
| 命令行参数、Stage 编排、cache、渲染 | `cli.py` |
| Plan A/B/C 各自怎么出框与关键点 | `plans.py`（基类跑循环，子类只实现 `_setup`/`_frame`） |
| 划水信号、计数、速度、信号图 | `metrics.py` |
| canvas ↔ 原相机双向映射（仅 Plan A） | `geometry.py` |
| 去重 + IoU 跟踪 | `tracking.py` |
| RTMPose 封装、COCO17、关键点插值 | `pose.py` |
| 骨架/标签绘制 | `draw.py` |
| 多路视频帧级随机读（仅 Plan A） | `video.py` |

## C++ 侧速查

| 关心什么 | 文件 |
| --- | --- |
| 命令行、渲染、预览窗口、ffmpeg 写出 | `src/main.cpp` |
| 三线程编排、显存预分配、计时 | `src/pipeline.cpp` + `include/swim/pipeline.h` |
| 所有 CUDA kernel（前处理/去重/裁切/SimCC 解码） | `src/kernels.cu` |
| TRT engine 构建与缓存、身份戳 | `src/trt_engine.cpp` |
| 解码（ffmpeg 管道 / OpenCV）与预取 | `src/frame_source.cpp` |
| 子进程管道（解码与编码共用） | `src/proc.cpp` |
| 跟踪 + 划水 + 速度（在线版） | `src/metrics.cpp` |

## 同步契约（改一处必须改另一处，编译器不会提醒你）

1. **算法常量**：`cpp/src/metrics.cpp` 顶部的常量是**手抄** `src/swim_analyse/metrics.py`
   的（`MEDIAN_WINDOW`/`SG_WINDOW`/`MIN_STROKE_INTERVAL`/`ADAPTIVE_WINDOW`/
   `ADAPTIVE_PROMINENCE_RATIO`/`HYSTERESIS_RATIO`/`ZC_MIN_INTERVAL`/
   `SPEED_SAMPLE_DT`/`SPEED_WINDOW_SEC`）。改任一侧必须同步，否则两边结果静默分叉。
2. **缓存键**：`cli.py` 的 `_KEY_ARGS` 必须登记**任何影响 Stage1/2 的新参数**。
   漏登记的后果是 `cache.pkl` 静默返回旧结果 —— 不报错，只是数字不对。
3. **CLI 参数命名**：Python 与 C++ 的同义参数保持同名（`--conf`/`--kpt-thr`/
   `--containment`/`--track-iou`/`--track-max-lost`/`--ppm`/`--signal`/`--stroke-type`），
   便于两边对照跑。新增参数请照此命名。
4. **权重与 ONNX**：改了 `weights/` 里的模型，必须重跑 `cpp/tools/export_onnx.py`；
   engine 会因 ONNX 的 mtime/size 变化自动重建（身份戳机制，不必手删）。
5. **默认数据集**：Python `run.sh` 与三个 `.bat` 默认都指向 `data/20260730/merged_3000f.mp4`，
   这样两条线的数字可直接比。**Plan A 例外**，它需要六路原相机视频，只有 `data/20260629/` 有
   （该目录未随包分发，本机没有 → Plan A 本机跑不了）。

## 怎么验证一处改动

```bash
bash test.sh                 # 语法 + 单元测试（约 2 秒，无需 GPU）
bash test.sh --full          # 再加 Python Plan C 与 C++ 各 30 帧真实冒烟
```

改了 C++ 且要证明"没改数值"，跑全量 3000 帧对基线：

```bash
cpp/build/Release/swim_analyse.exe --input data/20260730/merged_3000f.mp4 \
    --models cpp/models --json out.json --show-fps
```

**基线（RTX 4080 Laptop，`data/20260730/merged_3000f.mp4`，3000 帧，默认参数）**：
24107 人次、63 个 track、划水合计 376 次。纯分析与渲染两种模式的 `--json` 与
`--dump` 应**逐字节相同**；不同就说明渲染路径动了帧缓冲。
`--decoder cpu`（OpenCV/MSMF）是**另一组合法数字**（24245 人次、63 track），
因为 MSMF 与 swscale 的 YUV→BGR 换算不同，不是回归。

改了 Python 侧，用同一段跑 `bash run.sh C --max-frames N` 前后对比 `result.json`。
注意 `output/*/cache.pkl` 会跳过 Stage1/2，验证推理改动前先删。

## 数字的口径（引用性能数据必须带上这三项）

任何一个性能或统计数字都必须标注 **数据集 + 机器 + Plan**，否则无法比较：

| 口径 | 典型数字 |
| --- | --- |
| Plan A / `data/20260629` / H800 | 19.3 人每帧、81 track、3000 帧约 21.5 分钟 |
| Plan C(C++) / `data/20260730` / RTX 4080 Laptop | 8.0 人每帧、63 track、376 次划水、65–78 fps |
| Plan C(C++) / `data/20260730` / H800 | 12.4 ms 每帧（80 fps） |

两个数据集的人数密度差一倍以上，跨口径比较毫无意义。

## 文档归属（改文档前先确认该数字归谁管）

| 文档 | 唯一负责的内容 |
| --- | --- |
| `README.md` | 项目总览、入口、Plan A/B/C 对比、数据准备、Python 用法 |
| `cpp/README.md` | C++ 链路的一切：数据流、依赖、构建、显存、性能、与 Python 的对齐 |
| `docs/pipeline.html` | Plan A 的公式推导与逐段耗时拆解 |
| `docs/权重来源与复现.md` | 权重来源、训练复现 |
| `docs/windows.md` | Windows 环境要求与踩过的坑（含脚本编码规则） |
| 本页 | 导航、同步契约、验证方法、数字口径 |

同一个数字只在负责它的文档里写一份，别处引用时给链接不复制。

## 不要做的事

- 不要把 Python 与 C++ 的代码合并、或让一边 import/调用另一边。
- 不要删 `cli.py` 的 `--iou`（对传统 NMS 权重仍生效，yolo26 end2end 下才是空转，注释已说明）。
- 不要改 `kernels.cu` 里的**字面量**为中文：nvcc 前端在 ACP=936 机器上按 ANSI 解 `.cu`，
  中文字面量会吞掉右引号（加 BOM 与 `/utf-8` 都无效）。注释可以用中文。
- 不要改 `.bat`/`.ps1` 的编码约定，见 `docs/windows.md`。
- 不要重命名 `pipeline.cpp` 的计时标签（`0_src_wait` / `1_pre+detect` /
  `2_crop+pose+decode` / `7_track_metrics` / `8_frame_d2h`）—— 文档与对照脚本按名字取值。
- 不要改 `configs/pool_mesh.json` 的 `meshes` 顺序，也不要改 Plan A 传相机视频的顺序
  （`cam4 cam3 cam2 cam5 cam6 cam1`），标定时定死的。
