# swim_6cam_analyse

六路相机拼接全景视频的游泳动作分析：**逐人划水次数 + 瞬时速度 + 标注视频**。

输入一段六机位拼接的泳池俯视全景视频（外加六路原始相机视频），输出每个泳者的
划水次数、逐帧速度曲线，以及叠加了检测框 / ID / 划水数 / 速度 / 关键点骨架的视频。

## 快速开始

```bash
bash install.sh                # 建 .venv、装依赖、跑自检和单元测试
bash run.sh C                  # Plan C（推荐），默认数据
bash run.sh C --max-frames 300 # 先跑 300 帧确认链路
bash run.sh A                  # Plan A（交接原版，需六路原相机视频）
```

结果落在 `output/<视频名>_plan<X>/`：

| 文件 | 内容 |
| --- | --- |
| `<视频名>_plan<X>.mp4` | 标注视频（H.264） |
| `result.json` | 每个 ID 的划水次数与速度采样点 |
| `signal_plots/id*.png` | 每人的划水信号图（原始 + 平滑 + 事件标记） |
| `cache.pkl` | Stage1/2 中间结果，同 plan 重跑时自动复用 |

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
  **推荐用这套。**

> 评测口径：检测指标在人工标注 val（36 帧/639 框）上算，关键点用 PCK（误差按人体框高
> 归一化）而非 COCO OKS —— 本数据目标 area 中位 17880，OKS 容差达 17–29 px 而人体框高
> 仅 78 px，模型差异会被度量淹没。

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
configs/
  pool_mesh.json                 泳池 mesh 标定（六路相机的三角面片 + UV），仅 Plan A 用
  rtmpose-m_swim-256x192.py      Plan A 的原相机 RTMPose 推理配置
  rtmpose-m_canvas-192x256.py    Plan C 的画布 RTMPose 推理配置
weights/
  yolo_swim_detect.pt            泳者检测（微调），Plan A/C 共用
  rtmpose_m_swim.pth             原相机 RTMPose-m（Plan A）
  plans/planB_yolo26x_pose_canvas.pt   画布 yolo26-pose 一体（Plan B）
  plans/planC_rtmpose_m_canvas.pth     画布 RTMPose-m（Plan C）
src/swim_analyse/
  cli.py         统一入口：--plan A/B/C，Stage3/4 共用
  plans.py       三套方案的实现，统一接口 run() -> (all_boxes, raw_seq)
  geometry.py    canvas <-> source 双向映射（网格索引加速），Plan A 用
  tracking.py    检测框去重 + IoU 贪心跟踪
  pose.py        RTMPose 封装、COCO17 定义、关键点插值
  metrics.py     划水信号与计数、瞬时速度、信号图
  draw.py        骨架 / 标签 / 调试裁剪视频
  video.py       多路视频的帧级随机读取（带帧缓存），Plan A 用
tests/test_core.py               纯逻辑单元测试（无需 GPU 和数据）
```

三套方案共享同一份数据约定，因此新增方案只需在 `plans.py` 里实现 `run()`：

```python
all_boxes  {frame_idx: [(track_id, x1, y1, x2, y2, conf)]}    画布坐标
raw_seq    {track_id: [(frame_idx, kpts(17,2), scores(17,), cam_idx)]}
           cam_idx = -1 表示关键点已在画布坐标系
```

## 数据准备

`data/20260629/` 下需要：

```
merged_3000f.mp4      六路拼接后的全景视频（5002x2102）
cam1.mp4 ... cam6.mp4 六路原始相机视频（3840x2160）
```

用其他数据时通过环境变量或参数指定：

```bash
DATA_DIR=/path/to/data CANVAS=/path/to/merged.mp4 bash run.sh C
```

Plan B/C 只需 `CANVAS`（画布视频）；Plan A 还需 `DATA_DIR` 下的六路原相机视频。

> **相机顺序（仅 Plan A）**：`--camera-videos` 必须按 `pool_mesh.json` 里 `meshes` 数组的
> 顺序传入，即 `cam4 cam3 cam2 cam5 cam6 cam1`。mesh 的 `texture_basename` 依次是
> `camera_3/2/1/4/5/6`，与文件名并非同序 —— 这个对应关系在标定时确定，不要改动。

## 常用参数

```bash
bash run.sh C --help                          # 全部参数
bash run.sh C --signal wrist_x_head           # 换划水信号
bash run.sh C --stroke-type breaststroke      # 换泳姿
bash run.sh C --kpt-thr 0.5                   # 提高关键点置信度门槛
bash run.sh C --codec mp4v                    # 换编码（默认 h264）
DRAW_KEYPOINTS= bash run.sh C                 # 关掉骨架叠加（渲染更快）
```

调参时不必重跑 GPU：`output_dir/cache.pkl` 存在时会跳过 Stage1/2，直接从 Stage3 起算
（缓存记录了 plan 名，换 plan 会自动失效重算）。需要强制重跑推理时删掉该文件。

## 环境说明

版本被锁在 **Python 3.10 + torch 2.1.0(cu121) + mmcv 2.1.0 + mmpose 1.3.1**，
原因是 mmcv 的 CUDA 算子只有预编译 wheel 可用，而 OpenMMLab 只为特定
torch/CUDA 组合发布 wheel；`cu121/torch2.1.0` 是同时满足 mmpose 1.3.1 与
mmdet 3.2.0（两者都要求 `mmcv<2.2.0`）的组合。内网环境下 `install.sh` 需要代理
才能取到 mmcv wheel：

```bash
export https_proxy=http://<proxy>:<port> http_proxy=http://<proxy>:<port>
bash install.sh
```

## 性能参考

单张 H800，3000 帧 5002x2102 全景（约 100 秒视频，平均 19.3 人/帧，共 81 个 track）：

| 阶段 | 耗时 | 占比 | 主要成本 |
| --- | --- | --- | --- |
| Stage1 检测跟踪 | 146s | 11% | 3000 次画布整帧 YOLO |
| Stage2 关键点 | 683s | 53% | 推理 58% / 4K 解码 24% / 几何 14% / 补检 4% |
| Stage3 指标 | <1s | ~0% | 纯 CPU 信号处理 |
| Stage4 渲染 | 463s | 36% | 逐帧 mp4v 编码（编码本身占 85%） |

合计约 21.5 分钟，即约 13 倍实时。两个值得知道的细分结论：

- Stage2 日志里的 `inference` 有 **64% 实际花在 CPU 上**（把 4K 整帧按 bbox 仿射采样到
  192x256，逐 bbox 串行），GPU 只占 36%。所以关掉 `flip_test` 只省 14% 而非一半。
- Stage4 的 463 秒里**绘制只占约 4%**，85% 是视频编码——与算法无关。

有 `cache.pkl` 时跳过 Stage1+2，可省掉全程的 64%。
