# swim_6cam_analyse

六路相机拼接全景视频的游泳动作分析：**逐人划水次数 + 瞬时速度 + 标注视频**。

输入一段六机位拼接的泳池俯视全景视频（外加六路原始相机视频），输出每个泳者的
划水次数、逐帧速度曲线，以及叠加了检测框 / ID / 划水数 / 速度 / 关键点骨架的视频。

## 快速开始

```bash
bash install.sh          # 建 .venv、装依赖、跑自检和单元测试
bash run.sh              # 跑默认数据（data/20260629/merged_3000f.mp4）
bash run.sh --max-frames 300   # 先跑 300 帧确认链路
```

结果落在 `output/<视频名>/`：

| 文件 | 内容 |
| --- | --- |
| `<视频名>_annotated.mp4` | 标注视频 |
| `result.json` | 每个 ID 的划水次数与速度采样点 |
| `signal_plots/id*.png` | 每人的划水信号图（原始 + 平滑 + 事件标记） |
| `cache.pkl` | Stage1/2 中间结果，重跑时自动复用 |

## 工作原理

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

**为什么要绕回原始相机**：拼接画布是六路 4K 拉伸重采样后的产物，单个泳者在画布上
只有一两百像素宽，关键点回归不可靠。画布框经 mesh 几何映射回原始相机后能拿到几倍
的有效分辨率，因此检测跟踪在画布上做（全局、ID 连续），关键点在原图上做（清晰）。

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
  pool_mesh.json              泳池 mesh 标定（六路相机的三角面片 + UV）
  rtmpose-m_swim-256x192.py   RTMPose 纯推理配置
weights/
  yolo_swim_detect.pt         YOLO 泳者检测（微调）
  rtmpose_m_swim.pth          RTMPose-m 关键点（COCO17，游泳数据微调）
src/swim_analyse/
  pipeline.py    四个 Stage 的编排与 CLI
  geometry.py    canvas <-> source 双向映射（网格索引加速）
  tracking.py    YOLO 检测 + IoU 贪心跟踪
  pose.py        RTMPose 封装、COCO17 定义、关键点插值
  metrics.py     划水信号与计数、瞬时速度、信号图
  draw.py        骨架 / 标签 / 调试裁剪视频
  video.py       多路视频的帧级随机读取（带帧缓存）
tests/test_core.py            纯逻辑单元测试（无需 GPU 和数据）
```

## 数据准备

`data/20260629/` 下需要：

```
merged_3000f.mp4      六路拼接后的全景视频（5002x2102）
cam1.mp4 ... cam6.mp4 六路原始相机视频（3840x2160）
```

用其他数据时通过环境变量或参数指定：

```bash
DATA_DIR=/path/to/data CANVAS=/path/to/merged.mp4 bash run.sh
```

> **相机顺序**：`--camera-videos` 必须按 `pool_mesh.json` 里 `meshes` 数组的顺序传入，
> 即 `cam4 cam3 cam2 cam5 cam6 cam1`。mesh 的 `texture_basename` 依次是
> `camera_3/2/1/4/5/6`，与文件名并非同序 —— 这个对应关系在标定时确定，不要改动。

## 常用参数

```bash
bash run.sh --help                          # 全部参数
bash run.sh --signal wrist_x_head           # 换划水信号
bash run.sh --stroke-type breaststroke      # 换泳姿
bash run.sh --kpt-thr 0.5                   # 提高关键点置信度门槛
bash run.sh --debug-crops                   # 每人导出一个原视角裁剪+骨架小视频
DRAW_KEYPOINTS= bash run.sh                 # 关掉骨架叠加（渲染更快）
```

调参时不必重跑 GPU：`output_dir/cache.pkl` 存在时会跳过 Stage1/2，直接从 Stage3 起算。
需要重跑推理时删掉该文件。

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
