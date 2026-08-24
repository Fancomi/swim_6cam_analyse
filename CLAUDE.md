# 给 Agent 的导航（先读这页，再决定读哪些文件）

本仓库有**两条独立实现**同一套游泳分析算法。**先确认你在哪条线上，只读那条线的文件。**
跨线改动几乎总是错的：两边刻意各自维护，靠数值对照保证一致，不共享代码。

| 你的任务 | 走哪条线 | 只需读 |
| --- | --- | --- |
| 实时 / 上线 / 提速 / 显存 / TensorRT / CUDA kernel | **C++**（`cpp/`） | `cpp/README.md` + `cpp/include/swim/*.h` |
| 六路原相机 → 画布的**上游拼接**（NVDEC + CUDA） | **C++** | `cpp/README.md`「上游拼接」+ `cpp/include/swim/stitch.h` |
| 换检测或关键点模型、Plan A/B/C 对比、离线批处理、评测口径 | **Python**（`src/swim_analyse/`） | `README.md` +『Python 侧速查』 |
| 改划水计数 / 速度算法 | **两条都要改**（见『同步契约』） | `src/swim_analyse/metrics.py` + `cpp/src/metrics.cpp` |
| 双击运行、装环境、CI | 入口脚本 | 本页『入口脚本』一节 |
| 权重从哪来、怎么复现训练 | — | `docs/权重来源与复现.md` |
| 接现场 ZCam 相机（配 4K30、拉流、踩过的坑） | — | `docs/cameras.md` |
| Windows 环境坑（编码、TRT、ffmpeg） | — | `docs/windows.md` |

两条线的关系：C++ 只实现 **Plan C**（画布 detect + 画布 RTMPose），是 Python Plan C 的
重写，用同一份权重（`weights/` → `cpp/tools/export_onnx.py` → ONNX → TRT engine）。
Python 侧还有 Plan A（跨相机 mesh）与 Plan B（一体模型），C++ 没有也不打算有。

C++ 侧的输入有两条，产出同一个 `GpuFrame`，下游不感知差异：
`--input` 读已拼好的画布（mp4 / rtsp），`--cam-dir` 读六路 4K 原片在 GPU 上现拼
（NVDEC + CUDA，无中间画布 mp4）。**拼接的几何不在 C++ 里算** —— 由
`cpp/tools/build_stitch_lut.py` 把 `configs/pool_mesh.json` 烘成逐像素查找表
`cpp/models/stitch.lut`，kernel 只做 gather，详见 `cpp/README.md`「上游拼接」。

## 入口脚本（全在 `scripts/`；改之前先看 `docs/windows.md` 的编码规则）

**两个维度，别混**：`.bat` 选的是「结果去哪」，参数选的是「输入从哪来」。
同一份算法（逐帧 detect + pose + 跟踪 + 划水/速度，无缓存），四种组合都能跑。

| 入口 | 结果去哪 | 平台 |
| --- | --- | --- |
| `scripts/build.bat` | 构建 C++ + 导出 ONNX + 烘拼接表（首次一次） | Windows |
| `scripts/preview.bat` | **开窗口实时看**（按 q/ESC 停） | Windows |
| `scripts/analyse.bat` | **写文件**（json，加 `--out` 出标注 mp4），跑完退出 | Windows |
| `scripts/install.sh` | 建 `.venv` 装 Python 依赖 + 自检 | Linux / Git Bash |
| `scripts/cams.sh` | 六路 ZCam：探测 / 配 4K30 / 生成清单 / 直接起预览 | Git Bash |
| `scripts/run.sh` | Python 参考链路，`bash scripts/run.sh [A\|B\|C] [参数]` | Linux / Git Bash |
| `scripts/test.sh` | 秒级自检；`--full` 加 30 帧 GPU 冒烟 | Linux / Git Bash |
| `scripts/dist.bat` | **打交付包**（双击：先构建再打包，`inc` 出增量、`zip` 顺手压缩） | Windows |
| `scripts/dist.ps1` | 同上的实现（`dist.bat` 调它；也可单独跑，见下） | Windows PowerShell |

两个 `.bat` 的第一个参数选输入源，语法完全一致（`env.bat` 统一解析）：

| 第一个参数 | 输入 | 传给二进制 |
| --- | --- | --- |
| 省略 / `canvas` | 已拼好的全景视频（默认数据集） | `--input` |
| `6cam` | 六路 4K 原片，GPU 上现拼 | `--cam-dir` |
| 视频文件路径（可拖拽） | 那段全景视频 | `--input` |
| 目录路径（可拖拽） | 那个目录里的六路片段 | `--cam-dir` |
| **相机清单文件** | 每行 `<相机>=<地址>`，地址可为 `rtsp://` | `--cam-dir` |
| `rtsp://…` | 单路直播画布流（只有 preview 能用） | `--input` |

所以「带拼接的实时 pose」= `scripts\preview.bat 6cam`，「带拼接的批处理」=
`scripts\analyse.bat 6cam`。默认源可用 `SWIM_CANVAS` / `SWIM_CAM_DIR` 覆盖。
**接现场相机**走 `bash scripts/cams.sh`（它生成 `configs/cameras.txt` 再喂给
`--cam-dir`），细节见 `docs/cameras.md`。

### 交付包（双击 `scripts/dist.bat`）

**两级，对应「第一次拷过去」与「之后每次更新」**。两级都先跑 `build.bat`（已构建则空转），
所以「双击、等、拷目录」就是全部流程：

| 命令 | 产物 | 大小 |
| --- | --- | --- |
| `dist.bat` / `dist.ps1` | 全能包 `dist/swim_analyse/` | 约 3.0 GB / 42 文件 |
| `dist.bat inc` / `dist.ps1 -Inc` | 增量包 `dist/swim_analyse_update/` + `update.bat` | 0.5 MB |
| `dist.bat rebase` / `dist.ps1 -Rebase` | 只重写基准，不构建不出包 | — |
| 追加 `zip` / `-Zip` | 同名 `.zip`（Windows 自带 `tar.exe -a`，退到 `Compress-Archive`） | — |

**打包用 PowerShell 而不是 bash**：交付链路只在 Windows 上跑，不该拖上 Git Bash 这个
依赖。`cmd.exe` 调起的 bash 是**非登录 shell**，不读 `/etc/profile`，`dirname`/`md5sum`/
`mktemp` 全部 `command not found`，而 `$(dirname …)` 返回空又会把错误伪装成
「找不到 VC 运行库」，把人引去查一个完全健康的 Visual Studio（实测踩过）。
PowerShell 5.1 随系统装，`Get-FileHash -Algorithm MD5` / `Copy-Item`（会跟随 winget
的软链接取到 ffmpeg 真身）/ 通配 `Get-Item` 一样够用。
`dist.ps1` 含中文，所以是 **UTF-8 带 BOM + CRLF** —— 见 `docs/windows.md`。

**全能包 = 目标机零安装**，只要 NVIDIA 驱动 ≥ 550：exe + 运行期 DLL + `cudart64_12.dll` +
四个 VC 运行库 + `ffmpeg`/`ffprobe`（462 MB）+ 预烘 engine + **ONNX 与
`nvinfer_builder_resource`（合 1.8 GB）** + `stitch.lut` + `cameras.txt` 模板 +
`README.txt`（UTF-8 带 BOM）+ 两个入口 `run_1cam.bat`（单相机联调，`--input`）/
`run_6cam.bat`（六路上线，`--cam-dir`，改 `cameras.txt` 的 IP 即可）。
两个入口都**默认带 `--rot180`**（机位倒挂），现场追加 `--no-rot180` 转回正向 ——
旋转在帧源里就地完成、零额外开销，口径见 `cpp/README.md`「画面定向」。
入口自己 `set "PATH=%~dp0;%PATH%"`，所以自带的 ffmpeg 一定被用上（回退 OpenCV/MSMF
是 5.7 fps vs 77.7，且帧率报成 30.00 而真值 59.94）。
**带 ONNX 是刻意的**：engine 与「GPU 架构 + TRT 版本」烘死，带着它换代机器能自己现烘
（几分钟，之后秒开）。要省这 1.8 GB 就 `SWIM_DIST_LEAN=1`，代价是换架构即失效。

**换代显卡是两件独立的事，别以为 ONNX 全包了**：engine 靠 ONNX 现烘；**我们自己的
CUDA kernel 只能靠编进 exe 的 cubin**（`kernels.cu` / `stitch.cu`），ONNX 与它无关。
所以 `scripts/build.bat` 默认 `SWIM_CUDA_ARCH=89;120`（RTX40 + RTX50/Blackwell），
`dist.ps1` 打包时用 `cuobjdump --list-elf` 核一遍、缺档就告警，并把实际架构列表写进
交付包的 `README.txt`。TRT 侧不必担心：`nvinfer_10.dll` 自带 sm_100/103/120 的 cubin。
缺档不会报错，只是退化成驱动 JIT 那份 PTX —— 首次启动多等几十秒，没验证过。

**增量包 = 基础件（每次必带）+ 变了的大件**。基础件是 exe 与三个生成文本
（两个入口 + `README.txt`），合 0.5 MB —— 改代码只动这几个，不值得为省这点体积去赌
「检测对不对」，所以一律带上、不做判断。大件（DLL / ffmpeg / engine / ONNX /
`stitch.lut`）按 `dist/<名>.manifest` 比 md5，真变了才进包，并打印一行提示
「建议改打全能包」。
`update.bat` **只覆盖不删除**，且跳过 `cameras.txt`（现场 IP 在里面）。

**基准（manifest）只在第一次全能包时写，之后只有 `dist.bat rebase` 会重写。**
它记的是「**目标机手上是哪一份**」，而本地多打一次全能包并不等于拷过去了：旧实现
每次全能包都覆盖它，于是「打了全能包 #2 没部署、接着打增量」会把 #2 当基准，漏掉
目标机其实还缺的大件（现场表现为更新完仍是旧行为）。所以流程是
**打全能包 → 拷去部署 → `dist.bat rebase`**；忘了 rebase 只会让后续增量多带几个
大件，不会漏，方向是安全的那一边。`rebase` 刻意不跑 `build.bat`（重新构建可能重导
ONNX、把每个 engine 的身份戳都推走）。

一份清单（`<md5> <包内路径> <源路径>`）同时喂两级，两者对「包里该有什么」不会分叉；
顺手拿到的 md5 又当拷贝校验 —— 一次静默坏拷贝在目标机上表现为「反序列化失败」，
与「换了显卡架构」症状一模一样（实测踩过）。生成的文本文件（入口 / README /
`cameras.txt`）也走同一条路，改了入口脚本增量包会自动带上。

engine 在包里统一改成**裸名**（`models/detect.engine`）：目标机的身份戳必然与本机不同
（ONNX 的 mtime 变了），裸名才是它能认的那条路。`trt_engine.cpp` 按
**身份戳名 → 裸名 → 现烘** 三级找 engine，裸名那份不可用时有 ONNX 就现烘、没有就报错。

`scripts/env.bat` 是两个 `.bat` 共用的**命令行解析 + 源解析 + 前置检查**（定位
TensorRT、检查 exe/onnx/lut/ffmpeg），设好 `MODE`/`INPUT`/`ARGS`/`NAME`/`LABEL`/`EXE`
回给调用方，不单独运行。两个 launcher 因此各只剩十几行，差异只有「窗口 vs 文件」。
**所有脚本都自己 `cd` 到仓库根**（`.bat` 用 `%~dp0..`，`.sh` 用 `BASH_SOURCE/..`），
所以从任何目录双击或调用都一样，脚本内部的相对路径一律以根为基准。
C++ 侧没有 shell 入口，Linux 上直接调 `cpp/build/swim_analyse`（`cpp/README.md` 有命令）。

## 目录职责（放新文件前先对一眼）

| 目录 | 只放什么 |
| --- | --- |
| `scripts/` | 用户入口脚本（`.bat` / `.sh`）。不放 Python 模块，也不放构建产物 |
| `src/swim_analyse/` | Python 参考链路的库代码，唯一入口是 `cli.py` 的 `main()` |
| `cpp/src` `cpp/include/swim` | C++ 实现与其头文件 |
| `cpp/tools/` | 服务 C++ 链路的一次性 Python 工具（`export_onnx.py`、`build_stitch_lut.py`） |
| `cpp/models/` | ONNX、TRT engine、拼接查找表（除 `pose_meta.json` 外都不入库） |
| `configs/` | 模型推理配置与泳池 mesh 标定 |
| `tests/` | pytest，纯逻辑、无需 GPU 与数据 |
| `docs/` | 环境与推导类文档；性能数字归属见下面『文档归属』 |
| `data/` `output/` `weights/` | 输入、产物、权重，均不入库 |
| `dist/` | `scripts/dist.ps1` 出的交付包与 `*.manifest`（增量的基准），不入库；不要手工往里放东西（下次打包会整目录重建） |

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
| 所有推理侧 CUDA kernel（前处理/去重/裁切/SimCC 解码） | `src/kernels.cu` |
| TRT engine 构建与缓存、身份戳 | `src/trt_engine.cpp` |
| 解码（ffmpeg 管道 / OpenCV）与预取 | `src/frame_source.cpp` |
| **六路 NVDEC 解码 + 画布环** | `src/stitch_source.cpp` |
| **拼接 kernel + 查找表加载** | `src/stitch.cu` + `include/swim/stitch.h` |
| 拼接查找表的烘制（唯一写端） | `tools/build_stitch_lut.py` |
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
5. **拼接查找表的二进制格式**：写端只有 `cpp/tools/build_stitch_lut.py` 的
   `HEADER`/`CAMERA` 两个 `struct.Struct`，读端只有 `cpp/src/stitch.cu` 的
   `LutHeader`/`LutLane`。改一侧必须改另一侧 —— C++ 侧有 `static_assert(72/48)`
   会在编译期挡住尺寸不一致，但**字段顺序变了编译器发现不了**，表现为画布错乱。
   改了 `configs/pool_mesh.json` 或 `--ppm` 也要重跑该脚本（`scripts/build.bat` 会调）。
6. **默认数据集**：`scripts/run.sh` 与两个 `.bat`（省略参数时）默认都指向
   `data/20260730/merged_3000f.mp4`，这样两条线的数字可直接比。六路现拼的默认目录是
   `env.bat` 的 `SWIM_CAM_DIR`。**Plan A 例外**，它需要六路原相机视频，只有 `data/20260629/` 有
   （该目录未随包分发，本机没有 → Plan A 本机跑不了）。

## 怎么验证一处改动

```bash
bash scripts/test.sh         # 语法 + 单元测试（约 2 秒，无需 GPU）
bash scripts/test.sh --full  # 再加 Python Plan C、C++ 画布、C++ 六路拼接各 30 帧
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

改了拼接（`stitch.cu` / `build_stitch_lut.py` / `configs/pool_mesh.json`）：

```bash
# 1) 画布逐像素对照（不涉及推理，最灵敏）
cpp/build/Release/swim_analyse.exe --cam-dir <六路片段目录> --models cpp/models \
    --max-frames 1 --dump-canvas f0.png
# 2) 全量分析数字
cpp/build/Release/swim_analyse.exe --cam-dir <六路片段目录> --models cpp/models \
    --max-frames 3000 --json out.json --show-fps
```

**拼接基线（RTX 4080 Laptop，`20260730-4k-raw` 六路原片，3000 帧）**：画布 5002×2102、
25078 人次、91 个 track、划水合计 379 次、26.2 ms/帧（38.2 fps）。
首帧画布与离线 CPU 参考差 **mean|d| 0.34 灰阶 / 最大 3 / 100% 在 2 灰阶内**。
它与 `--input` 那条路是**两组不可 diff 的数字**（画布差 0.34 灰阶就足以让短 track
数量变化），各自对自己的基线。

**`--rot180` 又各自是一组基线**（画布 24233 人次 / 90 track / 361 划水；六路现拼
25578 / 119 / 359）：detect 对定向不是旋转等变的，转与不转不能相互 diff。旋转本身
的正确性用 `--dump-canvas` 验：转过的首帧应与「不转的首帧再 `[::-1,::-1]`」
**逐字节相同**，这是灵敏且与推理无关的判据。耗时不应变化（实测 +0.1 ms/帧，
在 run 间抖动内）。口径与实现见 `cpp/README.md`「画面定向」。

改了 Python 侧，用同一段跑 `bash scripts/run.sh C --max-frames N` 前后对比 `result.json`。
注意 `output/*/cache.pkl` 会跳过 Stage1/2，验证推理改动前先删。

## 数字的口径（引用性能数据必须带上这三项）

任何一个性能或统计数字都必须标注 **数据集 + 机器 + Plan**，否则无法比较：

| 口径 | 典型数字 |
| --- | --- |
| Plan A / `data/20260629` / H800 | 19.3 人每帧、81 track、3000 帧约 21.5 分钟 |
| Plan C(C++) / `data/20260730` 画布 / RTX 4080 Laptop | 8.0 人每帧、63 track、376 次划水、65–78 fps |
| Plan C(C++) / `20260730-4k-raw` 六路现拼 / RTX 4080 Laptop | 91 track、379 次划水、38 fps |
| Plan C(C++) / `data/20260730` / H800 | 12.4 ms 每帧（80 fps） |

两个数据集的人数密度差一倍以上，跨口径比较毫无意义。
**同一场录制的「画布 mp4」与「六路现拼」也是两个口径**：画布 mp4 多经一次 H.264
编解码，逐像素差 0.34 灰阶，track 数不可比。

## 文档归属（改文档前先确认该数字归谁管）

| 文档 | 唯一负责的内容 |
| --- | --- |
| `README.md` | 项目总览、入口、Plan A/B/C 对比、数据准备、Python 用法 |
| `cpp/README.md` | C++ 链路的一切：数据流、依赖、构建、显存、性能、与 Python 的对齐 |
| `docs/pipeline.html` | Plan A 的公式推导与逐段耗时拆解 |
| `docs/权重来源与复现.md` | 权重来源、训练复现 |
| `docs/cameras.md` | ZCam 相机侧的一切：两路流分工、movfmt、拉流约束、现场坑 |
| `docs/windows.md` | Windows 环境要求与踩过的坑（含脚本编码规则） |
| 本页 | 导航、同步契约、验证方法、数字口径 |

同一个数字只在负责它的文档里写一份，别处引用时给链接不复制。

## 不要做的事

- 不要把 Python 与 C++ 的代码合并、或让一边 import/调用另一边。
- 不要删 `cli.py` 的 `--iou`（对传统 NMS 权重仍生效，yolo26 end2end 下才是空转，注释已说明）。
- 不要改 `kernels.cu` / `stitch.cu` 里的**字面量**为中文：nvcc 前端在 ACP=936 机器上
  按 ANSI 解 `.cu`，中文字面量会吞掉右引号（加 BOM 与 `/utf-8` 都无效）。注释可以用中文。
- 不要改 `.bat`/`.ps1` 的编码约定，见 `docs/windows.md`。
- 不要重命名 `pipeline.cpp` 的计时标签（`0_src_wait` / `1_pre+detect` /
  `2_crop+pose+decode` / `7_track_metrics` / `8_frame_d2h`）—— 文档与对照脚本按名字取值。
- 不要改 `configs/pool_mesh.json` 的 `meshes` 顺序。它同时定死两件事：Plan A 传相机
  视频的顺序，以及 `build_stitch_lut.py` 里 `POOL_CAMERA_IDS` 的对应关系。
  两批数据的文件名与 mesh 的对应**不同**：`data/20260629/` 是
  `cam4 cam3 cam2 cam5 cam6 cam1`（README 记的那组），`20260730-4k-raw` 是
  `cam3 cam2 cam1 cam4 cam5 cam6`（靠贴图与片段首帧的像素相关性实测出来的，
  不是按文件名猜的）。重排 mesh 或用错这组对应，拼接会静默错位。
- 不要在 `stitch.cu` 里重写三角形光栅化。覆盖判定与仿射的语义来自 OpenCV 的
  `fillConvexPoly` / `getAffineTransform`，烘表脚本直接调它们；在 kernel 里重写等于
  维护第二套边界规则，差一个像素就是接缝错位且不报错。
- 不要在 `StitchFrameSource` 里先做 `cudaMalloc` 再建 hwdevice：
  `av_hwdevice_ctx_create(AV_CUDA_USE_PRIMARY_CONTEXT)` 要设置 primary context 的
  flags，runtime API 已激活它就会返回 -129（"Primary context already active with
  incompatible flags"）。构造顺序已按此排好，别调换。
