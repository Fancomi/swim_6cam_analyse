#!/usr/bin/env bash
# 运行游泳分析流水线。
#
#   bash run.sh                        # 用默认数据跑完整流程
#   bash run.sh --max-frames 300       # 只跑前 300 帧（快速验证）
#   bash run.sh --help                 # 全部可选参数
#
# 任何 swim-analyse 的参数都可以透传，且会覆盖这里的默认值（argparse 取最后一次）。
# 也可以用环境变量覆盖：
#   DATA_DIR / CANVAS / OUTPUT_DIR     输入输出路径
#   STROKE_TYPE / SIGNAL               泳姿 / 划水信号
#   DRAW_KEYPOINTS= SIGNAL_PLOTS=      置空则关闭（默认开启）
#   DEBUG_CROPS=1                      每人导出原视角裁剪+骨架小视频
#
# 数据布局约定（DATA_DIR 下）：
#   merged_3000f.mp4              六路拼接后的全景视频
#   cam1.mp4 ... cam6.mp4         六路原始相机视频
#
# 注意 --camera-videos 的顺序必须与 configs/pool_mesh.json 里 meshes 数组的
# 顺序严格对应，即 cam4 cam3 cam2 cam5 cam6 cam1（mesh 的 texture_basename
# 依次为 camera_3/2/1/4/5/6，与文件名并非同序，此顺序由标定时确定，勿改）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${VENV:-$ROOT/.venv}/bin/python"
[[ -x "$PYTHON" ]] || { echo "未找到 $PYTHON，请先执行 bash install.sh" >&2; exit 1; }

DATA_DIR="${DATA_DIR:-$ROOT/data/20260629}"
CANVAS="${CANVAS:-$DATA_DIR/merged_3000f.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/$(basename "${CANVAS%.*}")}"

CAMERAS=()
for i in 4 3 2 5 6 1; do CAMERAS+=("$DATA_DIR/cam$i.mp4"); done

# 开关型默认值：置空即关闭，例如 DRAW_KEYPOINTS= bash run.sh
OPTIONS=()
[[ -n "${DRAW_KEYPOINTS-1}" ]] && OPTIONS+=(--draw-keypoints)
[[ -n "${SIGNAL_PLOTS-1}" ]] && OPTIONS+=(--signal-plots)
[[ -n "${DEBUG_CROPS-}" ]] && OPTIONS+=(--debug-crops)

exec "$PYTHON" -m swim_analyse.pipeline \
  --canvas-video "$CANVAS" \
  --camera-videos "${CAMERAS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --stroke-type "${STROKE_TYPE:-freestyle}" \
  --signal "${SIGNAL:-elbow_angle}" \
  "${OPTIONS[@]}" \
  "$@"
