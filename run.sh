#!/usr/bin/env bash
# 三套方案的统一运行入口。
#
#   bash run.sh C                     # Plan C（推荐），默认数据
#   bash run.sh A --max-frames 300    # Plan A 跑前 300 帧
#   bash run.sh B                     # Plan B
#   bash run.sh C --help              # 全部可透传参数
#
# Plan 说明（实测对比见 docs/plans.md）：
#   A  画布 detect + 原相机 RTMPose + mesh 反投影（交接原版）
#      需要六路原相机视频；关键点分辨率最高，但最慢
#   B  画布 yolo26-pose 一体（单次前向出框+点）
#      只需画布视频；人数多时最省，但检测与关键点精度均最低
#   C  画布 detect + 画布 RTMPose（两阶段）
#      只需画布视频；检测召回与关键点精度均最好，速度与 B 持平
#
# 环境变量覆盖：
#   DATA_DIR / CANVAS / OUTPUT_DIR   输入输出
#   STROKE_TYPE / SIGNAL             泳姿 / 划水信号
#   DRAW_KEYPOINTS= SIGNAL_PLOTS=    置空则关闭（默认开）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
# venv 的解释器：Linux 在 bin/python，Windows 在 Scripts/python.exe
PYTHON="$VENV/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$VENV/Scripts/python.exe"
[[ -x "$PYTHON" ]] || { echo "[run] 未找到 $VENV 下的 Python，请先执行 bash install.sh" >&2; exit 1; }

PLAN="${1:-C}"
case "$PLAN" in
  A|B|C) shift ;;
  *) echo "[run] 用法: bash run.sh {A|B|C} [其他参数...]" >&2; exit 1 ;;
esac

DATA_DIR="${DATA_DIR:-$ROOT/data/20260629}"
CANVAS="${CANVAS:-$DATA_DIR/merged_3000f.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/$(basename "${CANVAS%.*}")_plan$PLAN}"

OPTS=(--plan "$PLAN" --canvas-video "$CANVAS" --output-dir "$OUTPUT_DIR"
      --stroke-type "${STROKE_TYPE:-freestyle}" --signal "${SIGNAL:-elbow_angle}")
[[ -n "${DRAW_KEYPOINTS-1}" ]] && OPTS+=(--draw-keypoints)
[[ -n "${SIGNAL_PLOTS-1}" ]] && OPTS+=(--signal-plots)

case "$PLAN" in
  A)
    # --camera-videos 的顺序必须与 configs/pool_mesh.json 的 meshes 数组一致，
    # 即 cam4 cam3 cam2 cam5 cam6 cam1（由标定时确定，勿改）
    CAMS=(); for i in 4 3 2 5 6 1; do CAMS+=("$DATA_DIR/cam$i.mp4"); done
    OPTS+=(--camera-videos "${CAMS[@]}"
           --yolo-model "$ROOT/weights/yolo_swim_detect.pt"
           --pose-config "$ROOT/configs/rtmpose-m_swim-256x192.py"
           --pose-checkpoint "$ROOT/weights/rtmpose_m_swim.pth")
    ;;
  B)
    OPTS+=(--pose-model "$ROOT/weights/plans/planB_yolo26x_pose_canvas.pt")
    ;;
  C)
    OPTS+=(--yolo-model "$ROOT/weights/yolo_swim_detect.pt"
           --pose-config "$ROOT/configs/rtmpose-m_canvas-192x256.py"
           --pose-checkpoint "$ROOT/weights/plans/planC_rtmpose_m_canvas.pth")
    ;;
esac

exec "$PYTHON" -m swim_analyse.cli "${OPTS[@]}" "$@"
