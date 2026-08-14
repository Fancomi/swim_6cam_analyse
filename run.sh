#!/usr/bin/env bash
# Python 参考链路入口（离线批处理，Plan A/B/C）。
# 想要实时/最快，用 C++ 链路：双击 run_preview.bat，或见 cpp/README.md。
#
#   bash run.sh                       # Plan C（推荐），默认数据
#   bash run.sh C --max-frames 300    # 先跑 300 帧确认链路
#   bash run.sh B                     # Plan B
#   bash run.sh A                     # Plan A（需六路原相机视频）
#   bash run.sh --help                # 透传给 CLI 的全部参数
#
# 三套方案只在「如何从画布得到每人的框与关键点」上不同，下游共用，指标可直接对比：
#   A  画布 detect + 原相机 RTMPose + mesh 反投影（交接原版）
#      需六路原相机视频；关键点分辨率最高，但最慢
#   B  画布 yolo26-pose 一体（单次前向出框+点）
#      只需画布视频；人数多时最省，但检测与关键点精度均最低
#   C  画布 detect + 画布 RTMPose（两阶段）
#      只需画布视频；检测召回与关键点精度均最好，速度与 B 持平
# 实测指标对比见 README.md「三套关键点方案」。
#
# 环境变量覆盖：
#   DATA_DIR      数据目录（默认 data/20260730；Plan A 用 data/20260629）
#   CANVAS        画布视频（默认 $DATA_DIR/merged_3000f.mp4）
#   OUTPUT_DIR    输出目录（默认 output/<视频名>_plan<X>）
#   STROKE_TYPE   泳姿（freestyle/backstroke/butterfly/breaststroke）
#   SIGNAL        划水信号（elbow_angle/wrist_x_head）
#   DRAW_KEYPOINTS= SIGNAL_PLOTS=    置空则关闭（默认开）
#   VENV          venv 位置（默认 .venv）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
# venv 的解释器：Linux 在 bin/python，Windows 在 Scripts/python.exe
PYTHON="$VENV/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$VENV/Scripts/python.exe"
[[ -x "$PYTHON" ]] || { echo "[run] 未找到 $VENV 下的 Python，请先执行 bash install.sh" >&2; exit 1; }
# Windows 的默认 stdout 是 GBK，中文日志会乱码
export PYTHONUTF8=1

# --help 不需要指定 plan；plan 缺省为 C
PLAN=C
case "${1:-}" in
  A|B|C) PLAN="$1"; shift ;;
  -h|--help) exec "$PYTHON" -m swim_analyse.cli --help ;;
  ""|-*) ;;                       # 无 plan 或直接给参数：走默认 Plan C
  *) echo "[run] 用法: bash run.sh [A|B|C] [透传参数...]（见 bash run.sh --help）" >&2; exit 1 ;;
esac

# Plan A 要六路原相机视频，只有旧数据集 20260629 有；B/C 只要画布，
# 默认对齐 C++ 链路的 20260730，两条链路的数字才可比。
DEFAULT_DATA="data/$([[ $PLAN == A ]] && echo 20260629 || echo 20260730)"
DATA_DIR="${DATA_DIR:-$ROOT/$DEFAULT_DATA}"
CANVAS="${CANVAS:-$DATA_DIR/merged_3000f.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/$(basename "${CANVAS%.*}")_plan$PLAN}"

need() { [[ -e "$1" ]] || { echo "[run] 缺少$2: $1" >&2; exit 1; }; }
need "$CANVAS" "画布视频（用 CANVAS=... 指定）"

OPTS=(--plan "$PLAN" --canvas-video "$CANVAS" --output-dir "$OUTPUT_DIR"
      --stroke-type "${STROKE_TYPE:-freestyle}" --signal "${SIGNAL:-elbow_angle}")
[[ -n "${DRAW_KEYPOINTS-1}" ]] && OPTS+=(--draw-keypoints)
[[ -n "${SIGNAL_PLOTS-1}" ]] && OPTS+=(--signal-plots)

DETECT="$ROOT/weights/yolo_swim_detect.pt"
case "$PLAN" in
  A)
    # --camera-videos 的顺序必须与 configs/pool_mesh.json 的 meshes 数组一致，
    # 即 cam4 cam3 cam2 cam5 cam6 cam1（由标定时确定，勿改）
    CAMS=(); for i in 4 3 2 5 6 1; do
      need "$DATA_DIR/cam$i.mp4" "原相机视频（Plan A 专用，用 DATA_DIR=... 指定）"
      CAMS+=("$DATA_DIR/cam$i.mp4")
    done
    need "$DETECT" "检测权重"; need "$ROOT/weights/rtmpose_m_swim.pth" "关键点权重"
    OPTS+=(--camera-videos "${CAMS[@]}"
           --yolo-model "$DETECT"
           --pose-config "$ROOT/configs/rtmpose-m_swim-256x192.py"
           --pose-checkpoint "$ROOT/weights/rtmpose_m_swim.pth")
    ;;
  B)
    need "$ROOT/weights/plans/planB_yolo26x_pose_canvas.pt" "一体权重"
    OPTS+=(--pose-model "$ROOT/weights/plans/planB_yolo26x_pose_canvas.pt")
    ;;
  C)
    need "$DETECT" "检测权重"
    need "$ROOT/weights/plans/planC_rtmpose_m_canvas.pth" "关键点权重"
    OPTS+=(--yolo-model "$DETECT"
           --pose-config "$ROOT/configs/rtmpose-m_canvas-192x256.py"
           --pose-checkpoint "$ROOT/weights/plans/planC_rtmpose_m_canvas.pth")
    ;;
esac

echo "[run] Plan $PLAN  输入 $CANVAS  输出 $OUTPUT_DIR"
exec "$PYTHON" -m swim_analyse.cli "${OPTS[@]}" "$@"
