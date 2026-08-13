#!/usr/bin/env bash
# 一键安装运行环境。默认在项目下创建 .venv，不污染系统 Python。
#
#   bash install.sh              # 安装 + 自检
#   bash install.sh --skip-test  # 只安装
#
# 环境要求：Linux + NVIDIA GPU（CUDA 驱动 ≥ 12.1）+ Python 3.10。
#
# 版本锁定的原因：mmcv 的 CUDA 算子只有预编译 wheel 可用（源码编译需数十分钟
# 且易失败），而 OpenMMLab 官方只为特定 torch/CUDA 组合发布 wheel。
# cu121/torch2.1.0 是同时提供 mmcv 2.1.0 wheel、且被 mmpose 1.3.1 与
# mmdet 3.2.0 共同支持（两者都要求 mmcv<2.2.0）的组合，因此整条链锁在这里。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PY_VERSION=3.10

TORCH=2.1.0
TORCHVISION=0.16.0
NUMPY=1.26.4          # 须与 requirements.txt 中的 numpy 版本一致
SETUPTOOLS="<81"
MMCV=2.1.0
MMPOSE=1.3.1
# 走官方索引页而不是写死 wheel 文件名，由 pip 按平台/解释器挑对应 wheel
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch${TORCH}/index.html"

log() { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[install] %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. 前置检查 ──────────────────────────────────────────────────────────────
# Windows/Git Bash 下 nvidia-smi 能通过，但 venv 路径、python3 存根、numpy 与
# setuptools 的安装顺序都不一样，不是换个路径能覆盖的，故直接早退
case "$(uname -s)" in
  Linux) ;;
  *) die "install.sh 仅支持 Linux；Windows 见 docs/迁移交接.md §4" ;;
esac
command -v nvidia-smi >/dev/null || die "未找到 nvidia-smi，本项目需要 NVIDIA GPU"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ── 2. 虚拟环境（优先 uv，缺失则回退 venv/virtualenv）───────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
  log "创建虚拟环境 $VENV"
  if command -v uv >/dev/null; then
    uv venv "$VENV" --python "$PY_VERSION"
  elif python3 -m venv "$VENV" 2>/dev/null; then
    :
  elif command -v virtualenv >/dev/null; then
    virtualenv -p "python$PY_VERSION" "$VENV"
  else
    die "无法创建虚拟环境，请先安装 uv 或 python${PY_VERSION}-venv"
  fi
fi
PYTHON="$VENV/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,10), sys.version' \
  || die "虚拟环境 Python 版本必须是 3.10（mmcv wheel 只提供 cp310）"

# uv 装得快得多，但它不会在 venv 里放 pip；两条路径都用 pip_install 统一入口
if command -v uv >/dev/null; then
  pip_install() { uv pip install --python "$PYTHON" "$@"; }
else
  "$PYTHON" -m ensurepip -q 2>/dev/null || true
  pip_install() { "$PYTHON" -m pip install "$@"; }
fi

# ── 3. 依赖 ─────────────────────────────────────────────────────────────────
log "安装 torch $TORCH (cu121)"
# 同批安装的原因：
#   numpy      torch 的 C 扩展基于 numpy1 ABI 编译，被后续依赖升到 numpy2 会让
#              import torch 报 "_ARRAY_API not found"
#   setuptools mmengine.get_installed_path 运行时 import pkg_resources，而
#              uv 创建的 venv 默认不含 setuptools；pkg_resources 在
#              setuptools 81 中被移除，故上限锁在 81 以下
pip_install -q "torch==$TORCH" "torchvision==$TORCHVISION" "numpy==$NUMPY" \
  "setuptools$SETUPTOOLS"
"$PYTHON" - <<'EOF' || die "torch 无法使用 GPU，请检查驱动与 CUDA 版本"
import sys, torch
print(f"[install] torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
sys.exit(0 if torch.cuda.is_available() else 1)
EOF

log "安装 mmcv $MMCV (预编译 wheel)"
pip_install -q "mmcv==$MMCV" -f "$MMCV_INDEX" \
  || die "mmcv wheel 下载失败。若在内网请先设置代理：export https_proxy=... http_proxy=..."

log "安装其余依赖"
pip_install -q -r "$ROOT/requirements.txt"
# mmpose 用 --no-deps：它声明依赖 chumpy，而 chumpy 打包方式陈旧（构建期缺
# setuptools 会失败），且 import 时用了 numpy>=1.24 已删除的 numpy.bool 等别名。
# 本项目只做 2D 关键点，不涉及 chumpy 服务的 SMPL 网格任务，requirements.txt
# 里已单独列出 mmpose 其余的运行时依赖。
pip_install -q --no-deps "mmpose==$MMPOSE"

log "安装本项目 (可编辑模式)"
pip_install -q --no-deps -e "$ROOT"

# ── 4. 自检 ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--skip-test" ]]; then
  log "已跳过自检。"
else
  log "依赖自检"
  "$PYTHON" - <<'EOF'
import mmcv, mmdet, mmengine, mmpose, torch, ultralytics, cv2, numpy, scipy, matplotlib
from mmcv.ops import nms                      # 验证 CUDA 算子已编译进 wheel
from mmpose.apis import init_model, inference_topdown
from ultralytics import YOLO
import swim_analyse.cli, swim_analyse.plans
print(f"[install] mmcv {mmcv.__version__}  mmengine {mmengine.__version__}  "
      f"mmpose {mmpose.__version__}  mmdet {mmdet.__version__}")
print(f"[install] ultralytics {ultralytics.__version__}  opencv {cv2.__version__}  "
      f"numpy {numpy.__version__}")
EOF

  log "单元测试"
  "$PYTHON" -m pytest "$ROOT/tests" -q

  log "模型加载检查"
  for f in weights/yolo_swim_detect.pt weights/rtmpose_m_swim.pth configs/pool_mesh.json; do
    [[ -f "$ROOT/$f" ]] || die "缺少文件 $f"
  done
  "$PYTHON" - <<EOF
from mmpose.apis import init_model
from ultralytics import YOLO
m = init_model("$ROOT/configs/rtmpose-m_swim-256x192.py",
               "$ROOT/weights/rtmpose_m_swim.pth", device="cuda:0")
print(f"[install] RTMPose 就绪：{m.dataset_meta['num_keypoints']} 个关键点")
YOLO("$ROOT/weights/yolo_swim_detect.pt")
print("[install] YOLO 就绪")
EOF
fi

log "完成。运行分析：bash run.sh --help"
