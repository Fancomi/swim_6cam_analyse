#!/usr/bin/env bash
# 一条命令验证仓库还好用。默认只跑秒级检查，加 --full 才跑 GPU 冒烟。
#
#   bash test.sh              # 单元测试 + 入口脚本语法（无需 GPU/权重，约 2 秒）
#   bash test.sh --full       # 再加 Python Plan C 与 C++ 各 30 帧的真实冒烟
#
# 冒烟的产物全部落在 output/_smoke_*，跑完自动删除。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUTF8=1
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1

PASS=0; FAIL=0
ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '\033[1;31m[fail]\033[0m %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '\033[1;33m[skip]\033[0m %s — %s\n' "$1" "$2"; }
# 跑一条命令：静默，失败时把输出尾部打出来
run() { local name="$1"; shift
  local log; log="$(mktemp)"
  if "$@" >"$log" 2>&1; then ok "$name"; else bad "$name"; tail -15 "$log"; fi
  rm -f "$log"
}

PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$ROOT/.venv/Scripts/python.exe"

# ── 1. 入口脚本语法 ─────────────────────────────────────────────────────────
for f in run.sh install.sh test.sh; do run "语法 $f" bash -n "$f"; done

# ── 2. 单元测试（纯逻辑，无需 GPU 与数据）───────────────────────────────────
if [[ -x "$PYTHON" ]]; then
  run "单元测试 pytest tests" "$PYTHON" -m pytest tests -q
else
  skip "单元测试" "未找到 .venv，先跑 bash install.sh"
fi

# ── 3. GPU 冒烟（--full）────────────────────────────────────────────────────
CANVAS="${CANVAS:-$ROOT/data/20260730/merged_3000f.mp4}"
if [[ $FULL -eq 0 ]]; then
  echo "（加 --full 可跑 30 帧的 Python + C++ 真实冒烟）"
elif [[ ! -f "$CANVAS" ]]; then
  skip "GPU 冒烟" "缺少画布视频 $CANVAS（用 CANVAS=... 指定）"
else
  SMOKE="$ROOT/output/_smoke_py"
  if [[ -x "$PYTHON" && -f weights/plans/planC_rtmpose_m_canvas.pth ]]; then
    rm -rf "$SMOKE"
    run "Python Plan C 30 帧" env CANVAS="$CANVAS" OUTPUT_DIR="$SMOKE" \
      bash run.sh C --max-frames 30
    [[ -f "$SMOKE/result.json" ]] && ok "Plan C 产出 result.json" \
                                  || bad "Plan C 未产出 result.json"
    rm -rf "$SMOKE"
  else
    skip "Python Plan C" "缺少 .venv 或 Plan C 权重"
  fi

  EXE="$ROOT/cpp/build/Release/swim_analyse.exe"
  [[ -x "$EXE" ]] || EXE="$ROOT/cpp/build/swim_analyse"
  if [[ -x "$EXE" && -f cpp/models/detect.onnx ]]; then
    # TRT 的动态库不在默认搜索路径里；Linux 用 RPATH（CMake 已写入），
    # Windows 只能靠 PATH
    TRT_LIB="${SWIM_TRT_LIB:-D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33/lib}"
    [[ -d "$TRT_LIB" ]] && export PATH="$TRT_LIB:$PATH"
    JSON="$ROOT/output/_smoke_cpp.json"
    run "C++ 30 帧" "$EXE" --input "$CANVAS" --models cpp/models \
        --max-frames 30 --json "$JSON"
    [[ -s "$JSON" ]] && ok "C++ 产出 json" || bad "C++ 未产出 json"
    rm -f "$JSON"
  else
    skip "C++ 冒烟" "未构建（双击 build.bat）或缺少 cpp/models/*.onnx"
  fi
fi

echo
printf '通过 %d，失败 %d\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
