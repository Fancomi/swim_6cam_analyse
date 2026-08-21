#!/usr/bin/env bash
# 一条命令验证仓库还好用。默认只跑秒级检查，加 --full 才跑 GPU 冒烟。
#
#   bash scripts/test.sh          # 单元测试 + 入口脚本语法（无需 GPU/权重，约 2 秒）
#   bash scripts/test.sh --full   # 再加 Python Plan C、C++ 画布、C++ 六路拼接各 30 帧
#
# 冒烟的产物全部落在 output/_smoke_*，跑完自动删除。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ 的上一级 = 仓库根
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
for f in scripts/run.sh scripts/install.sh scripts/test.sh scripts/cams.sh; do
  run "语法 $(basename "$f")" bash -n "$f"
done

# .bat 只能是「无 BOM + CRLF + 纯 ASCII」：cmd.exe 把 BOM 当第一条命令的一部分
# （`@echo off` 变乱码），中文又按系统 ANSI 代码页解，换台机器就是乱码。
# .ps1 反过来：**必须带 BOM**，Windows PowerShell 5.1 对无 BOM 文件按系统 ANSI
# 代码页解码，中文串的多字节序列被拆成全角字符、把引号吞进字符串，报的是
# 「Missing closing '}'」这种完全指错地方的语法错。两者都得是 CRLF。
# 破坏这些不会在本机报错，只在交付机上现身，所以在这里挡住。
# dist.ps1 生成的 .bat（run_*.bat / update.bat）同 .bat 规则，见其中的 Write-Gen。
# 行尾用「CR 数 == LF 数」判定：msys 的 grep 会先剥掉行尾 CR，`grep -v $'\r$'`
# 因此对 CRLF 文件也全命中，测不出东西。
enc_bad=""
has_bom() { [[ "$(head -c 3 "$1" | od -An -tx1 | tr -d ' ')" == efbbbf ]]; }
is_crlf() { [[ "$(tr -dc '\r' < "$1" | wc -c)" == "$(tr -dc '\n' < "$1" | wc -c)" ]]; }
for f in scripts/*.bat; do
  b="$(basename "$f")"
  has_bom "$f" && enc_bad+=" $b:BOM"
  LC_ALL=C grep -q $'[\x80-\xff]' "$f" && enc_bad+=" $b:非ASCII"
  is_crlf "$f" || enc_bad+=" $b:非CRLF"
done
for f in scripts/*.ps1; do
  b="$(basename "$f")"
  has_bom "$f" || enc_bad+=" $b:缺BOM"
  is_crlf "$f" || enc_bad+=" $b:非CRLF"
done
[[ -z "$enc_bad" ]] && ok "脚本编码（bat 无 BOM+ASCII / ps1 带 BOM / 均 CRLF）" \
                    || bad "脚本编码违规:$enc_bad"

# .ps1 语法：只信 Windows PowerShell 5.1 的 ParseFile —— pwsh 7 与 mac 上的
# PowerShell 无 BOM 默认按 UTF-8 解，会让上面那类乱码问题在本机静默通过。
if command -v powershell.exe >/dev/null; then
  for f in scripts/*.ps1; do
    w="$(cygpath -w "$ROOT/$f" 2>/dev/null || echo "$ROOT/$f")"
    run "语法 $(basename "$f")" powershell.exe -NoProfile -Command \
      "\$e=\$null; [System.Management.Automation.Language.Parser]::ParseFile('$w',[ref]\$null,[ref]\$e)|Out-Null; if(\$e){\$e;exit 1}"
  done
else
  skip "语法 ps1" "非 Windows，没有 powershell.exe"
fi

# ── 2. 单元测试（纯逻辑，无需 GPU 与数据）───────────────────────────────────
if [[ -x "$PYTHON" ]]; then
  run "单元测试 pytest tests" "$PYTHON" -m pytest tests -q
else
  skip "单元测试" "未找到 .venv，先跑 bash scripts/install.sh"
fi

# ── 3. GPU 冒烟（--full）────────────────────────────────────────────────────
CANVAS="${CANVAS:-$ROOT/data/20260730/merged_3000f.mp4}"
if [[ $FULL -eq 0 ]]; then
  echo "（加 --full 可跑 30 帧的 Python + C++（画布与六路拼接）真实冒烟）"
elif [[ ! -f "$CANVAS" ]]; then
  skip "GPU 冒烟" "缺少画布视频 $CANVAS（用 CANVAS=... 指定）"
else
  SMOKE="$ROOT/output/_smoke_py"
  if [[ -x "$PYTHON" && -f weights/plans/planC_rtmpose_m_canvas.pth ]]; then
    rm -rf "$SMOKE"
    run "Python Plan C 30 帧" env CANVAS="$CANVAS" OUTPUT_DIR="$SMOKE" \
      bash scripts/run.sh C --max-frames 30
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
    # Windows 只能靠 PATH。注意 Git Bash 的 PATH 以 ':' 分隔，"D:/x" 会被
    # 拆成 "D" 和 "/x" 两项而失效，所以要先转成 /d/x 形式。
    TRT_LIB="${SWIM_TRT_LIB:-D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33/lib}"
    if [[ -d "$TRT_LIB" ]]; then
      command -v cygpath >/dev/null && TRT_LIB="$(cygpath -u "$TRT_LIB")"
      export PATH="$TRT_LIB:$PATH"
    fi
    JSON="$ROOT/output/_smoke_cpp.json"
    run "C++ 30 帧" "$EXE" --input "$CANVAS" --models cpp/models \
        --max-frames 30 --json "$JSON"
    [[ -s "$JSON" ]] && ok "C++ 产出 json" || bad "C++ 未产出 json"
    rm -f "$JSON"

    # 六路上游拼接（--cam-dir）：要有六路原片与烘好的表才跑
    CAMS="${CAM_DIR:-D:/WindowsProject/workspace/SWIM/20260730-4k-raw}"
    if [[ -d "$CAMS" && -f cpp/models/stitch.lut ]]; then
      JSON="$ROOT/output/_smoke_stitch.json"
      run "C++ 六路拼接 30 帧" "$EXE" --cam-dir "$CAMS" --models cpp/models \
          --max-frames 30 --json "$JSON"
      [[ -s "$JSON" ]] && ok "六路拼接产出 json" || bad "六路拼接未产出 json"
      rm -f "$JSON"
    else
      skip "C++ 六路拼接" "缺少六路原片 $CAMS（用 CAM_DIR=... 指定）或 cpp/models/stitch.lut"
    fi
  else
    skip "C++ 冒烟" "未构建（双击 scripts/build.bat）或缺少 cpp/models/*.onnx"
  fi
fi

echo
printf '通过 %d，失败 %d\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
