#!/usr/bin/env bash
# 一条命令验证仓库还好用。三档由快到慢，各自回答一个不同的问题。
#
#   bash scripts/test.sh            # 能跑吗：单元测试 + 入口脚本语法（无 GPU，约 2 秒）
#   bash scripts/test.sh --full     # 跑得通吗：Python Plan C、C++ 画布、C++ 六路各 30 帧
#                                   #           外加 rot180 逐字节判据与 web 看板六路由
#   bash scripts/test.sh --baseline # 数字没变吗：四种组合各 3000 帧核对基线（约 5 分钟）
#
# 冒烟的产物全部落在 output/_smoke_*，跑完自动删除。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ 的上一级 = 仓库根
cd "$ROOT"
export PYTHONUTF8=1
FULL=0; BASE=0
case "${1:-}" in --full) FULL=1;; --baseline) BASE=1;; esac

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

# ── 3. GPU 档：--full「跑得通吗」30 帧；--baseline「数字没变吗」3000 帧 ──────
CANVAS="${CANVAS:-$ROOT/data/20260730/merged_3000f.mp4}"
CAMS="${CAM_DIR:-D:/WindowsProject/workspace/SWIM/20260730-4k-raw}"
EXE="$ROOT/cpp/build/Release/swim_analyse.exe"
[[ -x "$EXE" ]] || EXE="$ROOT/cpp/build/swim_analyse"

# TRT 的动态库不在默认搜索路径里；Linux 用 RPATH（CMake 已写入），Windows 只能
# 靠 PATH。注意 Git Bash 的 PATH 以 ':' 分隔，"D:/x" 会被拆成 "D" 和 "/x" 两项
# 而失效，所以要先转成 /d/x 形式。
trt_path() {
  local lib="${SWIM_TRT_LIB:-D:/WindowsProject/workspace/TRT/TensorRT-10.11.0.33/lib}"
  [[ -d "$lib" ]] || return 0
  command -v cygpath >/dev/null && lib="$(cygpath -u "$lib")"
  export PATH="$lib:$PATH"
}
# 交给 Windows 进程（exe / .venv 的 python.exe）的路径必须是 Windows 形式：
# msys 的 /tmp/xxx 会被当成相对路径，静默写不出文件而不报错。
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

# 全量核对：把 [Summary] 的三元组（人次 / track / 划水）与基线比。四种组合互不
# 可比，各自只对自己那一行，口径见 CLAUDE.md「怎么验证一处改动」。
baseline() { local name="$1" want="$2"; shift 2
  local log; log="$(mktemp)"
  if ! "$@" >"$log" 2>&1; then bad "$name（进程失败）"; tail -15 "$log"; rm -f "$log"; return; fi
  # 锚定 ^[Summary]：耗时表的表头也含「累计(s)」，不锚就会被它清空；
  # ghost 那行同样含「人次」但不含「累计」，两条都得排除掉。
  local got
  got="$(awk '/^\[Summary\].*累计/{n=$5;t=$7} /^\[Summary\].*划水合计/{s=$3}
              END{printf "%s %s %s",n,t,s}' "$log")"
  local ms; ms="$(awk '/ms\/帧/{for(i=1;i<=NF;i++) if($i=="ms/帧"){print $(i-1)" ms";exit}}' "$log")"
  rm -f "$log"
  [[ "$got" == "$want" ]] && ok "$name  $got  ${ms:-—}" \
                          || bad "$name 基线不符：期望「$want」实得「$got」"
}

# rot180 的正确性判据：转过的首帧 == 不转的首帧再 [::-1,::-1]，**逐字节**。
# 它比任何统计量都灵敏，且与推理无关 —— detect 对定向不是旋转等变的
# （letterbox + 卷积），统计三元组只能各自对基线，不能相互 diff。
rot_oracle() { local name="$1"; shift              # "$@" = 取一帧的命令前缀
  local d; d="$(mktemp -d)"
  local a b; a="$(win "$d/up.png")"; b="$(win "$d/rot.png")"
  if "$@" --max-frames 1 --dump-canvas "$a" >/dev/null 2>&1 &&
     "$@" --max-frames 1 --rot180 --dump-canvas "$b" >/dev/null 2>&1 &&
     "$PYTHON" -c "
import sys, cv2, numpy as np
a, b = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
sys.exit(0 if a is not None and b is not None and np.array_equal(a[::-1, ::-1], b) else 1)
" "$a" "$b"; then ok "$name"; else bad "$name"; fi
  rm -rf "$d"
}

# 看板：起一次服务，把六个路由都摸一遍。用非默认端口，免得撞上手工开着的那个。
# 判据刻意只到「协议对不对」：/meta 是合法 JSON、SSE 首条含帧号、MJPEG 首段有
# JPEG 魔数。像素与统计值另有基线管，这里只挡「路由/线程/析构」类回归。
web_smoke() { local name="$1"; shift
  command -v curl >/dev/null || { skip "$name" "没有 curl"; return; }
  local port=18099 log body; log="$(mktemp)"; body="$(mktemp)"
  "$@" --web --web-port $port --no-browser >"$log" 2>&1 &
  local pid=$!
  # engine 已缓存也要几秒起 TRT，轮询到监听为止
  local i
  for i in $(seq 40); do
    curl -s -m 1 "http://127.0.0.1:$port/meta" >/dev/null 2>&1 && break
    sleep 1
  done
  # 一律先落盘再判：SSE 与 MJPEG 是无限流，curl 只能靠 -m 超时收尾、必然以 28
  # 退出，而本脚本开了 pipefail —— `curl | grep` 会让这个退出码盖掉 grep 的结论。
  local bad_web=""
  probe() { curl -s -m "$1" -o "$body" "http://127.0.0.1:$port$2" || true; }
  probe 2 /meta
  "$PYTHON" -c "
import json,sys
m=json.load(sys.stdin)
assert m['w']>0 and m['h']>0 and len(m['skel'])==17 and m['nk']==17, m
" <"$body" >/dev/null 2>&1 || bad_web+=" /meta"
  probe 2 /
  grep -q '<!doctype html>' "$body" || bad_web+=" /"
  probe 2 /stats
  grep -q '^data: {"i":' "$body" || bad_web+=" /stats"
  probe 2 /canvas.mjpg
  grep -qa 'Content-Type: image/jpeg' "$body" || bad_web+=" /canvas.mjpg"
  [[ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/select?id=0")" == 204 ]] \
    || bad_web+=" /select"
  [[ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/nope")" == 404 ]] \
    || bad_web+=" 404"
  wait $pid                       # 帧数跑完自己退出；析构卡死会在这里挂住
  [[ -z "$bad_web" ]] && ok "$name" || { bad "$name 路由异常:$bad_web"; tail -8 "$log"; }
  rm -f "$log" "$body"
}

if [[ $FULL -eq 0 && $BASE -eq 0 ]]; then
  echo "（--full 跑 30 帧真实冒烟；--baseline 跑 3000 帧核对基线三元组，约 5 分钟）"
elif [[ ! -f "$CANVAS" ]]; then
  skip "GPU 档" "缺少画布视频 $CANVAS（用 CANVAS=... 指定）"
else
  trt_path
  CPP=0;    [[ -x "$EXE" && -f cpp/models/detect.onnx ]] && CPP=1
  STITCH=0; [[ $CPP -eq 1 && -d "$CAMS" && -f cpp/models/stitch.lut ]] && STITCH=1
  ORACLE=0; [[ $CPP -eq 1 && -x "$PYTHON" ]] && ORACLE=1
  [[ $CPP -eq 1 ]] || skip "C++ 档" "未构建（双击 scripts/build.bat）或缺少 cpp/models/*.onnx"
  [[ $STITCH -eq 1 || $CPP -eq 0 ]] || \
    skip "六路拼接" "缺少六路原片 $CAMS（用 CAM_DIR=... 指定）或 cpp/models/stitch.lut"

  if [[ $FULL -eq 1 ]]; then
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

    for t in "画布|--input|$CANVAS|$CPP" "六路拼接|--cam-dir|$CAMS|$STITCH"; do
      IFS='|' read -r nm flag src on <<<"$t"    # 不能用 ':' 分隔：CAM_DIR 是 D:/… 带盘符冒号
      [[ $on -eq 1 ]] || continue
      JSON="$ROOT/output/_smoke_${flag#--}.json"
      run "C++ $nm 30 帧" "$EXE" "$flag" "$src" --models cpp/models \
          --max-frames 30 --json "$JSON"
      [[ -s "$JSON" ]] && ok "C++ $nm 产出 json" || bad "C++ $nm 未产出 json"
      rm -f "$JSON"
      # 单帧的字节判据：比统计量灵敏，代价只是多一次 engine 加载
      [[ $ORACLE -eq 1 ]] && rot_oracle "rot180 逐字节（$nm）" \
        "$EXE" "$flag" "$src" --models cpp/models
    done

    # 看板只摸一遍（走画布那条路即可：web 层在帧源之上，与拼接无关）。
    # 帧数要够服务活到探完六个路由 —— 六次 curl 的超时合计约 11 s，而画布 60 fps
    # 上下，1200 帧（约 20 s）留足余量；给少了会只有 /meta 赶上，其余全报路由异常。
    [[ $CPP -eq 1 && $ORACLE -eq 1 ]] && web_smoke "web 看板六路由" \
      "$EXE" --input "$CANVAS" --models cpp/models --max-frames 1200

  else
    # 基线三元组（RTX 4080 Laptop / data/20260730 与 20260730-4k-raw / 默认参数）
    [[ $CPP -eq 1 ]] && {
      baseline "画布 3000 帧      " "24107 63 875" \
        "$EXE" --input "$CANVAS" --models cpp/models --max-frames 3000 --show-fps
      baseline "画布 3000 帧 rot180" "24233 90 859" \
        "$EXE" --input "$CANVAS" --models cpp/models --max-frames 3000 --show-fps --rot180
    }
    [[ $STITCH -eq 1 ]] && {
      baseline "六路 3000 帧      " "25078 91 883" \
        "$EXE" --cam-dir "$CAMS" --models cpp/models --max-frames 3000 --show-fps
      baseline "六路 3000 帧 rot180" "25578 119 883" \
        "$EXE" --cam-dir "$CAMS" --models cpp/models --max-frames 3000 --show-fps --rot180
    }
  fi
fi

echo
printf '通过 %d，失败 %d\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
