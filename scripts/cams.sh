#!/usr/bin/env bash
# 六路 ZCam 相机：配置 + 探测 + 生成清单，一条命令到位。
#
#   bash scripts/cams.sh probe            # 只探测：谁在线、当前分辨率/帧率/编码
#   bash scripts/cams.sh setup            # 下发 4K30 配置（会先停录制，movfmt 才可写）
#   bash scripts/cams.sh list             # 生成 configs/cameras.txt（喂给 --cam-dir）
#   bash scripts/cams.sh verify           # 逐台真实拉流，确认分辨率/帧率与期望一致
#   bash scripts/cams.sh run [参数...]    # list + 直接起实时预览
#
# 相机地址由「IP 基址 + 尾数」算出，尾数与相机名的对应写在 CAM_MAP 里 ——
# 这个对应关系由标定决定（见 configs/pool_mesh.json 的 meshes 顺序），不是按
# IP 顺序猜的。改现场网段只改 CAM_BASE。
#
# 现场约束（实测于 E2-M4 / 固件 1.0.8，一台相机 192.168.1.199）：
#   - RTSP 只有一个挂载点 `/live_stream`，给出的是 send_stream 选中的那一路，
#     所以「拉子流」不是换 URL 而是把相机的 send_stream 切到 Stream1。
#   - 同一台相机最多约 4 个并发 RTSP 会话，第 5 个起直接失败。六路各一台时无碍，
#     但要避免同时开多个消费者（本脚本 verify 是串行的）。
#   - movfmt（决定 stream0 的分辨率与帧率）在录制中是只读的：查询会看到
#     `"ro":1`，set 返回 `code:-1` 且静默不生效。所以 setup 会先停录制。
#
# 环境变量覆盖：
#   CAM_BASE=192.168.1     网段前缀（现场旧系统用过 192.168.3）
#   CAM_MAP="cam1=199 …"   相机名到 IP 尾数的映射，空格分隔
#   CAM_STREAM=0|1         拉哪一路（0=stream0 主流 4K，1=stream1 子流）
#   CAM_LIST=<路径>        清单输出位置（默认 configs/cameras.txt）
#   CAM_FMT=4KP29.97       主模式（决定 stream0 的分辨率与帧率）
#   CAM_KEEP_REC=1         setup 时不停录制（那样 movfmt 会改不动，仅用于只调码率）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUTF8=1

CAM_BASE="${CAM_BASE:-192.168.1}"
# 默认：本机目前只接了一台 192.168.1.199。全六路到位后把这里补齐，
# 或用 CAM_MAP 覆盖，格式 "cam3=101 cam2=102 …"。
CAM_MAP="${CAM_MAP:-cam1=199 cam2=199 cam3=199 cam4=199 cam5=199 cam6=199}"
CAM_STREAM="${CAM_STREAM:-0}"
CAM_LIST="${CAM_LIST:-$ROOT/configs/cameras.txt}"
CAM_FMT="${CAM_FMT:-4KP29.97}"
# stream1（子流）的目标参数。只有 CAM_STREAM=1 时下发。
SUB_W="${SUB_W:-1920}"; SUB_H="${SUB_H:-1080}"; SUB_FPS="${SUB_FPS:-30}"

FFPROBE="$ROOT/../swim_6cam_4k/third_party/ffmpeg/bin/ffprobe.exe"
command -v ffprobe >/dev/null && FFPROBE=ffprobe

ok()   { printf '\033[1;32m[ok]\033[0m   %s\n' "$1"; }
bad()  { printf '\033[1;31m[fail]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$1"; }

# 相机名列表（按 CAM_MAP 的书写顺序，只用于遍历；实际顺序由 stitch.lut 决定）
cam_names() { for kv in $CAM_MAP; do echo "${kv%%=*}"; done; }
cam_ip()    { for kv in $CAM_MAP; do [[ "${kv%%=*}" == "$1" ]] && echo "$CAM_BASE.${kv##*=}" && return; done; }

# ZCam 的 RTSP 只暴露一条路径，它给出的是「当前 send_stream 选中的那一路」。
# 所以拉子流不是换 URL，而是把相机的 send_stream 切到 Stream1（见 setup）。
cam_url() { echo "rtsp://$(cam_ip "$1")/live_stream"; }

# 带超时的 HTTP：连不上时不要无限期卡住（旧系统脚本踩过这个坑）
http() { curl -s -m 8 "$1"; }

probe_one() {
  local name="$1" ip; ip="$(cam_ip "$name")"
  local info; info="$(http "http://$ip/info" 2>/dev/null)"
  if [[ -z "$info" ]]; then bad "$name ($ip) HTTP 不通"; return 1; fi
  local model; model="$(sed -n 's/.*"model":"\([^"]*\)".*/\1/p' <<<"$info")"
  local fmt; fmt="$(http "http://$ip/ctrl/get?k=movfmt" | sed -n 's/.*"value":"\([^"]*\)".*/\1/p')"
  local send; send="$(http "http://$ip/ctrl/get?k=send_stream" | sed -n 's/.*"value":"\([^"]*\)".*/\1/p')"
  local mode; mode="$(http "http://$ip/ctrl/mode?action=query" | sed -n 's/.*"msg":"\([^"]*\)".*/\1/p')"
  local s0 s1
  s0="$(http "http://$ip/ctrl/stream_setting?action=query&index=stream0")"
  s1="$(http "http://$ip/ctrl/stream_setting?action=query&index=stream1")"
  fmt_stream() {   # width height fps encoder status -> 一行
    local j="$1"
    printf '%sx%s@%s %s %s' \
      "$(sed -n 's/.*"width":\([0-9]*\).*/\1/p' <<<"$j")" \
      "$(sed -n 's/.*"height":\([0-9]*\).*/\1/p' <<<"$j")" \
      "$(sed -n 's/.*"fps":\([0-9]*\).*/\1/p' <<<"$j")" \
      "$(sed -n 's/.*"encoderType":"\([^"]*\)".*/\1/p' <<<"$j")" \
      "$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' <<<"$j")"
  }
  printf '  %-5s %-15s %-9s movfmt=%-10s send=%-8s mode=%s\n' \
    "$name" "$ip" "$model" "$fmt" "$send" "$mode"
  printf '        stream0 %s\n        stream1 %s\n' \
    "$(fmt_stream "$s0")" "$(fmt_stream "$s1")"
  ok "$name 在线"
}

setup_one() {
  local name="$1" ip; ip="$(cam_ip "$name")"
  [[ -z "$(http "http://$ip/info")" ]] && { bad "$name ($ip) HTTP 不通，跳过"; return 1; }

  # movfmt 决定 stream0 的分辨率与帧率，它在录制中是只读的（实测查询到 "ro":1，
  # set 返回 code=-1 且静默不生效）。所以先停录制，改完再由使用方决定要不要恢复。
  local mode; mode="$(http "http://$ip/ctrl/mode?action=query" | sed -n 's/.*"msg":"\([^"]*\)".*/\1/p')"
  if [[ "$mode" == "rec_ing" && -z "${CAM_KEEP_REC:-}" ]]; then
    warn "$name 正在录制，先停录才能改 movfmt（CAM_KEEP_REC=1 可跳过）"
    http "http://$ip/ctrl/rec?action=stop" >/dev/null
    sleep 1
  fi
  http "http://$ip/ctrl/set?movfmt=$CAM_FMT" >/dev/null
  http "http://$ip/ctrl/stream_setting?index=stream0&venc=h264" >/dev/null
  http "http://$ip/ctrl/stream_setting?index=stream0&bitrate=50000000" >/dev/null

  if [[ "$CAM_STREAM" == "1" ]]; then
    # 子流：分辨率不能超过 stream0，且改尺寸前该流必须是 idle ——
    # 正在 streaming 时先把 send_stream 切到 Stream0 让它退下来。
    http "http://$ip/ctrl/set?send_stream=Stream0" >/dev/null
    sleep 1
    http "http://$ip/ctrl/stream_setting?index=stream1&venc=h264" >/dev/null
    http "http://$ip/ctrl/stream_setting?index=stream1&width=$SUB_W&height=$SUB_H" >/dev/null
    http "http://$ip/ctrl/stream_setting?index=stream1&fps=$SUB_FPS" >/dev/null
    http "http://$ip/ctrl/stream_setting?index=stream1&bitrate=8000000" >/dev/null
    http "http://$ip/ctrl/set?send_stream=Stream1" >/dev/null
  else
    http "http://$ip/ctrl/set?send_stream=Stream0" >/dev/null
  fi

  # 回读校验：ZCam 对不合法的组合是静默拒绝（code=-1），不查就会以为设上了
  local got; got="$(http "http://$ip/ctrl/get?k=movfmt" | sed -n 's/.*"value":"\([^"]*\)".*/\1/p')"
  if [[ "$got" == "$CAM_FMT" ]]; then ok "$name movfmt=$got"
  else bad "$name movfmt 仍是 $got（想要 $CAM_FMT）—— 录制停了吗？该值在录制中只读"; fi
}

# 真实拉一次流，确认 LUT 期望的分辨率与我们配的帧率都对得上。
# 串行做：同一台相机最多约 4 个并发 RTSP 会话，六路并行 verify 会互相挤掉。
verify_one() {
  local name="$1" url; url="$(cam_url "$name")"
  local line
  line="$("$FFPROBE" -v error -rtsp_transport tcp -timeout 8000000 \
          -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate \
          -of csv=p=0 "$url" 2>&1 | tail -1)"
  if [[ "$line" != *","*","* ]]; then bad "$name 拉不到流: $line"; return 1; fi
  # LUT 是按 3840x2160 烘的；分辨率不符会在 C++ 侧启动时报错，这里先拦一次
  if [[ "$line" == *",3840,2160,"* ]]; then ok "$name $line"
  else warn "$name $line（LUT 按 3840x2160 烘制，尺寸不符会被拒）"; fi
}

write_list() {
  mkdir -p "$(dirname "$CAM_LIST")"
  {
    echo "# 六路 ZCam 相机清单，由 scripts/cams.sh list 生成"
    echo "# 每行 <相机>=<地址>；相机名必须与 cpp/models/stitch.lut 里的一致"
    echo "# 网段 $CAM_BASE.*  拉流 stream$CAM_STREAM  主模式 $CAM_FMT"
    for n in $(cam_names); do echo "$n=$(cam_url "$n")"; done
  } > "$CAM_LIST"
  ok "写出 $CAM_LIST"
  sed 's/^/    /' "$CAM_LIST"
}

case "${1:-probe}" in
  probe)
    echo "探测 $CAM_BASE.* 上的相机："
    for n in $(cam_names); do probe_one "$n"; done
    ;;
  setup)
    echo "下发配置（stream$CAM_STREAM，主模式 $CAM_FMT）："
    for n in $(cam_names); do setup_one "$n"; done
    echo; echo "回读拉流校验（串行，相机并发会话有限）："
    for n in $(cam_names); do verify_one "$n"; done
    ;;
  verify)
    echo "逐台拉流校验（串行）："
    for n in $(cam_names); do verify_one "$n"; done
    ;;
  list) write_list ;;
  run)
    write_list
    exec "$ROOT/cpp/build/Release/swim_analyse.exe" --cam-dir "$CAM_LIST" \
         --models cpp/models --preview --show-fps "${@:2}"
    ;;
  *)
    echo "用法: bash scripts/cams.sh [probe|setup|verify|list|run]" >&2
    exit 1
    ;;
esac
