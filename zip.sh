#!/usr/bin/env bash
# 打包迁移包：完整代码 + 权重 + 测试数据（不含 git 历史/构建产物/旧数据集）
#
# 用途：在 Linux 上执行，产出 swim_6cam_analyse.zip，拷到 Windows 解压即可继续。
#   Windows 侧下拉 git 拿代码（与 zip 内一致），zip 主要补权重/测试数据/onnx。
#
# 用法：
#   bash zip.sh                # 默认输出到上级目录 swim_6cam_analyse.zip
#   bash zip.sh /path/out.zip  # 指定输出路径
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(dirname "$ROOT")/swim_6cam_analyse.zip}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[zip] 源目录: $ROOT"
echo "[zip] 输出:   $OUT"

# ── 1. 代码：git 管理的内容（等于仓库全量，干净无产物）───────────────
# 用 git 归档而不是 cp -r，避免把 .venv/.git/build 等带进去
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$TMP"
echo "[zip] 代码（git HEAD, $(git -C "$ROOT" rev-parse --short HEAD)）已归档"

# ── 2. 未入库但迁移必需的辅助文件 ─────────────────────────────────────
if [ -f "$ROOT/docs/迁移交接.md" ]; then
  cp "$ROOT/docs/迁移交接.md" "$TMP/docs/"
  echo "[zip] 迁移交接文档已加入"
fi
if [ -f "$ROOT/zip.sh" ]; then
  cp "$ROOT/zip.sh" "$TMP/zip.sh"
  echo "[zip] zip.sh 已加入"
fi

# ── 3. 权重：全部模型（zip 里必须含，Windows 无法下载）────────────────
mkdir -p "$TMP/weights/plans"
for f in weights/yolo_swim_detect.pt weights/rtmpose_m_swim.pth \
         weights/plans/planB_yolo26x_pose_canvas.pt weights/plans/planC_rtmpose_m_canvas.pth; do
  if [ -f "$ROOT/$f" ]; then
    cp "$ROOT/$f" "$TMP/$f"
    echo "[zip] 权重: $f ($(du -h "$ROOT/$f" | cut -f1))"
  else
    echo "[zip] 警告: 缺少 $f"
  fi
done

# ── 4. 测试数据：C++ 链路 canonical 视频（Plan B/C 与 C++ 都用它）─────
if [ -f "$ROOT/data/20260730/merged_3000f.mp4" ]; then
  mkdir -p "$TMP/data/20260730"
  cp "$ROOT/data/20260730/merged_3000f.mp4" "$TMP/data/20260730/"
  echo "[zip] 测试数据: data/20260730/merged_3000f.mp4 ($(du -h "$ROOT/data/20260730/merged_3000f.mp4" | cut -f1))"
else
  echo "[zip] 警告: 缺少 data/20260730/merged_3000f.mp4"
fi

# ── 5. C++ 的 onnx（加速 Windows 首次启动；engine 必须现场重建不带）───
for f in detect.onnx pose.onnx; do
  if [ -f "$ROOT/cpp/models/$f" ]; then
    cp "$ROOT/cpp/models/$f" "$TMP/cpp/models/"
    echo "[zip] onnx: cpp/models/$f"
  fi
done

# ── 6. 打包 ──────────────────────────────────────────────────────────
rm -f "$OUT"
(
  cd "$TMP"
  zip -r -q "$OUT" .
)
echo "[zip] 完成: $OUT"
du -h "$OUT"
echo
echo "=== 包内容预览 ==="
unzip -l "$OUT" | awk 'NR>3 {s+=$1; n++} END {printf "文件数 %d, 解压后约 %.1f GB\n", n, s/1e9}'
