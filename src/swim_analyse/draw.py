"""拼接画布视频的标注绘制：检测框、ID/划水数/速度标签、关键点骨架。"""

import hashlib

import cv2
import numpy as np

from .pose import KPT_COLORS, KPT_THR, SKELETON

FONT = cv2.FONT_HERSHEY_SIMPLEX


def id_color(track_id):
    """按 id 生成稳定配色（BGR），便于肉眼区分不同目标。"""
    h = hashlib.md5(str(track_id).encode()).digest()
    return int(h[2]), int(h[1]), int(h[0])


def draw_skeleton(img, kpts, scores, thr=KPT_THR, radius=3):
    """
    画单个目标的骨架。置信度不足或坐标为 NaN 的点跳过，连线两端任一不可用
    就整段不画——插值点置信度为 0，因此长距离缺失区间不会被连出贯穿画面的
    错误线段。
    """
    kpts = np.asarray(kpts, float)
    ok = (np.asarray(scores) >= thr) & np.isfinite(kpts).all(axis=1)
    # NaN 无法转整型，先置零；这些点的 ok 为 False，不会被画出
    pts = np.where(ok[:, None], kpts, 0.0).astype(int)
    for (i, j), color in SKELETON:
        if ok[i] and ok[j]:
            cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(img, tuple(pts[i]), radius, KPT_COLORS[i], -1, cv2.LINE_AA)
    return img


def draw_label(img, box, text, color, scale=1.2, thickness=3):
    """在框左上角画带底色的文字标签。"""
    x1, y1 = int(box[0]), int(box[1])
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, ty), FONT, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def crop_with_padding(frame, box, ratio=0.5):
    """按 ratio 向四周扩展后裁剪，返回 (裁剪图, 左上角偏移 (x, y))。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * ratio, (y2 - y1) * ratio
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, (0, 0)
    return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1)


class TrackCropWriter:
    """
    调试用：为每个目标写一个"原视角裁剪 + 骨架"的小视频。

    每个目标的裁剪尺寸随 bbox 变化，而视频编码要求固定分辨率，因此以该目标
    首帧的尺寸为基准，后续帧缩放对齐。
    """

    def __init__(self, out_dir, fps=10.0):
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.fps = fps
        self._writers = {}
        self._sizes = {}

    def write(self, track_id, frame, box, kpts, scores, thr=KPT_THR):
        import os
        crop, (ox, oy) = crop_with_padding(frame, box)
        if crop is None:
            return
        draw_skeleton(crop, np.asarray(kpts) - (ox, oy), scores, thr)
        if track_id not in self._writers:
            h, w = crop.shape[:2]
            self._sizes[track_id] = (w, h)
            self._writers[track_id] = cv2.VideoWriter(
                os.path.join(self.out_dir, f"track_{track_id}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        size = self._sizes[track_id]
        self._writers[track_id].write(
            crop if crop.shape[1::-1] == size else cv2.resize(crop, size))

    def close(self):
        for w in self._writers.values():
            w.release()
        if self._writers:
            print(f"[Debug] 已写出 {len(self._writers)} 个目标的裁剪视频 -> {self.out_dir}")
        self._writers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
