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
