"""
拼接画布上的逐帧目标检测 + 跨帧跟踪。

YOLO 只做检测（不用 model.track()），跟踪用纯 IoU 逐帧贪心匹配的
SimpleTracker：数据是俯视固定机位的泳道，目标运动平滑、无遮挡换位，
IoU 匹配已足够，且行为完全可预期、没有外部跟踪器的隐藏状态。
"""

import numpy as np


def _inter_area(a, b):
    """两组 xyxy 框的交集面积矩阵：a:(N,4) b:(M,4) -> (N,M)。"""
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    return (np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])) *
            np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])))


def _box_area(boxes):
    """xyxy 框面积：(N,4) -> (N,)。"""
    arr = np.asarray(boxes, float)
    return (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])


def _iou_matrix(a, b):
    """a:(N,4) b:(M,4) -> IoU 矩阵 (N,M)。空输入返回形状正确的空矩阵。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _inter_area(a, b)
    return inter / (_box_area(a)[:, None] + _box_area(b)[None, :] - inter + 1e-6)


def contained_keep_mask(boxes, thresh=0.7):
    """
    "几乎完全被另一个框包住"的重复检测判定，返回保留掩码 (N,) bool。

    用 交集/自身面积 而不是 IoU：同一个游泳者常被同时框出一大一小两个框，
    这种情况下 IoU 偏低会被 NMS 漏掉，而包含率接近 1。重复对里保留置信度高的。

    boxes: list of (x1,y1,x2,y2,conf)。调用方若还持有与 boxes 同序的其他数组
    （如关键点），用本掩码同步索引即可，不必按坐标反查下标。
    """
    n = len(boxes)
    keep = np.ones(n, bool)
    if n <= 1:
        return keep
    arr = np.array([b[:4] for b in boxes], float)
    conf = np.array([b[4] for b in boxes], float)
    # contained[i,j]: i 被 j 包住的比例
    contained = _inter_area(arr, arr) / np.maximum(_box_area(arr), 1e-6)[:, None]

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if contained[i, j] >= thresh:
                if conf[i] > conf[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return keep


def filter_contained_boxes(boxes, thresh=0.7):
    """去重后的框列表（保序）；掩码语义见 contained_keep_mask。"""
    return [box for box, k in zip(boxes, contained_keep_mask(boxes, thresh)) if k]


class SimpleTracker:
    """
    逐帧 IoU 贪心匹配跟踪器。

    每帧：算出"活跃 track 上一帧位置"与"本帧检测框"的 IoU，按从高到低贪心
    配对（达到 iou_thresh 才配）；未配上的检测框开新 id；未配上的 track 记
    一次丢失，连续丢失超过 max_lost 帧才删除（容忍短暂遮挡/漏检）。
    """

    def __init__(self, iou_thresh=0.3, max_lost=30):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks = {}        # tid -> {"bbox": (x1,y1,x2,y2), "lost": int}
        self.next_id = 0

    def update(self, detections):
        """detections: [(x1,y1,x2,y2,conf)] -> [(tid,x1,y1,x2,y2,conf)]，顺序不变。"""
        tids = list(self.tracks)
        ious = _iou_matrix([self.tracks[t]["bbox"] for t in tids],
                           [d[:4] for d in detections])

        assigned = [None] * len(detections)
        matched = set()
        pairs = np.argwhere(ious >= self.iou_thresh)
        for ti, di in sorted(pairs, key=lambda p: -ious[p[0], p[1]]):
            tid = tids[ti]
            if tid in matched or assigned[di] is not None:
                continue
            assigned[di] = tid
            matched.add(tid)

        results = []
        for di, det in enumerate(detections):
            tid = assigned[di]
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[tid] = {"bbox": tuple(det[:4]), "lost": 0}
            results.append((tid, *det))

        for tid in tids:
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        return results
