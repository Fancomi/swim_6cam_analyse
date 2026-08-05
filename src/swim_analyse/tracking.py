"""
拼接画布上的逐帧目标检测 + 跨帧跟踪。

YOLO 只做检测（不用 model.track()），跟踪用纯 IoU 逐帧贪心匹配的
SimpleTracker：数据是俯视固定机位的泳道，目标运动平滑、无遮挡换位，
IoU 匹配已足够，且行为完全可预期、没有外部跟踪器的隐藏状态。
"""

import numpy as np


def _iou_matrix(a, b):
    """a:(N,4) b:(M,4) -> IoU 矩阵 (N,M)。空输入返回形状正确的空矩阵。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    inter = (np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])) *
             np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])))
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)


def filter_contained_boxes(boxes, thresh=0.7):
    """
    去掉"几乎完全被另一个框包住"的重复检测（保留置信度高的那个）。

    用 交集/自身面积 而不是 IoU：同一个游泳者常被同时框出一大一小两个框，
    这种情况下 IoU 偏低会被 NMS 漏掉，而包含率接近 1。

    boxes: list of (x1,y1,x2,y2,conf)。
    """
    n = len(boxes)
    if n <= 1:
        return boxes
    arr = np.array([b[:4] for b in boxes], float)
    conf = np.array([b[4] for b in boxes], float)
    a = arr[:, None, :]
    b = arr[None, :, :]
    inter = (np.maximum(0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])) *
             np.maximum(0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])))
    self_area = ((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1]))[:, None]
    contained = inter / np.maximum(self_area, 1e-6)          # contained[i,j]: i 被 j 包住的比例

    keep = np.ones(n, bool)
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
    return [box for box, k in zip(boxes, keep) if k]


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


class SwimmerTracker:
    """YOLO 检测 + SimpleTracker 的封装。必须按视频真实顺序逐帧调用 update()。"""

    def __init__(self, model_path, device="0", conf=0.25, iou=0.3,
                 containment_thresh=0.7, track_iou=0.3, track_max_lost=30):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(f"cuda:{device}" if str(device).isdigit() else device)
        self.conf, self.iou = conf, iou
        self.containment_thresh = containment_thresh
        self.tracker = SimpleTracker(track_iou, track_max_lost)

    def update(self, frame):
        """输入画布帧，返回 [(tid, x1, y1, x2, y2, conf)]（画布像素坐标）。"""
        result = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            boxes = [(*map(float, xy), float(c)) for xy, c in zip(xyxy, confs)]
        return self.tracker.update(filter_contained_boxes(boxes, self.containment_thresh))
