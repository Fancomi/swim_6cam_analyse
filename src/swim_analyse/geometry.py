"""
拼接全景画布 <-> 各相机原始像素坐标 的双向几何映射。

mesh JSON 里每个相机是一张由三角形构成的面片：每个三角形同时给出 2D 世界
坐标 pos（米，决定它在拼接画布上的位置）和纹理坐标 uv（决定它在该相机原图
上的位置），一对三角形即确定一个仿射变换。于是：

    canvas ──M──▶ source        (投影：画布上的点/框落在相机原图哪里)
    canvas ◀─M⁻¹── source        (反投影：相机原图上的关键点画回画布)

两个方向都是"给定一个点，找出它属于哪个三角形"，为此各建一个均匀网格索引
（按三角形在该坐标系下的包围盒分桶），把线性扫描全部三角形降为只测同格内
的少数候选。反向的 M⁻¹ 也在构建时一次性预计算，避免逐点求逆。
"""

import json

import cv2
import numpy as np


def _tri_bbox(tri):
    return (tri[:, 0].min(), tri[:, 1].min(), tri[:, 0].max(), tri[:, 1].max())


def _in_triangle(px, py, tri, eps=1e-3):
    """重心坐标判定点是否在三角形内（含边界，eps 容差）。"""
    (x1, y1), (x2, y2), (x3, y3) = tri
    d = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(d) < 1e-9:
        return False
    a = ((x2 - px) * (y3 - py) - (x3 - px) * (y2 - py)) / d
    b = ((x3 - px) * (y1 - py) - (x1 - px) * (y3 - py)) / d
    return a >= -eps and b >= -eps and 1.0 - a - b >= -eps


def _affine(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


class _Grid:
    """三角形包围盒的均匀网格索引：给定一点，返回同格内的候选三角形下标。"""

    def __init__(self, boxes):
        spans = sorted(max(b[2] - b[0], b[3] - b[1]) for b in boxes) or [1.0]
        self.cell = max(float(spans[len(spans) // 2]), 1.0)
        self.buckets = {}
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for gx in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
                for gy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                    self.buckets.setdefault((gx, gy), []).append(i)

    def candidates(self, x, y):
        return self.buckets.get((int(x // self.cell), int(y // self.cell)), ())


class MeshProjector:
    """单个相机的 mesh，提供 canvas <-> source 双向点映射。"""

    def __init__(self, mesh, tex_wh, xmin, ymin, ppm, canvas_h):
        tex_w, tex_h = tex_wh
        # canvas_tris/M 的方向约定：M 把（去掉包围盒偏移的）画布点映射到相机原图，
        # M_inv 反之。故三角形按所在坐标系命名，不用 src/dst（方向随调用而异）。
        self.canvas_tris, self.M, self.M_inv, self.offset = [], [], [], []
        source_tris = []
        for tri in mesh["triangles"]:
            canvas = np.array([[(v["pos"][0] - xmin) * ppm,
                                canvas_h - 1 - (v["pos"][1] - ymin) * ppm]
                               for v in tri], np.float32)
            source = np.array([[v["uv"][0] * tex_w, (1.0 - v["uv"][1]) * tex_h]
                               for v in tri], np.float32)
            x, y, w, h = cv2.boundingRect(canvas)
            if w <= 0 or h <= 0:
                continue
            offset = np.float32([x, y])
            try:
                M = cv2.getAffineTransform(canvas - offset, source)
                M_inv = cv2.invertAffineTransform(M)
            except cv2.error:      # 退化三角形（三点共线），面积为 0 直接丢弃
                continue
            self.canvas_tris.append(canvas)
            self.M.append(M)
            self.M_inv.append(M_inv)
            self.offset.append(offset)
            source_tris.append(source)
        self._canvas_grid = _Grid([_tri_bbox(t) for t in self.canvas_tris])
        self._source_grid = _Grid([_tri_bbox(t) for t in source_tris])

    def canvas_to_source(self, cx, cy):
        """画布点 -> 该相机原图点；点不在本相机覆盖范围内返回 None。"""
        for i in self._canvas_grid.candidates(cx, cy):
            if _in_triangle(cx, cy, self.canvas_tris[i]):
                off = self.offset[i]
                return _affine(self.M[i], cx - off[0], cy - off[1])
        return None

    def source_to_canvas(self, sx, sy):
        """相机原图点 -> 画布点。仿射外推的结果用画布侧三角形边界过滤掉。"""
        for i in self._source_grid.candidates(sx, sy):
            lx, ly = _affine(self.M_inv[i], sx, sy)
            off = self.offset[i]
            cx, cy = lx + off[0], ly + off[1]
            if _in_triangle(cx, cy, self.canvas_tris[i]):
                return cx, cy
        return None

    def canvas_box_to_source_rect(self, box, steps=3):
        """
        画布框 -> 该相机原图上的轴对齐包围框。

        只在四条边上各采 steps 个点（默认 3，即角点+边中点）估算投影范围：
        这里只需要包围盒，不需要精确重建变形后的四边形，而本函数会被
        "每帧 × 每目标 × 每个候选相机"调用，采样点数直接决定 Stage2 开销。
        一个点都投不出来（该相机看不到这块区域）返回 None。
        """
        x1, y1, x2, y2 = box
        pts = []
        for t in np.linspace(0, 1, steps):
            pts += [(x1 + t * (x2 - x1), y1), (x1 + t * (x2 - x1), y2),
                    (x1, y1 + t * (y2 - y1)), (x2, y1 + t * (y2 - y1))]
        proj = [p for p in (self.canvas_to_source(cx, cy) for cx, cy in pts)
                if p is not None]
        if not proj:
            return None
        arr = np.array(proj, np.float32)
        return (arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max())


class MultiCameraProjector:
    """
    多相机 mesh 集合：把画布框映射到各相机，并按投影面积（≈清晰度）排序。

    camera_videos 的顺序必须与 mesh_json 的 meshes 数组顺序一一对应，这是
    mesh 数据自身的约定，与 texture_basename 命名是否直观无关，由调用方负责。
    """

    def __init__(self, mesh_json, camera_videos, canvas_h,
                 ppm=100.0, unit_scale=1.0, neg_v=True):
        with open(mesh_json) as f:
            meshes = json.load(f)["meshes"]
        if len(camera_videos) != len(meshes):
            raise ValueError(f"相机视频数({len(camera_videos)}) 与 mesh 数"
                             f"({len(meshes)}) 不一致，需按 mesh 顺序逐一传入")

        # 世界坐标换算成米，并按需翻转 V 轴（与 mesh 导出侧的坐标系约定一致）
        for m in meshes:
            for tri in m["triangles"]:
                for v in tri:
                    v["pos"][0] /= unit_scale
                    v["pos"][1] = (-v["pos"][1] if neg_v else v["pos"][1]) / unit_scale
        pos = np.array([v["pos"] for m in meshes for t in m["triangles"] for v in t])
        xmin, ymin = pos[:, 0].min(), pos[:, 1].min()

        self.camera_videos = camera_videos
        self.projectors = []
        for mesh, path in zip(meshes, camera_videos):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开相机视频: {path}")
            tex_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            self.projectors.append(
                MeshProjector(mesh, tex_wh, xmin, ymin, ppm, canvas_h))

    def rank_cameras(self, canvas_box):
        """
        返回所有能看到该画布框的相机，按投影面积从大到小（最清晰在前）：
            [(cam_idx, source_box), ...]
        """
        ranked = []
        for i, proj in enumerate(self.projectors):
            rect = proj.canvas_box_to_source_rect(canvas_box)
            if rect is None:
                continue
            area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
            ranked.append((area, i, rect))
        ranked.sort(key=lambda r: -r[0])
        return [(i, rect) for _, i, rect in ranked]

    def source_points_to_canvas(self, cam_idx, points_xy):
        """
        某相机原图坐标下的一批点 -> 画布坐标，形状 (N, 2)。
        映射失败的点（不在该相机 mesh 覆盖范围内）填 np.nan，调用方需跳过。
        """
        proj = self.projectors[cam_idx]
        out = np.full((len(points_xy), 2), np.nan)
        for i, (sx, sy) in enumerate(points_xy):
            r = proj.source_to_canvas(float(sx), float(sy))
            if r is not None:
                out[i] = r
        return out
