"""三套关键点方案（Plan A/B/C）的统一封装。

三者的差别只在"如何从画布视频得到每个人的框与关键点"，之后的划水计数、速度、
渲染完全共用（见 cli.py 的 Stage3/4）。因此这里把它们抽象成同一个接口：

    plan.run(canvas_video, total) -> (all_boxes, raw_seq)
    plan.to_canvas(raw_seq, interp) -> canvas_kpts        # 关键点转画布坐标供绘制

run() 里"开画布视频 -> 逐帧 read -> 收集 -> 释放"的骨架三套一字不差，故只在基类
留一份（模板方法）：子类实现 _setup()/_frame()，A 另需 _context()/_finish()。

数据约定（三套完全一致，Stage3/4 无需知道用了哪套）：
    all_boxes  {frame_idx: [(track_id, x1, y1, x2, y2, conf)]}   画布坐标
    raw_seq    {track_id: [(frame_idx, kpts(17,2), scores(17,), cam_idx)]}
               kpts 在 cam_idx 指定的坐标系下；cam_idx = -1 表示已是画布坐标

┌────────┬──────────────────────┬─────────────────────────────────────────────┐
│ Plan   │ 关键点在哪算          │ 代价与收益                                   │
├────────┼──────────────────────┼─────────────────────────────────────────────┤
│ A 交接 │ 原相机（mesh 映射）   │ 分辨率最高；需 6 路 4K 解码 + mesh，最慢      │
│ B 一体 │ 画布（yolo-pose）     │ 一次前向出框+点；人多时最省，精度最低          │
│ C 两阶段│ 画布（RTMPose 按框） │ 检测与关键点解耦，各自最优；速度与 B 持平     │
└────────┴──────────────────────┴─────────────────────────────────────────────┘

三套方案的实测指标对比见 README.md 的三套方案对比表。
"""

import contextlib
import time

import cv2

from .geometry import MultiCameraProjector
from .tracking import SimpleTracker, contained_keep_mask, filter_contained_boxes
from .video import MultiVideoReader

CANVAS_CAM = -1          # cam_idx 哨兵：关键点已在画布坐标系


def _boxes_of(result):
    """ultralytics 结果 -> [(x1,y1,x2,y2,conf)]（保序），无框时 []。

    这里不顺手去重：Plan B 还要用与本列表同序的关键点数组，只能自己拿掩码筛。
    """
    if result.boxes is None or not len(result.boxes):
        return []
    return [(*map(float, b), float(c)) for b, c in
            zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]


class Plan:
    """三套方案的公共接口与共享工具。"""

    name = "?"
    description = "?"
    needs_cameras = False        # 是否需要 6 路原相机视频

    def __init__(self, args, canvas_h):
        self.args = args
        self.canvas_h = canvas_h
        self.timings = {}
        # 三套都在画布坐标上跟踪、参数也相同，故在基类统一构造
        self.tracker = SimpleTracker(args.track_iou, args.track_max_lost)
        # 纯数字的 --device 视为 CUDA 序号，其余（cpu / cuda:1 等）原样透传
        self.device = (f"cuda:{args.device}" if str(args.device).isdigit()
                       else args.device)

    def run(self, canvas_video, total):
        """逐帧驱动 + 收集，子类只实现 _frame()。"""
        self._setup()
        cap = cv2.VideoCapture(canvas_video)
        if not cap.isOpened():          # 否则读到 0 帧只会静默返回空结果
            raise RuntimeError(f"无法打开视频: {canvas_video}")
        self.all_boxes, self.raw_seq = {}, {}
        try:
            with self._context():
                for fi in range(total):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._frame(fi, frame)
        finally:
            cap.release()
        self._finish()
        return self.all_boxes, self.raw_seq

    # ── 子类钩子 ────────────────────────────────────────────────────────────
    def _setup(self):
        """加载本方案的模型；run() 内才调用，保持惰性（见 _load_yolo）。"""

    def _context(self):
        """逐帧循环期间要持有的额外资源；只有 Plan A 需要（六路相机 reader）。"""
        return contextlib.nullcontext()

    def _frame(self, fi, frame):
        """处理画布第 fi 帧，结果写进 self.all_boxes / self.raw_seq。"""
        raise NotImplementedError

    def _finish(self):
        """收尾输出；只有 Plan A 需要（打印补检次数）。"""

    def to_canvas(self, raw_seq, interp):
        """默认：关键点已是画布坐标，直接用。"""
        return {tid: dict(frames) for tid, frames in interp.items()}

    # ── 共享工具 ────────────────────────────────────────────────────────────
    def _load_yolo(self, path):
        """加载 YOLO 权重并搬到 self.device。

        局部 import：ultralytics/mmpose 各要几秒才 import 完，而 Plan B 不需要
        mmpose、A/C 不需要 yolo-pose —— 都推迟到真正 run() 时才付出。
        """
        from ultralytics import YOLO

        model = YOLO(path)
        model.to(self.device)
        return model

    def _setup_det_rtmpose(self):
        """A/C 共用：画布检测器 + RTMPose（config/checkpoint 各自不同）。"""
        from .pose import RTMPoseEstimator

        a = self.args
        self.det = self._load_yolo(a.yolo_model)
        self.pose = RTMPoseEstimator(a.pose_config, a.pose_checkpoint, a.device)

    def _detect_track(self, fi, frame):
        """A/C 共用的前半段：检测去重 -> 跟踪 -> 记框，返回 tracked（无框时 []）。

        tracked[i][1:5] 就是去重后的框坐标本身（tracker 只在前面加了 tid、不改
        数值），下游要框直接切片，不必再单独接一份 boxes。
        """
        a = self.args
        boxes = self._timeit("detect", self._detect, self.det, frame,
                             a.conf, a.iou, a.containment)
        if not boxes:
            return []
        tracked = self.tracker.update(boxes)
        self.all_boxes[fi] = tracked
        return tracked

    def _timeit(self, key, fn, *a, **kw):
        t0 = time.time()
        try:
            return fn(*a, **kw)
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + time.time() - t0

    def report(self, total_sec, n_frames):
        parts = "  ".join(f"{k}={v:.1f}s({100*v/max(total_sec,1e-6):.0f}%)"
                          for k, v in self.timings.items())
        print(f"[{self.name}] 耗时 {total_sec:.1f}s = "
              f"{total_sec/max(n_frames,1)*1000:.0f} ms/帧  {parts}")

    @staticmethod
    def _detect(model, frame, conf, iou, containment):
        """YOLO 检测 + 包含率去重，返回 [(x1,y1,x2,y2,conf)]（保序）。"""
        boxes = _boxes_of(model(frame, conf=conf, iou=iou, verbose=False)[0])
        return filter_contained_boxes(boxes, containment)


class PlanA_CrossCamera(Plan):
    """
    Plan A（交接原版）：画布检测 -> mesh 映射到最清晰相机 -> 相机 RTMPose -> 反投影。

    关键点在原相机上算，分辨率最高（画布框 224x78 映射回相机后 365x85，面积 ×1.7）。
    代价是每帧要解 4.65 路 4K 帧、做 416 万次几何点查询，且 batch 被切碎
    （19.3 人分 4.65 组，RTMPose 的 18.2ms 固定开销被重复付出）。
    """

    name = "PlanA"
    description = "画布 detect + 原相机 RTMPose + mesh 反投影（交接原版）"
    needs_cameras = True

    def _setup(self):
        a = self.args
        self._setup_det_rtmpose()
        self.projector = MultiCameraProjector(
            a.mesh, a.camera_videos, canvas_h=self.canvas_h,
            ppm=a.ppm, unit_scale=a.unit_scale, neg_v=a.neg_v)
        self.n_retry = 0

    @contextlib.contextmanager
    def _context(self):
        """六路原相机的帧级随机读取器，只在逐帧循环期间持有。"""
        with MultiVideoReader(self.args.camera_videos) as reader:
            self.reader = reader
            yield

    def _frame(self, fi, frame):
        a = self.args
        tracked = self._detect_track(fi, frame)
        if not tracked:
            return

        # 每个目标按投影面积排序候选相机；最清晰那路按相机分组批量推理
        ranked, by_cam = {}, {}
        for tid, x1, y1, x2, y2, _c in tracked:
            cams = self._timeit("geometry", self.projector.rank_cameras,
                                (x1, y1, x2, y2))
            if not cams:
                continue
            ranked[tid] = cams
            by_cam.setdefault(cams[0][0], []).append((tid, cams[0][1]))

        low = []
        for ci, entries in by_cam.items():
            cf = self._timeit("video_io", self.reader.read, ci, fi)
            if cf is None:
                continue
            res = self._timeit("pose", self.pose, cf, [b for _, b in entries])
            for (tid, _box), (kp, sc) in zip(entries, res):
                if float(sc.mean()) < a.pose_score_thr:
                    low.append((tid, ci, kp, sc))
                else:
                    self.raw_seq.setdefault(tid, []).append((fi, kp, sc, ci))

        # 最清晰相机置信度不足 -> 用次清晰相机补检一次
        for tid, ci, kp, sc in low:
            second = next((c for c in ranked[tid] if c[0] != ci), None)
            if second is not None:
                better = self._timeit("retry", self._retry, fi, *second)
                if better is not None:
                    ci, (kp, sc) = second[0], better
                    self.n_retry += 1
            self.raw_seq.setdefault(tid, []).append((fi, kp, sc, ci))

    def _retry(self, fi, cam_idx, box):
        """次清晰相机上重推一次；读不到帧或仍不达标都返回 None（沿用原结果）。"""
        cf = self.reader.read(cam_idx, fi)
        if cf is None:
            return None
        (kp, sc), = self.pose(cf, [box])
        return (kp, sc) if float(sc.mean()) >= self.args.pose_score_thr else None

    def _finish(self):
        print(f"[{self.name}] 次清晰相机补检成功 {self.n_retry} 次")

    def to_canvas(self, raw_seq, interp):
        """关键点在各自相机坐标系，需经 mesh 反投影回画布。"""
        out = {}
        for tid, frames in interp.items():
            cam_of = {fi: cam for fi, _, _, cam in raw_seq.get(tid, [])}
            out[tid] = {
                fi: (self.projector.source_points_to_canvas(cam_of[fi], kp), sc)
                for fi, (kp, sc) in frames.items() if fi in cam_of
            }
        return out


class PlanB_UnifiedCanvas(Plan):
    """
    Plan B：画布 yolo26-pose 一体，一次前向同时出检测框与关键点。

    detect 与 pose 共享骨干，整图并行 —— 耗时几乎与人数无关（实测 17 人 35ms、
    68 人 17ms）。代价是一个骨干、一个输入尺度同时服务两个任务，两边都要折中：
    实测检测 AP50-95 0.477、关键点 PCK@5% 67.3%，均低于 Plan C。
    """

    name = "PlanB"
    description = "画布 yolo26-pose 一体（detect+pose 单次前向）"

    def _setup(self):
        self.model = self._load_yolo(self.args.pose_model)

    def _frame(self, fi, frame):
        a = self.args
        # 一次前向同时出框与点，故整帧只有这一项耗时，全部计入 pose
        r = self._timeit("pose", self.model, frame, conf=a.conf, verbose=False)[0]
        boxes = _boxes_of(r)
        if not boxes:
            return

        # yolo-pose 的 data 是 (N,17,3)：x, y, 可见性；只取 x,y
        kpts = r.keypoints.data.cpu().numpy()[:, :, :2]       # (N,17,2)
        scores = r.keypoints.conf.cpu().numpy()               # (N,17)

        # 去重掩码直接用于同步筛掉对应关键点（不做浮点相等反查下标）
        keep = contained_keep_mask(boxes, a.containment)
        kept = [b for b, k in zip(boxes, keep) if k]
        kpts, scores = kpts[keep], scores[keep]

        tracked = self.tracker.update(kept)
        self.all_boxes[fi] = tracked
        for (tid, *_), kp, sc in zip(tracked, kpts, scores):
            self.raw_seq.setdefault(tid, []).append((fi, kp, sc, CANVAS_CAM))


class PlanC_CanvasTwoStage(Plan):
    """
    Plan C：画布 yolo26 detect -> 画布 RTMPose（top-down，按框裁切）。

    检测与关键点解耦，各自用专门模型：检测器可专门优化召回（实测召回 0.972
    vs Plan B 的 0.917），RTMPose 按框裁切到 192x256 拿到高有效分辨率
    （PCK@5% 88.3% vs 67.3%）。相比 Plan A 省掉了 4K 解码与 mesh，
    且 batch 不再被切碎（一帧的人一次推完）。
    """

    name = "PlanC"
    description = "画布 yolo26 detect + 画布 RTMPose（两阶段，均在画布坐标）"

    def _setup(self):
        self._setup_det_rtmpose()

    def _frame(self, fi, frame):
        tracked = self._detect_track(fi, frame)
        if not tracked:
            return
        # 一帧的所有人一次推完（不像 Plan A 要按相机分组）
        res = self._timeit("pose", self.pose, frame, [t[1:5] for t in tracked])
        for (tid, *_), (kp, sc) in zip(tracked, res):
            self.raw_seq.setdefault(tid, []).append((fi, kp, sc, CANVAS_CAM))


PLANS = {"A": PlanA_CrossCamera, "B": PlanB_UnifiedCanvas, "C": PlanC_CanvasTwoStage}
