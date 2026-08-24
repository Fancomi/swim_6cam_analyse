"""RTMPose(mmpose) top-down 关键点回归 + COCO17 关键点定义。

RTMPose 是纯关键点回归模型，`inference_topdown(model, img, bboxes)` 原生接受
外部指定的 bbox 数组，模型内部按每个 bbox 做仿射采样到 192x256 再批量前向。
因此不需要像 YOLO-Pose 那样先手动 crop 成小图——直接把整帧 + 该(帧,相机)下
所有目标的 bbox 一次性传进去即可，坐标输出已在原图坐标系。
"""

import numpy as np

# COCO17 关键点索引
NOSE = 0
SHOULDER_L, SHOULDER_R = 5, 6
ELBOW_L, ELBOW_R = 7, 8
WRIST_L, WRIST_R = 9, 10

# 关键点置信度阈值：低于此值的点视为不可靠，不参与信号计算、插值和绘制
KPT_THR = 0.4

# Plan A 换相机用的"17 点平均置信度"门限：最清晰相机的结果均分低于此值时，
# 改用次清晰相机再推一次。与 KPT_THR 是两回事（那个是单点阈值），只是默认
# 取同一数值以保持历史行为。
POSE_SCORE_THR = 0.4

# 骨架连线 (i, j) 及配色（BGR）。同侧肢体同色：左肢绿、右肢蓝、躯干与头部橙。
_LIMB = (0, 255, 0)        # 左侧
_RIMB = (0, 128, 255)      # 右侧
_BODY = (255, 153, 51)     # 躯干/头部
SKELETON = [
    ((15, 13), _LIMB), ((13, 11), _LIMB),          # 左腿
    ((16, 14), _RIMB), ((14, 12), _RIMB),          # 右腿
    ((11, 12), _BODY), ((5, 11), _BODY), ((6, 12), _BODY), ((5, 6), _BODY),
    ((5, 7), _LIMB), ((7, 9), _LIMB),              # 左臂
    ((6, 8), _RIMB), ((8, 10), _RIMB),             # 右臂
    ((1, 2), _BODY), ((0, 1), _BODY), ((0, 2), _BODY),
    ((1, 3), _BODY), ((2, 4), _BODY),              # 头/耳
]
# 各关键点颜色：0-4 头部，奇数为左侧，偶数为右侧
KPT_COLORS = [_BODY] * 5 + [_LIMB if i % 2 else _RIMB for i in range(5, 17)]


class RTMPoseEstimator:
    """加载一次模型，按 (帧, bbox 列表) 批量回归关键点。"""

    def __init__(self, config, checkpoint, device="0"):
        from mmpose.apis import init_model, inference_topdown
        self._infer = inference_topdown
        self.model = init_model(
            config, checkpoint,
            device=f"cuda:{device}" if str(device).isdigit() else device)

    def __call__(self, frame, bboxes_xyxy):
        """
        frame: BGR 整帧；bboxes_xyxy: [(x1,y1,x2,y2)]（原图坐标，无需 padding，
        mmpose 内部会按固定比例扩展 bbox 再采样）。

        返回 [(kpts(17,2) float32, scores(17,) float32)]，与输入 bbox 一一对应。
        bbox 由外部给定，不存在"检测不到人"的情况，因此每个 bbox 必有结果。
        """
        if len(bboxes_xyxy) == 0:
            return []
        results = self._infer(self.model, frame,
                              np.asarray(bboxes_xyxy, np.float32), bbox_format="xyxy")
        return [(r.pred_instances.keypoints[0], r.pred_instances.keypoint_scores[0])
                for r in results]


def interpolate_keypoints(raw_seq):
    """
    对每个目标的关键点时序按关键点逐个做线性插值，填补低置信度/丢失的点。

    只在该关键点首末两个可靠帧之间插值（不做端点外推），插值出来的点置信度
    置 0——这样下游的"按置信度过滤"逻辑天然会把插值点排除在信号计算和骨架
    绘制之外，同时保留了它们的坐标供需要连续轨迹的场合使用。

    raw_seq: {tid: [(frame_idx, kpts(17,2), scores(17,), cam_idx)]}
    返回:    {tid: {frame_idx: (kpts(17,2), scores(17,))}}（帧集合与输入相同）
    """
    out = {}
    for tid, seq in raw_seq.items():
        if not seq:
            continue
        seq = sorted(seq, key=lambda s: s[0])
        fis = np.array([s[0] for s in seq])
        kpts = np.stack([s[1] for s in seq]).astype(float)
        scores = np.stack([s[2] for s in seq]).astype(float)

        for ki in range(kpts.shape[1]):
            valid = scores[:, ki] >= KPT_THR
            if valid.sum() < 2:
                continue
            # 待插值的帧：置信度不足，且落在首末可靠帧之间
            vfis = fis[valid]
            todo = ~valid & (fis >= vfis[0]) & (fis <= vfis[-1])
            if not todo.any():
                continue
            for axis in (0, 1):
                kpts[todo, ki, axis] = np.interp(fis[todo], vfis, kpts[valid, ki, axis])
            scores[todo, ki] = 0.0

        out[tid] = {int(fi): (kpts[i], scores[i]) for i, fi in enumerate(fis)}
    return out
