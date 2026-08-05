"""从关键点时序和检测框轨迹导出运动指标：划水次数、瞬时速度。

划水次数支持两种信号，接口统一（见 SIGNALS）：

  elbow_angle （默认）
      肩-肘-腕夹角的波谷计数。波谷是"高肘抓水"姿态，肩肘腕构型立体、角度
      对关键点噪声不敏感；波峰是手臂伸直、三点接近共线，此时夹角对坐标
      误差极敏感，因此检谷不检峰。

  wrist_x_head
      手腕沿泳道方向相对鼻子的带符号位移，做"由负转正"过零计数（手腕从
      头后摆到头前算一次）。

自由泳/仰泳左右手交替各算一次划水，取 min(左, 右) 更抗单侧漏检；其余泳姿
双臂同步，左右信号取均值后统一计数。
"""

import numpy as np
from scipy.ndimage import (maximum_filter1d, median_filter, minimum_filter1d)
from scipy.signal import find_peaks, savgol_filter

from .pose import (ELBOW_L, ELBOW_R, KPT_THR, NOSE, SHOULDER_L, SHOULDER_R,
                   WRIST_L, WRIST_R)

# 信号平滑：中值滤波去单帧跳变，再 Savitzky-Golay 保形平滑
MEDIAN_WINDOW = 5
SG_WINDOW, SG_POLYORDER = 11, 3

# 相邻两次划水的最小间隔（秒），用于抑制滤波后残留的高频抖动
MIN_STROKE_INTERVAL = 0.67

# 波谷检测的局部自适应窗口（秒）。信号可能分段呈现不同幅度（动作变化导致
# 后段震荡变浅），用全局统计量算阈值会系统性漏掉浅震荡段，故按局部统计。
ADAPTIVE_WINDOW = 6.0
ADAPTIVE_PROMINENCE_RATIO = 0.2

# 过零检测的滞回阈值（相对信号峰峰值），必须先跌破 -δ 再升破 +δ 才算一次，
# 防止零点附近抖动重复计数
HYSTERESIS_RATIO = 0.15
ZC_MIN_INTERVAL = 0.3

# 瞬时速度：每 SAMPLE_DT 秒输出一个值，用最近 WINDOW_SEC 秒的位移求平均
SPEED_SAMPLE_DT = 0.2
SPEED_WINDOW_SEC = 2.0


def _fill_nan(signal):
    """线性插值填补 NaN；全 NaN 时返回全 0。"""
    signal = np.asarray(signal, float).copy()
    nan = np.isnan(signal)
    if nan.all():
        return np.zeros_like(signal)
    idx = np.arange(len(signal))
    signal[nan] = np.interp(idx[nan], idx[~nan], signal[~nan])
    return signal


def _odd(n, upper):
    """不超过 n 和 upper 的最大正奇数，用于把滤波窗口收敛到信号长度内。"""
    m = min(int(n), int(upper))
    return max(1, m if m % 2 else m - 1)


def _smooth(signal):
    """中值 + Savitzky-Golay 平滑。信号过短时自动缩小窗口或跳过 SG。"""
    n = len(signal)
    med = median_filter(signal, size=_odd(n, MEDIAN_WINDOW))
    win = _odd(n, SG_WINDOW)
    return med if win < SG_POLYORDER + 2 else savgol_filter(med, win, SG_POLYORDER)


def _local_stats(x, win):
    """以各点为中心、宽 win（取奇数）的截断滑动窗口的均值与峰峰值。"""
    n = len(x)
    half = _odd(n, win) // 2
    lo = np.maximum(0, np.arange(n) - half)
    hi = np.minimum(n, np.arange(n) + half + 1)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    mean = (cum[hi] - cum[lo]) / (hi - lo)
    # mode='nearest' 用边界值补齐，而边界值本身就在截断窗口内，
    # 因此补齐后的极值与截断窗口的极值完全相同
    size = 2 * half + 1
    span = (maximum_filter1d(x, size, mode="nearest")
            - minimum_filter1d(x, size, mode="nearest"))
    return mean, span


def _detect_valleys(signal, fps):
    """局部自适应阈值的波谷检测，返回 (平滑信号, 波谷索引)。"""
    smoothed = _smooth(_fill_nan(signal))
    if float(np.ptp(smoothed)) < 1e-6:
        return smoothed, np.array([], int)
    distance = max(1, int(fps * MIN_STROKE_INTERVAL))
    inverted = -smoothed                     # find_peaks 只找峰，取负后峰即原信号的谷
    win = min(max(distance * 2, int(fps * ADAPTIVE_WINDOW)), len(inverted))
    mean, span = _local_stats(inverted, win)
    valleys, _ = find_peaks(
        inverted, distance=distance, height=mean,
        prominence=np.maximum(span * ADAPTIVE_PROMINENCE_RATIO, 1e-6))
    return smoothed, valleys


def _detect_zero_crossings(signal, fps):
    """滞回式"由负转正"过零检测，返回 (平滑信号, 穿越索引)。"""
    smoothed = _smooth(_fill_nan(signal))
    amplitude = float(np.ptp(smoothed))
    if amplitude < 1e-6:
        return smoothed, np.array([], int)
    delta = amplitude * HYSTERESIS_RATIO
    min_gap = max(1, int(fps * ZC_MIN_INTERVAL))

    crossings, below, last = [], smoothed[0] < 0, -min_gap
    for i, v in enumerate(smoothed):
        if below and v > delta:
            if i - last >= min_gap:
                crossings.append(i)
                last = i
            below = False
        elif not below and v < -delta:
            below = True
    return smoothed, np.array(crossings, int)


def _angle(kpts, scores, a, b, c):
    """∠abc（弧度）。三点任一置信度不足返回 NaN。"""
    if min(scores[a], scores[b], scores[c]) < KPT_THR:
        return np.nan
    v1, v2 = kpts[a] - kpts[b], kpts[c] - kpts[b]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def _elbow_angles(frames):
    """frames: [(kpts, scores)] -> 左右肘角度序列（弧度）。"""
    return tuple(
        np.array([_angle(k, s, sh, el, wr) for k, s in frames])
        for sh, el, wr in ((SHOULDER_L, ELBOW_L, WRIST_L),
                           (SHOULDER_R, ELBOW_R, WRIST_R)))


def _wrist_x_head(frames):
    """
    frames: [(kpts, scores)] -> 左右手腕沿前进方向相对鼻子的带符号位移（像素）。

    前进方向由该目标鼻子 x 坐标的首末净位移判定，使"手腕在头前方"恒为正，
    与游动朝向无关。
    """
    nose_x = [k[NOSE][0] for k, s in frames if s[NOSE] >= KPT_THR]
    sign = -1.0 if len(nose_x) >= 2 and nose_x[-1] < nose_x[0] else 1.0
    out = []
    for wrist in (WRIST_L, WRIST_R):
        out.append(np.array([
            sign * (k[wrist][0] - k[NOSE][0])
            if min(s[wrist], s[NOSE]) >= KPT_THR else np.nan
            for k, s in frames]))
    return tuple(out)


# 每种信号：如何从关键点算信号、如何从信号找划水事件、画图时的纵轴含义
SIGNALS = {
    "elbow_angle": dict(build=_elbow_angles, detect=_detect_valleys,
                        ylabel="Elbow angle (deg)", to_deg=True),
    "wrist_x_head": dict(build=_wrist_x_head, detect=_detect_zero_crossings,
                         ylabel="Wrist x - Nose x (px)", to_deg=False),
}


def count_strokes(interp_data, fps, stroke_type, signal="elbow_angle"):
    """
    interp_data: interpolate_keypoints 的输出 {tid: {frame_idx: (kpts, scores)}}

    返回 {tid: {
        "count":       划水总次数,
        "frame_idxs":  该目标有数据的帧号 (T,),
        "sides":       [(名称, 原始信号, 平滑信号, 事件索引)]  # 供画图
        "event_frames":[各侧事件帧号数组]                     # 供累计计数
    }}
    """
    spec = SIGNALS[signal]
    split_sides = stroke_type in ("freestyle", "backstroke")
    results = {}

    for tid, frame_map in interp_data.items():
        items = sorted(frame_map.items())
        frame_idxs = np.array([fi for fi, _ in items])
        raw_l, raw_r = spec["build"]([kf for _, kf in items])

        if split_sides:
            series = [("Left", raw_l), ("Right", raw_r)]
        else:
            # 左右均值；两侧同帧都缺失时保持 NaN（交给 _fill_nan 统一处理）
            both = np.stack([raw_l, raw_r])
            valid = ~np.isnan(both)
            n_valid = valid.sum(0)
            avg = np.where(n_valid > 0,
                           np.where(valid, both, 0.0).sum(0) / np.maximum(n_valid, 1),
                           np.nan)
            series = [("Both", avg)]

        sides, event_frames = [], []
        for name, raw in series:
            smoothed, events = spec["detect"](raw, fps)
            sides.append((name, raw, smoothed, events))
            event_frames.append(frame_idxs[events] if len(events) else np.empty(0, int))

        results[tid] = {
            "count": min(len(e) for e in event_frames),
            "frame_idxs": frame_idxs,
            "sides": sides,
            "event_frames": event_frames,
        }
    return results


def cumulative_counts(stroke_results, total_frames):
    """
    {tid: 每帧的累计划水次数 (total_frames,)}。

    多侧信号时取各侧累计次数的较小值——只有左右手都完成了一次摆动才算一次
    完整划水，与 count_strokes 里 min(左, 右) 的口径一致。
    """
    frames = np.arange(total_frames)
    return {
        tid: np.min([np.searchsorted(ef, frames, side="right")
                     for ef in res["event_frames"]], axis=0)
        for tid, res in stroke_results.items()
    }


def compute_speed(all_boxes, fps, ppm, total_frames,
                  sample_dt=SPEED_SAMPLE_DT, window_sec=SPEED_WINDOW_SEC):
    """
    用画布坐标下检测框中心的轨迹估算瞬时速度（m/s）。

    每 sample_dt 秒取一个采样点，用最近 window_sec 秒窗口内首末两点的位移
    除以对应时间差。取框中心而非某个关键点：框中心来自跟踪器、帧间连续性好，
    也不需要经过 mesh 反投影（画布坐标本身就是等比米制，除以 ppm 即得米）。

    返回 (samples, frame_map):
        samples   {tid: [(时间秒, 速度)]}       —— 稀疏采样点，用于落盘
        frame_map {tid: (total_frames,) 数组}   —— 逐帧取值（前向填充，
                                                   首个采样点之前为 NaN）
    """
    tracks = {}
    for fi, items in all_boxes.items():
        for tid, x1, y1, x2, y2, _conf in items:
            tracks.setdefault(tid, {})[fi] = ((x1 + x2) / 2, (y1 + y2) / 2)

    window_frames = window_sec * fps
    step = max(1, round(sample_dt * fps))
    samples, frame_map = {}, {}

    for tid, per_frame in tracks.items():
        fis = np.array(sorted(per_frame))
        if len(fis) < 2:
            continue
        pts = np.array([per_frame[fi] for fi in fis])

        pairs = []
        for fi in range(int(fis[0]), int(fis[-1]) + 1, step):
            win = np.flatnonzero((fis >= fi - window_frames) & (fis <= fi))
            if len(win) < 2:
                continue
            dt = (fis[win[-1]] - fis[win[0]]) / fps
            if dt > 1e-6:
                dist_m = float(np.linalg.norm(pts[win[-1]] - pts[win[0]])) / ppm
                pairs.append((fi / fps, dist_m / dt))
        samples[tid] = pairs

        arr = np.full(total_frames, np.nan)
        for t, v in pairs:
            fi = int(round(t * fps))
            if 0 <= fi < total_frames:
                arr[fi] = v
        # 前向填充：采样点之间保持上一次的速度值，首个采样点之前保持 NaN
        src = np.maximum.accumulate(np.where(np.isnan(arr), -1, np.arange(total_frames)))
        frame_map[tid] = np.where(src >= 0, arr[np.maximum(src, 0)], np.nan)

    return samples, frame_map


def save_signal_plots(stroke_results, out_dir, fps, stroke_type, signal):
    """每个目标存一张信号图：原始 + 平滑信号，划水事件用叉号标出。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = SIGNALS[signal]
    os.makedirs(out_dir, exist_ok=True)

    for tid, res in stroke_results.items():
        sides = res["sides"]
        time_axis = res["frame_idxs"] / fps
        fig, axes = plt.subplots(len(sides), 1, figsize=(16, 4 * len(sides)),
                                 sharex=True, squeeze=False)
        for ax, (name, raw, smoothed, events) in zip(axes[:, 0], sides):
            conv = np.degrees if spec["to_deg"] else (lambda v: v)
            ax.plot(time_axis, conv(_fill_nan(raw)), color="lightgrey", lw=1.0, label="raw")
            ax.plot(time_axis, conv(smoothed), color="seagreen", lw=1.8, label="median+SG")
            if not spec["to_deg"]:
                ax.axhline(0, color="grey", lw=0.8, ls="--")
            if len(events):
                ax.scatter(time_axis[events], conv(smoothed)[events], color="crimson",
                           marker="x", s=50, lw=2, zorder=5, label="stroke")
            ax.set_ylabel(spec["ylabel"])
            ax.set_title(f"ID:{tid}  {name}  strokes={res['count']}  "
                         f"stroke:{stroke_type}  signal:{signal}")
            ax.legend(fontsize=9, loc="upper right")
        axes[-1, 0].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"id{tid}.png"), dpi=130)
        plt.close(fig)
    print(f"[Plot] 已保存 {len(stroke_results)} 张信号图 -> {out_dir}")
