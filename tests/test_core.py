"""纯逻辑模块的单元测试（不需要 GPU / 模型权重 / 视频文件）。

    pytest tests/ -q      或      python tests/test_core.py
"""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from swim_analyse.cli import cache_key, load_cache, parse_args
from swim_analyse.geometry import MeshProjector, _Grid, _in_triangle
from swim_analyse.metrics import (_detect_valleys, _detect_zero_crossings, _fill_nan,
                                  _local_stats, _smooth, compute_speed, count_strokes,
                                  cumulative_counts)
from swim_analyse.pose import interpolate_keypoints
from swim_analyse.tracking import (SimpleTracker, contained_keep_mask,
                                   filter_contained_boxes)

FPS = 30.0


# ── geometry ────────────────────────────────────────────────────────────────

def _square_mesh(x0=0.0, y0=0.0, size=10.0):
    """一个 size×size 米的方形面片（两个三角形），UV 铺满整张纹理。"""
    corners = {(0, 0): (0.0, 0.0), (1, 0): (1.0, 0.0),
               (1, 1): (1.0, 1.0), (0, 1): (0.0, 1.0)}

    def vert(i, j):
        u, v = corners[(i, j)]
        return {"pos": [x0 + i * size, y0 + j * size], "uv": [u, v]}

    return {"triangles": [[vert(0, 0), vert(1, 0), vert(1, 1)],
                          [vert(0, 0), vert(1, 1), vert(0, 1)]]}


def test_in_triangle():
    tri = np.array([[0, 0], [10, 0], [0, 10]], float)
    assert _in_triangle(1, 1, tri)
    assert _in_triangle(0, 0, tri)          # 顶点算在内
    assert not _in_triangle(9, 9, tri)      # 斜边外侧


def test_grid_finds_candidates():
    grid = _Grid([(0, 0, 10, 10), (100, 100, 110, 110)])
    assert 0 in grid.candidates(5, 5)
    assert 1 in grid.candidates(105, 105)
    assert grid.candidates(1e6, 1e6) == ()


def test_projector_roundtrip():
    """canvas -> source -> canvas 应回到原点；范围外的点返回 None。"""
    ppm, size, tex = 100.0, 10.0, (640, 480)
    canvas_h = int(size * ppm)
    proj = MeshProjector(_square_mesh(size=size), tex, 0.0, 0.0, ppm, canvas_h)

    for cx, cy in [(100.0, 100.0), (500.0, 700.0), (900.0, 300.0)]:
        src = proj.canvas_to_source(cx, cy)
        assert src is not None
        assert 0 <= src[0] <= tex[0] + 1 and 0 <= src[1] <= tex[1] + 1
        back = proj.source_to_canvas(*src)
        assert back is not None
        assert np.allclose(back, (cx, cy), atol=1e-2)

    assert proj.canvas_to_source(-50, -50) is None
    assert proj.source_to_canvas(-10, -10) is None


def test_canvas_box_to_source_rect():
    ppm, size, tex = 100.0, 10.0, (640, 480)
    proj = MeshProjector(_square_mesh(size=size), tex, 0.0, 0.0, ppm, int(size * ppm))
    rect = proj.canvas_box_to_source_rect((200, 200, 400, 400))
    assert rect is not None
    x1, y1, x2, y2 = rect
    assert x2 > x1 and y2 > y1
    assert 0 <= x1 and x2 <= tex[0] + 1
    assert proj.canvas_box_to_source_rect((-500, -500, -400, -400)) is None


# ── tracking ────────────────────────────────────────────────────────────────

def test_filter_contained_boxes():
    big = (0, 0, 100, 100, 0.9)
    small = (10, 10, 30, 30, 0.5)      # 完全在 big 内部，置信度更低 -> 被丢
    far = (500, 500, 600, 600, 0.8)
    kept = filter_contained_boxes([big, small, far])
    assert kept == [big, far]
    # 小框置信度更高时保留小框
    assert filter_contained_boxes([big, (10, 10, 30, 30, 0.99)])[0][4] == 0.99
    assert filter_contained_boxes([]) == []


def test_contained_keep_mask_indexes_parallel_arrays():
    """掩码要能直接索引与 boxes 同序的关键点数组，且分数重复时不错配。"""
    boxes = [(0, 0, 100, 100, 0.5), (10, 10, 30, 30, 0.5),
             (500, 500, 600, 600, 0.5)]     # 三个 conf 完全相同
    keep = contained_keep_mask(boxes)
    assert keep.dtype == bool and keep.shape == (3,)
    assert list(keep) == [True, False, True], "conf 相等时丢弃被包住的那个"
    kpts = np.arange(3 * 17 * 2, dtype=float).reshape(3, 17, 2)
    assert np.array_equal(kpts[keep], kpts[[0, 2]])
    assert [b for b, k in zip(boxes, keep) if k] == filter_contained_boxes(boxes)


def test_tracker_keeps_id_across_frames():
    tracker = SimpleTracker(iou_thresh=0.3, max_lost=2)
    ids = [tracker.update([(x, 0, x + 100, 100, 0.9)])[0][0] for x in (0, 10, 20)]
    assert len(set(ids)) == 1, "平滑移动的目标应保持同一个 id"


def test_tracker_new_id_after_jump_and_lost_expiry():
    tracker = SimpleTracker(iou_thresh=0.3, max_lost=1)
    first = tracker.update([(0, 0, 100, 100, 0.9)])[0][0]
    jumped = tracker.update([(500, 500, 600, 600, 0.9)])[0][0]
    assert jumped != first, "IoU 为 0 的跳变应分配新 id"
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == {}, "超过 max_lost 的 track 应被清除"


def test_tracker_matches_two_targets_independently():
    tracker = SimpleTracker()
    a = tracker.update([(0, 0, 100, 100, 0.9), (400, 0, 500, 100, 0.9)])
    b = tracker.update([(410, 0, 510, 100, 0.9), (5, 0, 105, 100, 0.9)])   # 顺序反了
    assert {t[0] for t in a} == {t[0] for t in b}
    assert b[0][0] == a[1][0] and b[1][0] == a[0][0], "应按 IoU 而非输入顺序配对"


# ── pose ────────────────────────────────────────────────────────────────────

def _kpts(x, score=1.0, n=17):
    return np.full((n, 2), float(x)), np.full(n, float(score))


def test_interpolate_fills_middle_only():
    raw = {1: [(0, *_kpts(0)),
               (1, *_kpts(999, score=0.0)),      # 中间的低置信度帧 -> 插值
               (2, *_kpts(10)),
               (3, *_kpts(999, score=0.0))]}     # 末尾之后 -> 不外推
    out = interpolate_keypoints(raw)[1]
    assert np.allclose(out[1][0], 5.0), "中间帧应线性插值"
    assert out[1][1][0] == 0.0, "插值点置信度置 0"
    assert np.allclose(out[3][0], 999.0), "末端不做外推，保留原值"
    assert set(out) == {0, 1, 2, 3}, "插值不增删帧"


def test_interpolate_skips_when_too_few_valid():
    raw = {1: [(0, *_kpts(0, score=0.0)), (1, *_kpts(5, score=1.0))]}
    out = interpolate_keypoints(raw)[1]
    assert np.allclose(out[0][0], 0.0), "有效点少于 2 个时原样保留"


# ── metrics: 信号处理 ───────────────────────────────────────────────────────

def test_fill_nan():
    assert np.allclose(_fill_nan([0.0, np.nan, 2.0]), [0, 1, 2])
    assert np.allclose(_fill_nan([np.nan] * 3), 0.0)
    assert np.allclose(_fill_nan([np.nan, 1.0]), [1, 1]), "端点用最近有效值补齐"


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12, 100])
def test_smooth_preserves_length(n):
    assert len(_smooth(np.linspace(0, 1, n))) == n


def test_local_stats_matches_naive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    win = 7
    mean, span = _local_stats(x, win)
    half = win // 2
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        assert mean[i] == pytest.approx(w.mean())
        assert span[i] == pytest.approx(np.ptp(w))


def test_detect_valleys_counts_periods():
    """4 个周期的正弦 -> 4 个波谷。"""
    t = np.arange(0, 4.0, 1 / FPS)
    signal = np.sin(2 * np.pi * 1.0 * t)
    _, valleys = _detect_valleys(signal, FPS)
    assert len(valleys) == 4
    assert np.all(signal[valleys] < -0.9), "谷点应落在信号低位"


def test_detect_valleys_finds_shallow_segment():
    """
    前段深谷 + 后段浅谷。局部自适应阈值应在浅段仍检出大部分波谷，而同样参数
    下的全局阈值会把浅段整段淹没（这正是改用局部统计量的原因）。
    过渡区附近的窗口仍混有深段样本，因此浅段允许少量漏检。
    """
    from scipy.signal import find_peaks

    t = np.arange(0, 16.0, 1 / FPS)
    signal = np.where(t < 8.0, 1.0, 0.15) * np.sin(2 * np.pi * 1.0 * t)

    _, valleys = _detect_valleys(signal, FPS)
    deep = (valleys / FPS < 8).sum()
    shallow = (valleys / FPS >= 8).sum()

    smoothed = _smooth(_fill_nan(signal))
    inverted = -smoothed
    global_valleys, _ = find_peaks(inverted, distance=int(FPS * 0.67),
                                   height=inverted.mean(),
                                   prominence=np.ptp(inverted) * 0.2)

    assert deep == 8, "深震荡段应全部检出"
    assert shallow >= 6, f"浅段漏检过多（{shallow}/8）"
    assert (global_valleys / FPS >= 8).sum() == 0, "全局阈值本应淹没浅段（对照）"


def test_detect_valleys_flat_signal():
    _, valleys = _detect_valleys(np.zeros(100), FPS)
    assert len(valleys) == 0


def test_zero_crossings_hysteresis():
    # 从负相位起步的 4 个周期 -> 4 次由负转正
    t = np.arange(0, 4.0, 1 / FPS)
    _, cross = _detect_zero_crossings(-np.cos(2 * np.pi * 1.0 * t), FPS)
    assert len(cross) == 4

    # 起点恰为 0 且随即转正时不算穿越（未经历"低态"）
    _, cross = _detect_zero_crossings(np.sin(2 * np.pi * 1.0 * t), FPS)
    assert len(cross) == 3

    # 零点附近的小抖动不应触发计数
    noisy = 0.001 * np.sin(2 * np.pi * 8 * t) - 0.5
    _, cross = _detect_zero_crossings(noisy, FPS)
    assert len(cross) == 0


# ── metrics: 划水与速度 ─────────────────────────────────────────────────────

def _synthetic_track(n_strokes=4, fps=FPS, speed_px=10.0, n_kpts=17):
    """
    构造一个"沿 x 前进 + 手臂周期摆动"的关键点序列，每个周期恰好一次划水。

    几何：鼻子沿 x 匀速前进，肩/肘固定在鼻子后方，手腕绕肘摆动。这样
    肘角度每周期出现一次波谷、手腕相对鼻子的 x 位移每周期一次由负转正，
    两种信号的期望次数都等于 n_strokes。
    """
    t = np.arange(0, n_strokes / 1.0, 1 / fps)
    frames = {}
    for i, ti in enumerate(t):
        x = ti * speed_px
        theta = np.pi / 2 + np.pi / 3 * np.sin(2 * np.pi * 1.0 * ti)
        kpts = np.zeros((n_kpts, 2))
        kpts[0] = (x, 0.0)                                   # 鼻子
        for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
            kpts[shoulder] = (x - 1.0, 0.0)
            kpts[elbow] = (x - 0.5, 0.0)
            kpts[wrist] = kpts[elbow] + 2.0 * np.array([np.cos(theta), np.sin(theta)])
        frames[i] = (kpts, np.ones(n_kpts))
    return frames, len(t)


def test_count_strokes_and_cumulative():
    frames, n = _synthetic_track(n_strokes=4)
    res = count_strokes({7: frames}, FPS, "freestyle", "elbow_angle")[7]
    assert res["count"] == 4
    assert len(res["sides"]) == 2, "自由泳左右分开统计"

    cum = cumulative_counts({7: res}, n)[7]
    assert cum.shape == (n,)
    assert cum[0] == 0 and cum[-1] == 4
    assert np.all(np.diff(cum) >= 0), "累计计数必须单调不减"


def test_count_strokes_non_alternating_uses_single_series():
    frames, _ = _synthetic_track(n_strokes=3)
    res = count_strokes({1: frames}, FPS, "breaststroke", "elbow_angle")[1]
    assert len(res["sides"]) == 1 and res["sides"][0][0] == "Both"
    assert res["count"] == 3


def test_wrist_signal_direction_invariance():
    """反向游动（x 递减）不应改变划水次数——前进方向自动判定。"""
    frames, _ = _synthetic_track(n_strokes=4)
    forward = count_strokes({1: frames}, FPS, "freestyle", "wrist_x_head")[1]["count"]
    mirrored = {fi: (kpts * np.array([-1.0, 1.0]), sc) for fi, (kpts, sc) in frames.items()}
    backward = count_strokes({1: mirrored}, FPS, "freestyle", "wrist_x_head")[1]["count"]
    assert forward == backward > 0


def test_count_strokes_all_low_confidence():
    frames = {i: (np.zeros((17, 2)), np.zeros(17)) for i in range(60)}
    res = count_strokes({1: frames}, FPS, "freestyle", "elbow_angle")[1]
    assert res["count"] == 0, "全部低于置信度阈值时不应凭空计数"


def test_compute_speed_constant_motion():
    """每帧移动 ppm/fps 像素 = 恒定 1 m/s。"""
    ppm, n = 100.0, 300
    step = ppm / FPS
    all_boxes = {fi: [(1, fi * step, 0.0, fi * step + 50, 50.0, 0.9)] for fi in range(n)}
    samples, frame_map = compute_speed(all_boxes, FPS, ppm, n)

    speeds = [v for _, v in samples[1]]
    assert speeds and np.allclose(speeds, 1.0, atol=1e-6)
    assert frame_map[1].shape == (n,)
    assert np.isnan(frame_map[1][0]), "首个采样点之前无速度值"
    assert np.isclose(frame_map[1][-1], 1.0)


def test_compute_speed_ignores_single_frame_track():
    samples, frame_map = compute_speed({0: [(9, 0, 0, 10, 10, 0.9)]}, FPS, 100.0, 10)
    assert 9 not in samples and 9 not in frame_map


# ── cli: Stage1/2 缓存键 ────────────────────────────────────────────────────

def _args(tmp_path, weight, *extra):
    """构造一份 Plan C 的最小参数（不触碰视频，只要路径存在即可算指纹）。"""
    return parse_args([
        "--plan", "C", "--canvas-video", str(tmp_path / "canvas.mp4"),
        "--output-dir", str(tmp_path), "--yolo-model", str(weight),
        "--pose-config", str(weight), "--pose-checkpoint", str(weight), *extra])


def test_cache_key_covers_params_and_weight_mtime(tmp_path):
    weight = tmp_path / "w.pt"
    weight.write_bytes(b"v1")
    base = cache_key(_args(tmp_path, weight), 100)

    assert cache_key(_args(tmp_path, weight), 100) == base, "同参同文件应稳定命中"
    assert cache_key(_args(tmp_path, weight), 200) != base, "帧数不同应 miss"
    assert cache_key(_args(tmp_path, weight, "--conf", "0.5"), 100) != base
    assert cache_key(_args(tmp_path, weight, "--containment", "0.5"), 100) != base
    # Stage3/4 的参数不该影响 Stage1/2 缓存
    assert cache_key(_args(tmp_path, weight, "--kpt-thr", "0.9"), 100) == base
    assert cache_key(_args(tmp_path, weight, "--signal", "wrist_x_head"), 100) == base

    other = tmp_path / "w2.pt"
    other.write_bytes(b"v1")
    assert cache_key(_args(tmp_path, other), 100) != base, "换权重路径应 miss"
    weight.write_bytes(b"version-2-longer")             # 同名覆盖 -> 大小变化
    assert cache_key(_args(tmp_path, weight), 100) != base, "权重内容变了应 miss"


def test_load_cache_hit_miss_and_legacy(tmp_path):
    path = tmp_path / "cache.pkl"
    assert load_cache(str(path), "k1") is None, "文件不存在 -> miss"

    payload = {"key": "k1", "all_boxes": {}, "raw_seq": {}}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    assert load_cache(str(path), "k1") == payload
    assert load_cache(str(path), "k2") is None, "key 不符 -> miss"

    with open(path, "wb") as f:                        # 旧格式：只有 plan，无 key
        pickle.dump({"plan": "C", "all_boxes": {}, "raw_seq": {}}, f)
    assert load_cache(str(path), "k1") is None, "旧格式 -> miss 而非异常"

    path.write_bytes(b"not a pickle")                  # 损坏文件也只能 miss
    assert load_cache(str(path), "k1") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
