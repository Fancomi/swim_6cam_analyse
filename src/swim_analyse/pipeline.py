"""
六路相机拼接全景视频的游泳分析流水线：检测跟踪 -> 关键点 -> 划水/速度 -> 标注视频。

    Stage1  在拼接画布上逐帧 YOLO 检测 + IoU 跟踪，得到每帧每个 id 的画布框
    Stage2  用 mesh 几何把画布框反投影到"投影面积最大（最清晰）"的那路相机，
            读该相机同帧原图，把该(帧,相机)下所有目标的框一次性交给 RTMPose
            批量回归关键点；整体置信度不达标的目标改用次清晰相机补检一次
    Stage3  关键点时序插值 -> 划水次数（肘角度波谷 / 手腕过零）+ 瞬时速度
    Stage4  重读拼接视频，叠加框、ID/划水数/速度标签、可选关键点骨架

Stage1+2 的结果会缓存到 output_dir/cache.pkl，重跑时若缓存存在则直接跳到
Stage3——调信号参数、换绘制选项都不必重跑 GPU 推理。

用法见 run.sh，或 `python -m swim_analyse.pipeline --help`。
"""

import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np

from .draw import TrackCropWriter, draw_label, draw_skeleton, id_color
from .geometry import MultiCameraProjector
from .metrics import (SIGNALS, compute_speed, count_strokes, cumulative_counts,
                      save_signal_plots)
from .pose import KPT_THR, RTMPoseEstimator, interpolate_keypoints
from .tracking import SwimmerTracker
from .video import MultiVideoReader, video_meta


class Progress:
    """按帧数间隔打印进度与各环节累计耗时。"""

    def __init__(self, stage, total, interval=100):
        self.stage, self.total, self.interval = stage, total, interval
        self.start = time.time()
        self.timers = {}

    def timeit(self, key, fn, *args, **kwargs):
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            self.timers[key] = self.timers.get(key, 0.0) + time.time() - t0

    def step(self, done, **extra):
        if done % self.interval:
            return
        parts = [f"{k}={v:.1f}s" for k, v in self.timers.items()]
        parts += [f"{k}={v}" for k, v in extra.items()]
        print(f"[{self.stage}] {done}/{self.total} 已耗时 {time.time() - self.start:.1f}s "
              + "  ".join(parts), flush=True)

    def done(self, msg=""):
        elapsed = time.time() - self.start
        detail = "  ".join(f"{k}={v:.1f}s({100 * v / max(elapsed, 1e-6):.0f}%)"
                           for k, v in self.timers.items())
        print(f"[{self.stage}] 完成 耗时 {elapsed:.1f}s  {detail}  {msg}")


def stage1_track(canvas_video, yolo_model, device, max_frames=0):
    """逐帧检测 + 跟踪，返回 ({frame_idx: [(tid,x1,y1,x2,y2,conf)]}, (w,h,fps,帧数))。"""
    w, h, fps, total = video_meta(canvas_video)
    if max_frames:
        total = min(total, max_frames)
    print(f"[Stage1] 拼接视频 {w}x{h} {fps:.2f}fps 处理 {total} 帧 -> {canvas_video}")
    tracker = SwimmerTracker(yolo_model, device=device)
    cap = cv2.VideoCapture(canvas_video)
    prog = Progress("Stage1", total)

    all_boxes, n_read = {}, 0
    while n_read < total:
        ok, frame = cap.read()
        if not ok:
            break
        tracked = tracker.update(frame)
        if tracked:
            all_boxes[n_read] = tracked
        n_read += 1
        prog.step(n_read)
    cap.release()
    prog.done(f"{n_read}帧, {len(all_boxes)}帧有检测")
    return all_boxes, (w, h, fps, n_read)


def stage2_pose(all_boxes, projector, camera_videos, estimator,
                kpt_thr=KPT_THR, debug_dir=None):
    """
    每帧把画布框映射到最清晰相机、按相机分组批量回归关键点。

    补检：某目标在最清晰相机下的整体关键点置信度低于 kpt_thr（水花/遮挡/
    画面边缘裁切都会导致这种情况），改用次清晰相机对该目标单独再推一次；
    达标则采用新结果（坐标系同步切换到该相机），否则保留原结果——低置信度
    点在下游会被过滤，不需要额外的处理分支。只补检一次，不再往下扩展。

    返回 {tid: [(frame_idx, kpts(17,2), scores(17,), cam_idx)]}，坐标为该
    目标当帧所用相机的原图坐标系。
    """
    frames = sorted(all_boxes)
    prog = Progress("Stage2", frames[-1] if frames else 0)
    raw_seq, n_retry = {}, 0
    debug = TrackCropWriter(debug_dir) if debug_dir else None

    with MultiVideoReader(camera_videos) as reader:
        try:
            for fi in frames:
                # 每个目标按投影面积排序的候选相机；最清晰的那路按相机分组批量推理
                ranked, by_camera = {}, {}
                for tid, x1, y1, x2, y2, _conf in all_boxes[fi]:
                    cams = prog.timeit("geometry", projector.rank_cameras, (x1, y1, x2, y2))
                    if not cams:
                        continue
                    ranked[tid] = cams
                    by_camera.setdefault(cams[0][0], []).append((tid, cams[0][1]))

                low_conf = []       # [(tid, cam_idx, box, kpts, scores)]
                for cam_idx, entries in by_camera.items():
                    frame = prog.timeit("video_io", reader.read, cam_idx, fi)
                    if frame is None:
                        continue
                    results = prog.timeit("inference", estimator, frame,
                                          [box for _, box in entries])
                    for (tid, box), (kpts, scores) in zip(entries, results):
                        if float(scores.mean()) < kpt_thr:
                            low_conf.append((tid, cam_idx, box, kpts, scores))
                            continue
                        raw_seq.setdefault(tid, []).append((fi, kpts, scores, cam_idx))
                        if debug:
                            debug.write(tid, frame, box, kpts, scores, kpt_thr)

                for tid, cam_idx, box, kpts, scores in low_conf:
                    second = next((c for c in ranked[tid] if c[0] != cam_idx), None)
                    if second is not None:
                        t0 = time.time()
                        frame2 = reader.read(second[0], fi)
                        if frame2 is not None:
                            (kpts2, scores2), = estimator(frame2, [second[1]])
                            if float(scores2.mean()) >= kpt_thr:
                                cam_idx, box, kpts, scores = (second[0], second[1],
                                                              kpts2, scores2)
                                n_retry += 1
                        prog.timers["retry"] = prog.timers.get("retry", 0.0) + time.time() - t0
                    raw_seq.setdefault(tid, []).append((fi, kpts, scores, cam_idx))
                    if debug:
                        frame = reader.read(cam_idx, fi)
                        if frame is not None:
                            debug.write(tid, frame, box, kpts, scores, kpt_thr)

                prog.step(fi, 补检成功=n_retry)
        finally:
            if debug:
                debug.close()

    prog.done(f"{len(raw_seq)}个目标有关键点, 补检成功{n_retry}次")
    return raw_seq


def back_project_keypoints(raw_seq, interp_data, projector):
    """把插值后的关键点从各自相机的原图坐标反投影回画布坐标。

    interpolate_keypoints 不增删帧，所以每帧所用的相机可以直接从 raw_seq 取，
    不需要就近查找。返回 {tid: {frame_idx: (kpts_canvas(17,2), scores)}}。
    """
    canvas = {}
    for tid, frames in interp_data.items():
        cam_of = {fi: cam for fi, _, _, cam in raw_seq.get(tid, [])}
        canvas[tid] = {
            fi: (projector.source_points_to_canvas(cam_of[fi], kpts), scores)
            for fi, (kpts, scores) in frames.items() if fi in cam_of
        }
    return canvas


def stage4_render(canvas_video, out_path, all_boxes, cum_map, meta,
                  canvas_kpts=None, speed_map=None, kpt_thr=KPT_THR):
    """重读拼接视频并叠加标注，写出结果视频（只渲染 meta 里声明的帧数）。"""
    w, h, fps, total = meta
    cap = cv2.VideoCapture(canvas_video)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    prog = Progress("Stage4", total)

    for fi in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        for tid, x1, y1, x2, y2, _conf in all_boxes.get(fi, []):
            color = id_color(tid)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"ID:{tid} Strokes:{int(cum_map[tid][fi]) if tid in cum_map else 0}"
            speed = speed_map.get(tid) if speed_map else None
            if speed is not None and not np.isnan(speed[fi]):
                label += f" Speed:{speed[fi]:.2f}m/s"
            draw_label(frame, (x1, y1), label, color)

            entry = canvas_kpts.get(tid, {}).get(fi) if canvas_kpts else None
            if entry is not None:
                draw_skeleton(frame, *entry, thr=kpt_thr)

        writer.write(frame)
        prog.step(fi + 1)
    cap.release()
    writer.release()
    prog.done(f"-> {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="六路相机拼接全景视频的游泳划水计数与速度分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    g = p.add_argument_group("输入")
    g.add_argument("--canvas-video", required=True, help="拼接后的全景视频")
    g.add_argument("--camera-videos", nargs="+", required=True,
                   help="六路原始相机视频，顺序必须与 mesh 的 meshes 数组一致")
    g.add_argument("--mesh", default=os.path.join(root, "configs/pool_mesh.json"),
                   help="泳池 mesh JSON")

    g = p.add_argument_group("模型")
    g.add_argument("--yolo-model", default=os.path.join(root, "weights/yolo_swim_detect.pt"))
    g.add_argument("--pose-config",
                   default=os.path.join(root, "configs/rtmpose-m_swim-256x192.py"))
    g.add_argument("--pose-checkpoint", default=os.path.join(root, "weights/rtmpose_m_swim.pth"))
    g.add_argument("--device", default="0", help="GPU 序号")

    g = p.add_argument_group("几何")
    g.add_argument("--ppm", type=float, default=100.0, help="画布每米对应像素数")
    g.add_argument("--unit-scale", type=float, default=1.0, help="mesh 世界坐标单位换算")
    g.add_argument("--no-neg-v", dest="neg_v", action="store_false",
                   help="不翻转 mesh 的 V 轴（默认翻转）")

    g = p.add_argument_group("分析")
    g.add_argument("--stroke-type", default="freestyle",
                   choices=["freestyle", "backstroke", "butterfly", "breaststroke", "unknown"],
                   help="自由泳/仰泳左右手分别计数取较小值，其余泳姿左右取均值")
    g.add_argument("--signal", default="elbow_angle", choices=sorted(SIGNALS),
                   help="划水信号：肘角度波谷 或 手腕相对头部过零")
    g.add_argument("--kpt-thr", type=float, default=KPT_THR, help="关键点置信度阈值")
    g.add_argument("--max-frames", type=int, default=0,
                   help="只处理前 N 帧（0 表示全部），用于快速验证")

    g = p.add_argument_group("输出")
    g.add_argument("--output-dir", required=True)
    g.add_argument("--draw-keypoints", action="store_true", help="在结果视频上叠加骨架")
    g.add_argument("--signal-plots", action="store_true", help="每个目标存一张信号图")
    g.add_argument("--debug-crops", action="store_true",
                   help="每个目标存一个原视角裁剪+骨架的小视频")
    g.add_argument("--cache", default=None,
                   help="Stage1/2 结果缓存路径（默认 output_dir/cache.pkl），存在则复用")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.output_dir, "cache.pkl")

    if os.path.exists(cache_path):
        print(f"[Pipeline] 复用缓存 {cache_path}（删除该文件可强制重跑 Stage1/2）")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        all_boxes, raw_seq, meta = cache["all_boxes"], cache["raw_seq"], cache["meta"]
    else:
        all_boxes, meta = stage1_track(args.canvas_video, args.yolo_model,
                                       args.device, args.max_frames)
        projector = MultiCameraProjector(
            args.mesh, args.camera_videos, canvas_h=meta[1],
            ppm=args.ppm, unit_scale=args.unit_scale, neg_v=args.neg_v)
        raw_seq = stage2_pose(
            all_boxes, projector, args.camera_videos,
            RTMPoseEstimator(args.pose_config, args.pose_checkpoint, args.device),
            kpt_thr=args.kpt_thr,
            debug_dir=os.path.join(args.output_dir, "debug_crops")
            if args.debug_crops else None)
        with open(cache_path, "wb") as f:
            pickle.dump({"all_boxes": all_boxes, "raw_seq": raw_seq, "meta": meta}, f)
        print(f"[Pipeline] 已缓存 Stage1/2 结果 -> {cache_path}")

    _w, canvas_h, fps, total_frames = meta

    interp = interpolate_keypoints(raw_seq)
    strokes = count_strokes(interp, fps, args.stroke_type, args.signal)
    cum_map = cumulative_counts(strokes, total_frames)
    speed_samples, speed_map = compute_speed(all_boxes, fps, args.ppm, total_frames)

    summary = {
        str(tid): {"strokes": res["count"],
                   "speed_mps": [[round(t, 2), round(v, 3)] for t, v in speed_samples.get(tid, [])]}
        for tid, res in sorted(strokes.items())
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for tid, res in sorted(strokes.items()):
        speeds = [v for _, v in speed_samples.get(tid, [])]
        print(f"[Result] ID {tid}: 划水 {res['count']} 次"
              + (f", 均速 {np.mean(speeds):.2f} m/s, 峰值 {np.max(speeds):.2f} m/s"
                 if speeds else ""))

    if args.signal_plots:
        save_signal_plots(strokes, os.path.join(args.output_dir, "signal_plots"),
                          fps, args.stroke_type, args.signal)

    canvas_kpts = None
    if args.draw_keypoints:
        projector = MultiCameraProjector(
            args.mesh, args.camera_videos, canvas_h=canvas_h,
            ppm=args.ppm, unit_scale=args.unit_scale, neg_v=args.neg_v)
        canvas_kpts = back_project_keypoints(raw_seq, interp, projector)

    out_video = os.path.join(
        args.output_dir,
        os.path.splitext(os.path.basename(args.canvas_video))[0] + "_annotated.mp4")
    stage4_render(args.canvas_video, out_video, all_boxes, cum_map, meta,
                  canvas_kpts=canvas_kpts, speed_map=speed_map, kpt_thr=args.kpt_thr)
    print(f"[Pipeline] 全部完成，结果目录 {args.output_dir}")


if __name__ == "__main__":
    main()
