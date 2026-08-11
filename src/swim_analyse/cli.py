"""
六路相机游泳分析统一入口：三套关键点方案（--plan A/B/C）共用同一套下游。

    --plan A   画布 detect + 原相机 RTMPose + mesh 反投影（交接原版）
    --plan B   画布 yolo26-pose 一体（detect+pose 单次前向）
    --plan C   画布 detect + 画布 RTMPose（两阶段，推荐）

三套方案只在"如何得到每人的框与关键点"上不同（见 plans.py），之后的
关键点插值、划水计数、速度估计、标注渲染完全共用 —— 因此换方案不影响
下游任何逻辑，指标也可直接对比。

    Stage1/2  由 plan 负责：-> all_boxes, raw_seq
    Stage3    关键点插值 -> 划水次数（肘角度波谷 / 手腕过零）+ 瞬时速度
    Stage4    叠加框/标签/骨架，H264 编码输出

Stage1/2 的结果缓存到 output_dir/cache.pkl（含 plan 名），重跑时若缓存存在
且 plan 一致则直接跳到 Stage3 —— 调信号参数、换绘制选项都不必重跑 GPU。

用法见 run.sh，或 `python -m swim_analyse.cli --help`。
"""

import argparse
import json
import os
import pickle
import subprocess
import time

import cv2
import numpy as np

from .draw import draw_label, draw_skeleton, id_color
from .metrics import (SIGNALS, compute_speed, count_strokes, cumulative_counts,
                      save_signal_plots)
from .plans import PLANS
from .pose import KPT_THR, interpolate_keypoints
from .video import video_meta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="六路相机游泳分析（三套关键点方案统一入口）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("方案")
    g.add_argument("--plan", required=True, choices=sorted(PLANS),
                   help="A=原相机RTMPose(交接版)  B=画布yolo-pose一体  C=画布两阶段")

    g = p.add_argument_group("输入")
    g.add_argument("--canvas-video", required=True, help="拼接后的全景视频")
    g.add_argument("--camera-videos", nargs="+", default=None,
                   help="六路原始相机视频（仅 Plan A 需要），顺序须与 mesh 一致")
    g.add_argument("--mesh", default=os.path.join(ROOT, "configs/pool_mesh.json"))
    g.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")

    g = p.add_argument_group("模型")
    g.add_argument("--yolo-model", default=os.path.join(ROOT, "weights/yolo_swim_detect.pt"),
                   help="检测权重（Plan A/C）")
    g.add_argument("--pose-model", default=None,
                   help="yolo-pose 权重（Plan B）")
    g.add_argument("--pose-config", default=None,
                   help="RTMPose config（Plan A/C）")
    g.add_argument("--pose-checkpoint", default=None,
                   help="RTMPose 权重（Plan A/C）")
    g.add_argument("--device", default="0")

    g = p.add_argument_group("检测与跟踪")
    g.add_argument("--conf", type=float, default=0.25)
    g.add_argument("--iou", type=float, default=0.3, help="检测 NMS 的 IoU")
    g.add_argument("--containment", type=float, default=0.7,
                   help="包含率去重阈值：交集/自身面积 超过此值视为重复框")
    g.add_argument("--track-iou", type=float, default=0.3)
    g.add_argument("--track-max-lost", type=int, default=30)

    g = p.add_argument_group("几何（Plan A）")
    g.add_argument("--ppm", type=float, default=100.0, help="画布每米对应像素数")
    g.add_argument("--unit-scale", type=float, default=1.0)
    g.add_argument("--no-neg-v", dest="neg_v", action="store_false",
                   help="不翻转 mesh 的 V 轴（默认翻转）")

    g = p.add_argument_group("分析")
    g.add_argument("--stroke-type", default="freestyle",
                   choices=["freestyle", "backstroke", "butterfly", "breaststroke", "unknown"])
    g.add_argument("--signal", default="elbow_angle", choices=sorted(SIGNALS))
    g.add_argument("--kpt-thr", type=float, default=KPT_THR)

    g = p.add_argument_group("输出")
    g.add_argument("--output-dir", required=True)
    g.add_argument("--draw-keypoints", action="store_true")
    g.add_argument("--signal-plots", action="store_true")
    g.add_argument("--codec", default="h264", choices=["h264", "mp4v"])
    g.add_argument("--cache", default=None)

    a = p.parse_args(argv)
    _check(p, a)
    return a


def _check(parser, a):
    """按 plan 校验必需参数，避免跑到一半才报错。"""
    if a.plan == "A":
        if not a.camera_videos:
            parser.error("Plan A 需要 --camera-videos（六路原相机视频）")
        if not (a.pose_config and a.pose_checkpoint):
            parser.error("Plan A 需要 --pose-config 与 --pose-checkpoint")
    elif a.plan == "B":
        if not a.pose_model:
            parser.error("Plan B 需要 --pose-model（yolo-pose 权重）")
    elif a.plan == "C":
        if not (a.pose_config and a.pose_checkpoint):
            parser.error("Plan C 需要 --pose-config 与 --pose-checkpoint（画布 RTMPose）")


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.output_dir, "cache.pkl")

    w, h, fps, total = video_meta(args.canvas_video)
    if args.max_frames:
        total = min(total, args.max_frames)
    plan = PLANS[args.plan](args, canvas_h=h)
    print(f"[Plan {args.plan}] {plan.description}")
    print(f"[Meta] {w}x{h} {fps:.2f}fps 处理 {total} 帧 -> {args.canvas_video}")

    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("plan") != args.plan:
            print(f"[Cache] 缓存来自 Plan {cached.get('plan')}，与当前 Plan "
                  f"{args.plan} 不符，重新计算")
            cached = None

    if cached:
        print(f"[Cache] 复用 {cache_path}（删除该文件可强制重跑）")
        all_boxes, raw_seq = cached["all_boxes"], cached["raw_seq"]
        if args.plan == "A":      # to_canvas 需要 projector，重建
            from .geometry import MultiCameraProjector
            plan.projector = MultiCameraProjector(
                args.mesh, args.camera_videos, canvas_h=h,
                ppm=args.ppm, unit_scale=args.unit_scale, neg_v=args.neg_v)
    else:
        t0 = time.time()
        all_boxes, raw_seq = plan.run(args.canvas_video, total)
        plan.report(time.time() - t0, total)
        with open(cache_path, "wb") as f:
            pickle.dump({"plan": args.plan, "all_boxes": all_boxes,
                         "raw_seq": raw_seq, "meta": (w, h, fps, total)}, f)
        print(f"[Cache] 已写 {cache_path}")
    print(f"[Stage1/2] {len(all_boxes)} 帧有检测，{len(raw_seq)} 个 track 有关键点")

    # ── Stage3 划水 + 速度 ──────────────────────────────────────────────────
    interp = interpolate_keypoints(raw_seq)
    strokes = count_strokes(interp, fps, args.stroke_type, args.signal)
    cum_map = cumulative_counts(strokes, total)
    speed_samples, speed_map = compute_speed(all_boxes, fps, args.ppm, total)

    summary = {
        str(tid): {"strokes": res["count"],
                   "speed_mps": [[round(t, 2), round(v, 3)]
                                 for t, v in speed_samples.get(tid, [])]}
        for tid, res in sorted(strokes.items())
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for tid, res in sorted(strokes.items()):
        sp = [v for _, v in speed_samples.get(tid, [])]
        print(f"[Result] ID {tid}: 划水 {res['count']} 次"
              + (f", 均速 {np.mean(sp):.2f} m/s, 峰值 {np.max(sp):.2f} m/s" if sp else ""))
    print(f"[Stage3] {len(strokes)} 个 track，划水合计 "
          f"{sum(r['count'] for r in strokes.values())} 次")

    if args.signal_plots:
        save_signal_plots(strokes, os.path.join(args.output_dir, "signal_plots"),
                          fps, args.stroke_type, args.signal)

    # ── Stage4 渲染 ─────────────────────────────────────────────────────────
    canvas_kpts = plan.to_canvas(raw_seq, interp) if args.draw_keypoints else None
    out_video = os.path.join(
        args.output_dir,
        f"{os.path.splitext(os.path.basename(args.canvas_video))[0]}_plan{args.plan}.mp4")
    render(args.canvas_video, out_video, all_boxes, cum_map, (w, h, fps, total),
           canvas_kpts=canvas_kpts, speed_map=speed_map,
           kpt_thr=args.kpt_thr, codec=args.codec)
    print(f"[Done] Plan {args.plan} 完成 -> {args.output_dir}")


def render(canvas_video, out_path, all_boxes, cum_map, meta, canvas_kpts=None,
           speed_map=None, kpt_thr=KPT_THR, codec="h264"):
    """
    叠加标注并写出视频。

    codec='h264' 走 ffmpeg libx264（文件约为 mp4v 的一半，画质更好，且
    浏览器/播放器兼容性好）；imageio-ffmpeg 不可用时自动回退 opencv mp4v。
    """
    w, h, fps, total = meta
    cap = cv2.VideoCapture(canvas_video)
    proc = writer = None
    if codec == "h264":
        try:
            import imageio_ffmpeg
            proc = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", out_path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except ImportError:
            print("[Render] 未安装 imageio-ffmpeg，回退 mp4v")
    if proc is None:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    t0 = time.time()
    try:
        for fi in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            for tid, x1, y1, x2, y2, _c in all_boxes.get(fi, []):
                color = id_color(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{tid} Strokes:{int(cum_map[tid][fi]) if tid in cum_map else 0}"
                spd = speed_map.get(tid) if speed_map else None
                if spd is not None and not np.isnan(spd[fi]):
                    label += f" Speed:{spd[fi]:.2f}m/s"
                draw_label(frame, (x1, y1), label, color)
                if canvas_kpts is not None:
                    entry = canvas_kpts.get(tid, {}).get(fi)
                    if entry is not None:
                        draw_skeleton(frame, *entry, thr=kpt_thr)
            if proc:
                proc.stdin.write(frame.tobytes())
            else:
                writer.write(frame)
            if (fi + 1) % 500 == 0:
                print(f"[Stage4] {fi+1}/{total} 已耗时 {time.time()-t0:.0f}s", flush=True)
    finally:
        cap.release()
        if proc:
            proc.stdin.close()
            proc.wait()
        else:
            writer.release()
    print(f"[Stage4] 完成 耗时 {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
