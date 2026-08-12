"""把 Plan C 的两个模型导出为 fp16 ONNX，供 TensorRT 构建 engine。

TensorRT 11 起强类型模式恒开、`BuilderFlag::kFP16` 已移除 —— 精度完全由 ONNX
自身的 dtype 决定，因此必须直接导出 fp16 的 ONNX，否则 engine 会跑 fp32。

两个模型的接口（C++ 侧按此对接）：

  detect  yolo26 (ultralytics)
      in   images     [1, 3, H, W]        fp16, RGB, /255, letterbox
      out  output0    [1, 4+nc, N]        cx,cy,w,h + 类别分数（未做 NMS）
  pose    RTMPose-m
      in   input      [B, 3, 256, 192]    fp16, RGB, ImageNet 归一化
      out  simcc_x    [B, 17, 384]        384 = 192 * simcc_split_ratio
           simcc_y    [B, 17, 512]        512 = 256 * simcc_split_ratio

pose 的 batch 维动态（1..40），detect 固定 batch=1（画布每帧只有一张）。

用法：
    python cpp/tools/export_onnx.py --out cpp/models
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))


def export_detect(out_dir, weights, imgsz, half):
    """ultralytics 自带导出器：它会处理 head 的解码与输出拼接。"""
    from ultralytics import YOLO
    m = YOLO(weights)
    path = m.export(format='onnx', imgsz=imgsz, half=half, simplify=True,
                    dynamic=False, opset=17, device=0)
    dst = os.path.join(out_dir, 'detect.onnx')
    os.replace(path, dst)
    print(f'[detect] {dst}  imgsz={imgsz} half={half}')
    return dst


class PoseWrap(torch.nn.Module):
    """只保留 backbone + head，剥掉 mmpose 的 data_preprocessor 与后处理。

    归一化在 C++ 的 CUDA kernel 里做（与 data_preprocessor 的 mean/std 一致），
    这样图像进 GPU 后无需回到 CPU。
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        simcc_x, simcc_y = self.head(self.backbone(x))
        return simcc_x, simcc_y


def export_pose(out_dir, config, ckpt, max_batch, half):
    from swim_analyse.pose import RTMPoseEstimator
    est = RTMPoseEstimator(config, ckpt, '0')
    net = PoseWrap(est.model).eval().cuda()
    if half:
        net = net.half()
    w, h = est.model.cfg.codec['input_size']          # (192, 256)
    dummy = torch.randn(1, 3, h, w, device='cuda', dtype=torch.half if half else torch.float)

    dst = os.path.join(out_dir, 'pose.onnx')
    torch.onnx.export(
        net, dummy, dst, opset_version=17,
        input_names=['input'], output_names=['simcc_x', 'simcc_y'],
        dynamic_axes={'input': {0: 'batch'},
                      'simcc_x': {0: 'batch'}, 'simcc_y': {0: 'batch'}})
    with torch.no_grad():
        sx, sy = net(dummy)
    print(f'[pose] {dst}  输入 [B,3,{h},{w}] half={half} maxBatch={max_batch}')
    print(f'[pose] 输出 simcc_x {tuple(sx.shape)}  simcc_y {tuple(sy.shape)}')
    # 供 C++ 侧对齐的元信息
    meta = dict(input_w=w, input_h=h, num_kpts=sx.shape[1],
                simcc_x=sx.shape[2], simcc_y=sy.shape[2],
                split_ratio=est.model.cfg.codec['simcc_split_ratio'],
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    return dst, meta


def main():
    p = argparse.ArgumentParser(description='Plan C 模型导出为 fp16 ONNX')
    p.add_argument('--out', default=os.path.join(ROOT, 'cpp/models'))
    p.add_argument('--detect-weights', default=os.path.join(ROOT, 'weights/yolo_swim_detect.pt'))
    p.add_argument('--detect-imgsz', type=int, default=640,
                   help='必须与检测器训练时的 imgsz 一致（yolo_swim_detect 是 640）。'
                        '推理期偏离训练尺度会显著掉召回：实测 1920 输入下每帧只检出 '
                        '7.5 人，而 640 是 12 人')
    p.add_argument('--pose-config', default=os.path.join(ROOT, 'configs/rtmpose-m_canvas-192x256.py'))
    p.add_argument('--pose-ckpt', default=os.path.join(ROOT, 'weights/plans/planC_rtmpose_m_canvas.pth'))
    p.add_argument('--max-batch', type=int, default=40, help='pose 的最大人数')
    p.add_argument('--fp32', action='store_true', help='导出 fp32（默认 fp16）')
    p.add_argument('--only', choices=['detect', 'pose'], help='只导其中一个')
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    half = not a.fp32
    if a.only != 'pose':
        export_detect(a.out, a.detect_weights, a.detect_imgsz, half)
    if a.only != 'detect':
        _dst, meta = export_pose(a.out, a.pose_config, a.pose_ckpt, a.max_batch, half)
        import json
        with open(os.path.join(a.out, 'pose_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'[meta] {a.out}/pose_meta.json')


if __name__ == '__main__':
    main()
