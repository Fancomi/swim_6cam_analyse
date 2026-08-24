// CUDA kernel：全链路 GPU 驻留的三个关键环节。
//
//   1 preprocess   画布 BGR uint8 -> detect 输入（RGB fp16, /255, letterbox）
//   2 crop_affine  按框裁切 -> pose 输入（RGB fp16, ImageNet 归一化, batch 拼接）
//   3 simcc_decode SimCC logits -> 关键点坐标（argmax + 亚像素）
//
// 之所以必须自己写这三个：把它们放 CPU 会引入每帧 30 MB 的 D2H+H2D，
// 直接吃掉 TensorRT 省下的时间。
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "swim/common.h"

namespace swim {

/// 画布 letterbox 到方形 detect 输入的几何参数（CPU 侧算一次，供还原坐标用）。
struct Letterbox {
  float scale = 1.f;   // 原图 -> 输入 的缩放
  int   pad_x = 0;     // 左侧填充（像素，输入坐标系）
  int   pad_y = 0;     // 顶部填充

  static Letterbox make(int src_w, int src_h, int dst) {
    Letterbox lb;
    lb.scale = static_cast<float>(dst) / (src_w > src_h ? src_w : src_h);
    lb.pad_x = (dst - static_cast<int>(src_w * lb.scale + 0.5f)) / 2;
    lb.pad_y = (dst - static_cast<int>(src_h * lb.scale + 0.5f)) / 2;
    return lb;
  }
  /// detect 输入坐标 -> 画布坐标
  __host__ __device__ float to_src_x(float x) const { return (x - pad_x) / scale; }
  __host__ __device__ float to_src_y(float y) const { return (y - pad_y) / scale; }
};

/// 1) BGR uint8 [h,w,3] -> RGB fp16 [1,3,dst,dst]，双线性 + letterbox（填充 114）。
void launch_preprocess(const uint8_t* src, int src_w, int src_h,
                       __half* dst, int dst_size, Letterbox lb,
                       cudaStream_t stream);

/// 2) 按框裁切到 pose 输入。复刻 mmpose GetBBoxCenterScale + TopdownAffine：
///    框先补成 kPoseW:kPoseH 的宽高比，再乘 kBoxPadding，然后仿射采样。
///    boxes_d: [n,4] float(x1,y1,x2,y2) 画布坐标；dst: [n,3,kPoseH,kPoseW] fp16。
///    同时输出每个框的采样窗中心与尺度（供关键点还原到画布坐标）。
void launch_crop_affine(const uint8_t* src, int src_w, int src_h,
                        const float* boxes_d, int n,
                        __half* dst, float* centers_d, float* scales_d,
                        cudaStream_t stream);

/// 3) SimCC 解码。simcc_x:[n,K,Wx] simcc_y:[n,K,Wy]（fp32），
///    取各轴 argmax，用峰值±1 邻域做加权求亚像素，再按 centers/scales
///    还原到画布坐标。输出 kpts_d:[n,K,2]、scores_d:[n,K]。
void launch_simcc_decode(const float* simcc_x, const float* simcc_y,
                         int n, int num_kpts, int wx, int wy,
                         const float* centers_d, const float* scales_d,
                         float* kpts_d, float* scores_d, cudaStream_t stream);

/// 辅助：把 detect 的 end2end 输出 [1,max_det,6](x1,y1,x2,y2,conf,cls)
/// 按 conf 阈值筛选并映射回画布坐标，写入紧凑数组 boxes_d[n,4]+conf_d[n]。
/// n 写入 count_d（device 上一个 int，已按 max_keep 夹紧），调用方读回后决定
/// pose 的 batch。名次由前缀计数确定，与 det 原始顺序一致（无原子操作，可复现）。
void launch_filter_boxes(const float* det, int max_det, float conf_thr,
                         Letterbox lb, int src_w, int src_h, int max_keep,
                         float* boxes_d, float* conf_d, int* count_d,
                         cudaStream_t stream);

/// 包含率去重：丢弃"几乎完全被另一个框包住"的重复检测（保留 conf 高的）。
///
/// 用 交集/自身面积 而非 IoU —— 同一个泳者常被同时框出一大一小两个框，
/// 这种情况 IoU 偏低会躲过 yolo26 的 end2end NMS，但包含率接近 1。
/// 残留的重复框会在跟踪时抢占 IoU 匹配、迫使新建 ID：实测不做此去重，
/// 1000 帧的 track 数从 36 涨到 163（4.5 倍），表现为框"一下有一下没"。
///
/// cap = 调用方为 boxes_d/conf_d 预留的容量（--max-persons），用于夹紧 count_d。
/// 就地压缩 boxes_d/conf_d 并更新 count_d。n<=cap<=40，单线程串行即可
/// （严格复刻 Python filter_contained_boxes 的顺序语义，含提前 break）。
void launch_dedup_boxes(float* boxes_d, float* conf_d, int* count_d, int cap,
                        float thresh, cudaStream_t stream);

}  // namespace swim
