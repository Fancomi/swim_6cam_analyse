#include "swim/kernels.h"

namespace swim {
namespace {

constexpr int kBlock = 256;
inline int grid(int n, int b = kBlock) { return (n + b - 1) / b; }

// device 侧不能访问 host 的 std::array 常量，这里复制一份（值与 common.h 一致）
__device__ __constant__ float d_mean[3] = {123.675f, 116.28f, 103.53f};
__device__ __constant__ float d_std[3]  = {58.395f, 57.12f, 57.375f};

/// 双线性采样 BGR uint8，越界返回 fill。通道序返回 RGB。
__device__ inline void sample_rgb(const uint8_t* src, int w, int h,
                                  float fx, float fy, float fill, float out[3]) {
  if (fx < 0.f || fy < 0.f || fx > w - 1.f || fy > h - 1.f) {
    out[0] = out[1] = out[2] = fill;
    return;
  }
  const int   x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
  const int   x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
  const float ax = fx - x0, ay = fy - y0;
  const uint8_t* p00 = src + (static_cast<size_t>(y0) * w + x0) * 3;
  const uint8_t* p01 = src + (static_cast<size_t>(y0) * w + x1) * 3;
  const uint8_t* p10 = src + (static_cast<size_t>(y1) * w + x0) * 3;
  const uint8_t* p11 = src + (static_cast<size_t>(y1) * w + x1) * 3;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float top = p00[c] * (1 - ax) + p01[c] * ax;
    const float bot = p10[c] * (1 - ax) + p11[c] * ax;
    out[2 - c] = top * (1 - ay) + bot * ay;    // BGR -> RGB
  }
}

__global__ void k_preprocess(const uint8_t* src, int sw, int sh,
                             __half* dst, int size, Letterbox lb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size * size) return;
  const int x = i % size, y = i / size;
  float rgb[3];
  sample_rgb(src, sw, sh, lb.to_src_x(x + 0.5f) - 0.5f,
             lb.to_src_y(y + 0.5f) - 0.5f, 114.f, rgb);
  const int plane = size * size;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    dst[c * plane + i] = __float2half(rgb[c] * (1.f / 255.f));
}

/// mmpose GetBBoxCenterScale：框补成 pose 输入的宽高比，再乘 padding。
__device__ inline void box_to_window(float x1, float y1, float x2, float y2,
                                     float& cx, float& cy, float& sw, float& sh) {
  cx = (x1 + x2) * 0.5f;
  cy = (y1 + y2) * 0.5f;
  float w = x2 - x1, h = y2 - y1;
  const float aspect = static_cast<float>(kPoseW) / kPoseH;
  if (w > h * aspect) h = w / aspect; else w = h * aspect;
  sw = w * kBoxPadding;
  sh = h * kBoxPadding;
}

__global__ void k_crop_affine(const uint8_t* src, int sw, int sh,
                              const float* boxes, int n,
                              __half* dst, float* centers, float* scales) {
  const int total = n * kPoseH * kPoseW;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int px = i % kPoseW;
  const int py = (i / kPoseW) % kPoseH;
  const int b  = i / (kPoseW * kPoseH);

  const float* bx = boxes + b * 4;
  float cx, cy, wwin, hwin;
  box_to_window(bx[0], bx[1], bx[2], bx[3], cx, cy, wwin, hwin);
  // 采样窗左上角 -> 源坐标（窗口尺寸 wwin×hwin 映射到 kPoseW×kPoseH）
  const float fx = cx - wwin * 0.5f + (px + 0.5f) * wwin / kPoseW;
  const float fy = cy - hwin * 0.5f + (py + 0.5f) * hwin / kPoseH;

  float rgb[3];
  sample_rgb(src, sw, sh, fx - 0.5f, fy - 0.5f, 0.f, rgb);
  const int plane = kPoseH * kPoseW;
  const int off   = b * 3 * plane + py * kPoseW + px;
#pragma unroll
  for (int c = 0; c < 3; ++c)
    dst[off + c * plane] = __float2half((rgb[c] - d_mean[c]) / d_std[c]);

  if (px == 0 && py == 0) {
    centers[b * 2] = cx;      centers[b * 2 + 1] = cy;
    scales[b * 2]  = wwin;    scales[b * 2 + 1]  = hwin;
  }
}

/// 峰值 ±1 的二次插值求亚像素（与 mmpose 的 dark 解码同思路，精度足够）。
__device__ inline float refine_peak(const float* p, int a, int len) {
  if (a <= 0 || a >= len - 1) return static_cast<float>(a);
  const float l = p[a - 1], c = p[a], r = p[a + 1];
  const float d = l - 2.f * c + r;
  return fabsf(d) < 1e-9f ? static_cast<float>(a) : a + 0.5f * (l - r) / d;
}

/// 一个 block 处理一个 (person, keypoint)：先对 x 轴、再对 y 轴求 argmax。
__global__ void k_simcc_decode(const float* sx, const float* sy,
                               int n, int K, int wx, int wy,
                               const float* centers, const float* scales,
                               float* kpts, float* scores) {
  const int idx = blockIdx.x;                 // person*K + k
  if (idx >= n * K) return;
  const int b = idx / K;

  // 分别在两条轴上找峰值（长度 wx/wy 均 <= 1024，单线程扫描足够快且无需同步）
  if (threadIdx.x != 0) return;
  float best_x = -1e30f, best_y = -1e30f;
  int   arg_x = 0, arg_y = 0;
  const float* px = sx + static_cast<size_t>(idx) * wx;
  const float* py = sy + static_cast<size_t>(idx) * wy;
  for (int i = 0; i < wx; ++i) if (px[i] > best_x) { best_x = px[i]; arg_x = i; }
  for (int i = 0; i < wy; ++i) if (py[i] > best_y) { best_y = py[i]; arg_y = i; }

  const float fx = refine_peak(px, arg_x, wx) / kSimccRatio;   // pose 输入坐标
  const float fy = refine_peak(py, arg_y, wy) / kSimccRatio;

  // pose 输入坐标 -> 画布坐标
  const float cx = centers[b * 2], cy = centers[b * 2 + 1];
  const float ww = scales[b * 2],  wh = scales[b * 2 + 1];
  kpts[idx * 2]     = cx - ww * 0.5f + fx * ww / kPoseW;
  kpts[idx * 2 + 1] = cy - wh * 0.5f + fy * wh / kPoseH;
  scores[idx] = fminf(best_x, best_y);
}

__global__ void k_filter_boxes(const float* det, int max_det, float thr,
                               Letterbox lb, int sw, int sh, int max_keep,
                               float* boxes, float* conf, int* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_det) return;
  const float* d = det + i * 6;               // x1,y1,x2,y2,conf,cls
  if (d[4] < thr) return;
  const int k = atomicAdd(count, 1);
  if (k >= max_keep) return;                  // 超出预留则丢弃（保持显存恒定）
  float* b = boxes + k * 4;
  b[0] = fmaxf(0.f, fminf(lb.to_src_x(d[0]), sw - 1.f));
  b[1] = fmaxf(0.f, fminf(lb.to_src_y(d[1]), sh - 1.f));
  b[2] = fmaxf(0.f, fminf(lb.to_src_x(d[2]), sw - 1.f));
  b[3] = fmaxf(0.f, fminf(lb.to_src_y(d[3]), sh - 1.f));
  conf[k] = d[4];
}

/// 包含率去重 + 就地压缩。单线程：n<=40，O(n²) 仅约 1600 次比较，
/// 且要严格复刻 Python filter_contained_boxes 的顺序语义（含 break）。
__global__ void k_dedup_boxes(float* boxes, float* conf, int* count, float thr) {
  if (threadIdx.x || blockIdx.x) return;
  const int n = min(*count, kMaxPersons);
  if (n <= 1) { *count = n; return; }

  bool keep[kMaxPersons];
  for (int i = 0; i < n; ++i) keep[i] = true;

  for (int i = 0; i < n; ++i) {
    if (!keep[i]) continue;
    const float* a = boxes + i * 4;
    const float area_a = fmaxf((a[2] - a[0]) * (a[3] - a[1]), 1e-6f);
    for (int j = 0; j < n; ++j) {
      if (i == j || !keep[j]) continue;
      const float* b = boxes + j * 4;
      const float ix = fmaxf(0.f, fminf(a[2], b[2]) - fmaxf(a[0], b[0]));
      const float iy = fmaxf(0.f, fminf(a[3], b[3]) - fmaxf(a[1], b[1]));
      if (ix * iy / area_a < thr) continue;     // i 被 j 包住的比例
      if (conf[i] > conf[j]) {
        keep[j] = false;
      } else {
        keep[i] = false;
        break;                                  // i 已被弃，不必再比
      }
    }
  }
  // 就地前移压缩（保序，与 Python 一致）
  int w = 0;
  for (int i = 0; i < n; ++i) {
    if (!keep[i]) continue;
    if (w != i) {
      for (int k = 0; k < 4; ++k) boxes[w * 4 + k] = boxes[i * 4 + k];
      conf[w] = conf[i];
    }
    ++w;
  }
  *count = w;
}

}  // namespace

void launch_preprocess(const uint8_t* src, int sw, int sh, __half* dst,
                       int size, Letterbox lb, cudaStream_t s) {
  const int n = size * size;
  k_preprocess<<<grid(n), kBlock, 0, s>>>(src, sw, sh, dst, size, lb);
}

void launch_crop_affine(const uint8_t* src, int sw, int sh, const float* boxes,
                        int n, __half* dst, float* centers, float* scales,
                        cudaStream_t s) {
  if (n <= 0) return;
  const int total = n * kPoseH * kPoseW;
  k_crop_affine<<<grid(total), kBlock, 0, s>>>(src, sw, sh, boxes, n, dst,
                                               centers, scales);
}

void launch_simcc_decode(const float* sx, const float* sy, int n, int K,
                         int wx, int wy, const float* centers,
                         const float* scales, float* kpts, float* scores,
                         cudaStream_t s) {
  if (n <= 0) return;
  k_simcc_decode<<<n * K, 32, 0, s>>>(sx, sy, n, K, wx, wy, centers, scales,
                                      kpts, scores);
}

void launch_filter_boxes(const float* det, int max_det, float thr, Letterbox lb,
                         int sw, int sh, int max_keep, float* boxes,
                         float* conf, int* count, cudaStream_t s) {
  SWIM_CUDA(cudaMemsetAsync(count, 0, sizeof(int), s));
  k_filter_boxes<<<grid(max_det), kBlock, 0, s>>>(det, max_det, thr, lb, sw, sh,
                                                  max_keep, boxes, conf, count);
}

void launch_dedup_boxes(float* boxes, float* conf, int* count, float thresh,
                        cudaStream_t s) {
  k_dedup_boxes<<<1, 1, 0, s>>>(boxes, conf, count, thresh);
}

}  // namespace swim
