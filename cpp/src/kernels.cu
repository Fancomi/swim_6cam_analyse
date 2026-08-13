#include "swim/kernels.h"

namespace swim {
namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;      // k_simcc_decode 的 blockDim，必须与之一致
inline int grid(int n, int b = kBlock) { return (n + b - 1) / b; }

// ── 本文件的检查宏（文案必须 ASCII）───────────────────────────────────────
// nvcc 的前端（EDG）在中文 ACP(936) 机器上按 ANSI 解 .cu 源码，UTF-8 的中文
// **字面量**会被拆成非法字节序列并吞掉右引号，报 `missing closing quote`；
// -Xcompiler=/utf-8 只作用于 cl，加 BOM 也无效（实测「的/须/为/正/查/失/败」
// 均触发）。中文**注释**不受影响，因此本文件注释照旧中文、字面量一律 ASCII，
// 也因此不能在 .cu 里用 common.h 的 SWIM_CHECK（它的前缀是中文）。
#define KCHECK(cond, msg)                                                     \
  do {                                                                        \
    if (!(cond)) throw std::runtime_error(std::string("kernels: ") + (msg));   \
  } while (0)

/// kernel 启动后立刻取错误：配置错误（grid/block 非法、共享内存超限）在这里就能
/// 暴露，否则要等到下一次同步才报，且错误现场已丢失。
#define KLAUNCHED(name) KCHECK(cudaPeekAtLastError() == cudaSuccess,           \
    std::string(name) + " launch failed: " + cudaGetErrorString(cudaGetLastError()))

// device 侧不能访问 host 的 constexpr 聚合，归一化参数只在这里定义一份
// （值与 RTMPose data_preprocessor 一致，RGB 顺序）
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
  // 采样窗 -> 源坐标，用**索引约定**（输出下标 px 直接映射，不加半像素）：
  // mmpose 的 TopdownAffine 走 cv2.warpAffine，而 warpAffine 就是把整数目标
  // 下标代进逆矩阵采样。这条也正是 k_simcc_decode 里逆映射的逆，两者严格互逆
  // （若这里改用半像素约定，窗宽偏离 kPoseW 时会留下 0.5*(wwin/kPoseW-1) 的
  // 系统性偏移，且与训练时的裁切不一致）。preprocess 那边是 cv2.resize 语义，
  // 半像素才对 —— 两个 kernel 的约定不同是有意的。
  const float fx = cx - wwin * 0.5f + px * wwin / kPoseW;
  const float fy = cy - hwin * 0.5f + py * hwin / kPoseH;

  float rgb[3];
  sample_rgb(src, sw, sh, fx, fy, 0.f, rgb);
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

/// 一个 warp 处理一个 (person, keypoint)：两条轴各做一次跨步扫描 + warp 归约。
/// 平票取小下标，与单线程顺序扫描（严格 >）的 argmax 结果逐值一致。
/// 步长与归约都用编译期常量 kWarp（而非内置 warpSize），归约循环才能真正展开。
__device__ inline void warp_argmax(const float* p, int len, float& best, int& arg) {
  best = -1e30f;
  arg  = 0;
  for (int i = threadIdx.x; i < len; i += kWarp)
    if (p[i] > best) { best = p[i]; arg = i; }
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    const float v = __shfl_down_sync(0xffffffffu, best, off);
    const int   a = __shfl_down_sync(0xffffffffu, arg, off);
    if (v > best || (v == best && a < arg)) { best = v; arg = a; }
  }
}

__global__ void k_simcc_decode(const float* sx, const float* sy,
                               int n, int K, int wx, int wy,
                               const float* centers, const float* scales,
                               float* kpts, float* scores) {
  const int idx = blockIdx.x;                 // person*K + k
  if (idx >= n * K) return;
  const int b = idx / K;

  const float* px = sx + static_cast<size_t>(idx) * wx;
  const float* py = sy + static_cast<size_t>(idx) * wy;
  float best_x, best_y;
  int   arg_x, arg_y;
  warp_argmax(px, wx, best_x, arg_x);
  warp_argmax(py, wy, best_y, arg_y);
  if (threadIdx.x != 0) return;               // 归约结果在 lane 0

  const float fx = refine_peak(px, arg_x, wx) / kSimccRatio;   // pose 输入坐标
  const float fy = refine_peak(py, arg_y, wy) / kSimccRatio;

  // pose 输入坐标 -> 画布坐标
  const float cx = centers[b * 2], cy = centers[b * 2 + 1];
  const float ww = scales[b * 2],  wh = scales[b * 2 + 1];
  kpts[idx * 2]     = cx - ww * 0.5f + fx * ww / kPoseW;
  kpts[idx * 2 + 1] = cy - wh * 0.5f + fy * wh / kPoseH;
  scores[idx] = fminf(best_x, best_y);
}

/// end2end 输出 -> 紧凑框数组。名次由"前面通过阈值的个数"决定：无原子操作，
/// 结果与 det 的原始顺序严格一致（可复现），也天然复刻 Python 的先后语义。
///
/// 判据必须与前缀计数完全同式，否则名次与写入不自洽 —— 故抽成 device 函数。
/// conf > 0 这一条不可省：max_det=300 走的是 topk，真实目标不足时余下槽位是
/// 极低分候选（常量恰为 0）。--conf 0 合法（validate 允许 [0,1)），若只判
/// >= thr，300 个空槽会全部"通过"，尾部塞满零面积框：dedup 去不掉（交集/1e-6
/// 恒为 0），tracker 也永不匹配（并集面积 0 -> IoU 0），于是每帧新建 40 个
/// track，人次与 summary 单调膨胀。
__device__ inline bool det_pass(const float* det, int i, float thr) {
  const float c = det[i * 6 + 4];
  return c > 0.f && c >= thr;
}

__global__ void k_filter_boxes(const float* det, int max_det, float thr,
                               Letterbox lb, int sw, int sh, int max_keep,
                               float* boxes, float* conf, int* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= max_det) return;
  const float* d = det + i * 6;               // x1,y1,x2,y2,conf,cls
  const bool pass = det_pass(det, i, thr);

  int k = 0;
  for (int j = 0; j < i; ++j) if (det_pass(det, j, thr)) ++k;
  // 末尾线程统一写总数并按预留量夹紧：下游只会看到 <= max_keep，无需再 clamp
  if (i == max_det - 1) *count = min(k + (pass ? 1 : 0), max_keep);

  if (!pass || k >= max_keep) return;         // 超出预留则丢弃（保持显存恒定）
  float* b = boxes + k * 4;
  b[0] = fmaxf(0.f, fminf(lb.to_src_x(d[0]), sw - 1.f));
  b[1] = fmaxf(0.f, fminf(lb.to_src_y(d[1]), sh - 1.f));
  b[2] = fmaxf(0.f, fminf(lb.to_src_x(d[2]), sw - 1.f));
  b[3] = fmaxf(0.f, fminf(lb.to_src_y(d[3]), sh - 1.f));
  conf[k] = d[4];
}

/// 包含率去重 + 就地压缩。单线程：n<=cap<=40，O(n²) 仅约 1600 次比较，
/// 且要严格复刻 Python filter_contained_boxes 的顺序语义（含 break）。
/// cap 是调用方为 boxes/conf 预留的容量（--max-persons），不能用编译期上限代替，
/// 否则 cap<40 时 keep[] 与 boxes[] 都会越界（曾表现为 0xC0000409 直接崩进程）。
__global__ void k_dedup_boxes(float* boxes, float* conf, int* count, int cap,
                              float thr) {
  if (threadIdx.x || blockIdx.x) return;
  const int n = min(min(*count, cap), kMaxPersons);
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
  KLAUNCHED("k_preprocess");
}

void launch_crop_affine(const uint8_t* src, int sw, int sh, const float* boxes,
                        int n, __half* dst, float* centers, float* scales,
                        cudaStream_t s) {
  if (n <= 0) return;
  const int total = n * kPoseH * kPoseW;
  k_crop_affine<<<grid(total), kBlock, 0, s>>>(src, sw, sh, boxes, n, dst,
                                               centers, scales);
  KLAUNCHED("k_crop_affine");
}

void launch_simcc_decode(const float* sx, const float* sy, int n, int K,
                         int wx, int wy, const float* centers,
                         const float* scales, float* kpts, float* scores,
                         cudaStream_t s) {
  if (n <= 0) return;
  k_simcc_decode<<<n * K, kWarp, 0, s>>>(sx, sy, n, K, wx, wy, centers, scales,
                                         kpts, scores);
  KLAUNCHED("k_simcc_decode");
}

void launch_filter_boxes(const float* det, int max_det, float thr, Letterbox lb,
                         int sw, int sh, int max_keep, float* boxes,
                         float* conf, int* count, cudaStream_t s) {
  KCHECK(max_det > 0 && max_keep > 0, "filter_boxes: max_det/max_keep must be positive");
  k_filter_boxes<<<grid(max_det), kBlock, 0, s>>>(det, max_det, thr, lb, sw, sh,
                                                  max_keep, boxes, conf, count);
  KLAUNCHED("k_filter_boxes");
}

void launch_dedup_boxes(float* boxes, float* conf, int* count, int cap,
                        float thresh, cudaStream_t s) {
  KCHECK(cap > 0 && cap <= kMaxPersons, "dedup_boxes: cap must be within 1..kMaxPersons");
  k_dedup_boxes<<<1, 1, 0, s>>>(boxes, conf, count, cap, thresh);
  KLAUNCHED("k_dedup_boxes");
}

}  // namespace swim
