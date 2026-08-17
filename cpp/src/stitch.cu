// 拼接 kernel 与查找表加载。
//
// 字面量一律 ASCII：nvcc 的 EDG 前端在 ACP=936 的机器上按 ANSI 解 .cu，
// 中文字面量会吞掉右引号（加 BOM 与 /utf-8 都无效）。注释可以用中文。
#include "swim/stitch.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace swim {
namespace {

#define KCHECK(cond, msg)                                                     \
  do {                                                                        \
    if (!(cond)) throw std::runtime_error(std::string("stitch: ") + (msg));    \
  } while (0)

// ── 查找表文件格式（与 cpp/tools/build_stitch_lut.py 一一对应）──────────────
// 改任一侧必须同步改另一侧：这里是唯一的读端，那里是唯一的写端。
#pragma pack(push, 1)
struct LutHeader {
  char     magic[8];          // "SWSTLUT1"
  uint32_t version;           // 1
  uint32_t header_bytes;      // == sizeof(LutHeader)
  uint32_t canvas_w, canvas_h;
  uint32_t src_w, src_h;
  uint32_t lane_count;
  uint32_t lane_record_bytes; // == sizeof(LutLane)
  uint32_t body_crc32;        // 保留：读端不校验（表是本地生成物，非分发件）
  char     reserved[28];
};
struct LutLane {
  char     camera[16];        // UTF-8, NUL 结尾
  uint32_t bx, by, bw, bh;
  uint64_t coords_offset;     // 文件绝对偏移
  uint64_t weight_offset;
};
#pragma pack(pop)
// 字面量必须 ASCII（见文件头）：这两条与 build_stitch_lut.py 的
// HEADER/CAMERA struct 尺寸一一对应，改一侧必须改另一侧。
static_assert(sizeof(LutHeader) == 72, "LutHeader must be 72 bytes");
static_assert(sizeof(LutLane) == 48, "LutLane must be 48 bytes");

// ── NV12 采样 ─────────────────────────────────────────────────────────────
// BORDER_REFLECT：关于像素 -0.5 反射，与 GPU sampler 的 mirror 寻址同义。
__device__ __forceinline__ int mirror(int i, int n) {
  if (i < 0) i = -1 - i;
  if (i >= n) i = 2 * n - 1 - i;
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// bt601 limited -> full range BGR。这是 cv2.VideoCapture 与 swscale
// "in_color_matrix=bt601:in_range=tv" 的口径（实测逐字节相同），必须与它一致：
// 换成文件标签声明的 bt709 会带来 mean 3.6 灰阶的整体偏差。
//
// 取整用 floor(x - 0.5) 而不是 floor 或 round：swscale 内部是 16 位定点查表，
// 逐值拟合它的三种候选里这一种最接近（对 4K 整帧 mean|d| 0.36 / floor 0.62 /
// round 1.12），残差是无偏噪声而不是偏置。
__device__ __forceinline__ uchar3 nv12_to_bgr(int yv, int cb_v, int cr_v) {
  const float y  = (float(yv) - 16.f) * (255.f / 219.f);
  const float cb = (float(cb_v) - 128.f) * (255.f / 224.f);
  const float cr = (float(cr_v) - 128.f) * (255.f / 224.f);
  const float b = floorf(y + 1.772f * cb - 0.5f);
  const float g = floorf(y - 0.344136f * cb - 0.714136f * cr - 0.5f);
  const float r = floorf(y + 1.402f * cr - 0.5f);
  return make_uchar3(
      (unsigned char)fminf(fmaxf(b, 0.f), 255.f),
      (unsigned char)fminf(fmaxf(g, 0.f), 255.f),
      (unsigned char)fminf(fmaxf(r, 0.f), 255.f));
}

// 在 (x, y) 处采样一路 NV12，返回 BGR float3。
//
// 插值顺序刻意与离线参考一致：4 个 tap 各自 chroma 点复制 + 转 BGR + 截断，
// 再在 BGR 上双线性。反过来先插 chroma 再转色，在 chroma 边缘偏差大一倍
// (实测独占像素 mean|d| 1.69 vs 1.08)。
__device__ __forceinline__ float3 sample_nv12(const uint8_t* luma, int lpitch,
                                              const uint8_t* chroma, int cpitch,
                                              int w, int h, float x, float y) {
  const int x0 = int(floorf(x)), y0 = int(floorf(y));
  const float fx = x - float(x0), fy = y - float(y0);
  const int xs[2] = {mirror(x0, w), mirror(x0 + 1, w)};
  const int ys[2] = {mirror(y0, h), mirror(y0 + 1, h)};
  float3 acc = make_float3(0.f, 0.f, 0.f);
  for (int j = 0; j < 2; ++j) {
    const float wy = j ? fy : 1.f - fy;
    for (int i = 0; i < 2; ++i) {
      const float wx = i ? fx : 1.f - fx;
      const int px = xs[i], py = ys[j];
      // chroma 平面半分辨率，两个分量交错存放 (U,V)
      const uint8_t* uv = chroma + size_t(py >> 1) * cpitch + size_t(px >> 1) * 2;
      const uchar3 c = nv12_to_bgr(luma[size_t(py) * lpitch + px], uv[0], uv[1]);
      const float k = wx * wy;
      acc.x += k * float(c.x);
      acc.y += k * float(c.y);
      acc.z += k * float(c.z);
    }
  }
  return acc;
}

// 逐画布像素累加各路贡献。每线程一个像素，寄存器里累加后写一次，
// 无原子、无中间 float 画布；累加顺序固定为 lane 顺序，故逐位可复现。
__global__ void stitch_kernel(const StitchLane* lanes, int n_lanes,
                              uint8_t* canvas, int cw, int ch,
                              int src_w, int src_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cw || y >= ch) return;

  float3 sum = make_float3(0.f, 0.f, 0.f);
  float  wsum = 0.f;
  for (int i = 0; i < n_lanes; ++i) {
    const StitchLane L = lanes[i];
    const int lx = x - L.bx, ly = y - L.by;
    if (lx < 0 || ly < 0 || lx >= L.bw || ly >= L.bh) continue;
    const size_t k = size_t(ly) * L.bw + size_t(lx);
    const float wq = float(L.weight[k]) * (1.f / 65535.f);
    if (wq <= 0.f) continue;             // bbox 内但未覆盖
    const float2 c = L.coords[k];
    const float3 v = sample_nv12(L.luma, L.luma_pitch, L.chroma, L.chroma_pitch,
                                 src_w, src_h, c.x, c.y);
    sum.x += wq * v.x;
    sum.y += wq * v.y;
    sum.z += wq * v.z;
    wsum += wq;
  }
  uint8_t* p = canvas + (size_t(y) * cw + size_t(x)) * 3;
  if (wsum <= 0.f) {                     // 未覆盖像素写黑（画布 100% 覆盖，兜底）
    p[0] = p[1] = p[2] = 0;
    return;
  }
  // 权重烘表时已归一化到和为 1（偏差 <= u16 的 1 个 LSB），除以 wsum 只是
  // 兜掉量化残差与边界像素，不改变正常像素的值。
  // 末尾截断而不是四舍五入：离线参考是 clip(acc,0,255).astype(uint8)，也就是
  // 截断。跟着它能把残差压到亚灰阶（实测 mean|d| 0.31，全部 <= 2）。
  const float inv = 1.f / wsum;
  p[0] = (unsigned char)fminf(fmaxf(sum.x * inv, 0.f), 255.f);
  p[1] = (unsigned char)fminf(fmaxf(sum.y * inv, 0.f), 255.f);
  p[2] = (unsigned char)fminf(fmaxf(sum.z * inv, 0.f), 255.f);
}

}  // namespace

void launch_stitch(const StitchLane* lanes, int n_lanes, uint8_t* canvas,
                   int canvas_w, int canvas_h, int src_w, int src_h,
                   cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((canvas_w + block.x - 1) / block.x,
                  (canvas_h + block.y - 1) / block.y);
  stitch_kernel<<<grid, block, 0, stream>>>(lanes, n_lanes, canvas, canvas_w,
                                            canvas_h, src_w, src_h);
}

std::unique_ptr<StitchLut> StitchLut::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  KCHECK(f.good(), "cannot open LUT " + path +
                       " (run cpp/tools/build_stitch_lut.py)");
  f.seekg(0, std::ios::end);
  const std::streamoff size = f.tellg();
  f.seekg(0);
  KCHECK(size > std::streamoff(sizeof(LutHeader)), "LUT truncated: " + path);

  LutHeader h{};
  f.read(reinterpret_cast<char*>(&h), sizeof h);
  KCHECK(std::memcmp(h.magic, "SWSTLUT1", 8) == 0, "bad LUT magic: " + path);
  KCHECK(h.version == 1, "unsupported LUT version");
  KCHECK(h.header_bytes == sizeof(LutHeader), "LUT header size mismatch");
  KCHECK(h.lane_record_bytes == sizeof(LutLane), "LUT lane record size mismatch");
  KCHECK(h.lane_count >= 1 && int(h.lane_count) <= kMaxLanes,
         "LUT lane count out of range 1.." + std::to_string(kMaxLanes));
  KCHECK(h.canvas_w > 0 && h.canvas_h > 0 && h.src_w > 0 && h.src_h > 0,
         "LUT has non-positive dimensions");

  std::vector<LutLane> recs(h.lane_count);
  f.read(reinterpret_cast<char*>(recs.data()),
         std::streamsize(recs.size() * sizeof(LutLane)));
  KCHECK(f.good(), "LUT lane table truncated");

  auto lut = std::unique_ptr<StitchLut>(new StitchLut());
  lut->canvas_w_ = int(h.canvas_w);
  lut->canvas_h_ = int(h.canvas_h);
  lut->src_w_ = int(h.src_w);
  lut->src_h_ = int(h.src_h);

  // 一次 cudaMalloc 装下全部表：每路两块 (coords, weight)，按 8 字节对齐排布，
  // 这样运行期零分配、也不用为每路各存一个指针的生命周期。
  size_t total = 0;
  std::vector<size_t> coord_bytes(recs.size()), weight_bytes(recs.size());
  for (size_t i = 0; i < recs.size(); ++i) {
    const auto& r = recs[i];
    KCHECK(r.bw > 0 && r.bh > 0, "LUT lane has empty bbox");
    KCHECK(int(r.bx + r.bw) <= lut->canvas_w_ && int(r.by + r.bh) <= lut->canvas_h_,
           "LUT lane bbox exceeds the canvas");
    coord_bytes[i] = size_t(r.bw) * r.bh * sizeof(float2);
    weight_bytes[i] = size_t(r.bw) * r.bh * sizeof(uint16_t);
    const uint64_t end = r.weight_offset + weight_bytes[i];
    KCHECK(r.coords_offset >= h.header_bytes && end <= uint64_t(size),
           "LUT blob out of bounds");
    total += coord_bytes[i] + ((weight_bytes[i] + 7) & ~size_t(7));
  }
  SWIM_CUDA(cudaMalloc(&lut->blob_, total));

  std::vector<char> staging;
  size_t cursor = 0;
  auto* base = static_cast<char*>(lut->blob_);
  // 从文件读一块到显存的 cursor 处，返回该块的 device 指针（并推进 cursor）
  auto upload = [&](uint64_t offset, size_t bytes) -> const void* {
    staging.resize(bytes);
    f.seekg(std::streamoff(offset));
    f.read(staging.data(), std::streamsize(bytes));
    KCHECK(f.good(), "LUT blob read failed");
    char* dst = base + cursor;
    SWIM_CUDA(cudaMemcpy(dst, staging.data(), bytes, cudaMemcpyHostToDevice));
    cursor += (bytes + 7) & ~size_t(7);
    return dst;
  };
  for (size_t i = 0; i < recs.size(); ++i) {
    const auto& r = recs[i];
    StitchLane lane{};
    lane.bx = int(r.bx); lane.by = int(r.by);
    lane.bw = int(r.bw); lane.bh = int(r.bh);
    lane.coords = static_cast<const float2*>(upload(r.coords_offset, coord_bytes[i]));
    lane.weight = static_cast<const uint16_t*>(upload(r.weight_offset, weight_bytes[i]));
    lut->lanes_.push_back(lane);
    char name[17] = {};
    std::memcpy(name, r.camera, 16);
    lut->names_.emplace_back(name);
  }
  printf("[Stitch] LUT %s: canvas %dx%d, source %dx%d, %u lanes, %.1f MB\n",
         path.c_str(), lut->canvas_w_, lut->canvas_h_, lut->src_w_, lut->src_h_,
         h.lane_count, double(total) / 1e6);
  return lut;
}

StitchLut::~StitchLut() {
  if (blob_) cudaFree(blob_);
}

}  // namespace swim
