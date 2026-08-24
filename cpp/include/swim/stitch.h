// 上游拼接源：六路 4K H.264/HEVC -> NVDEC 解到显存 -> CUDA 拼成画布 -> GpuFrame。
//
// 全程不回 CPU：解码直出 AV_PIX_FMT_CUDA(NV12) 的 CUdeviceptr，拼接 kernel 直接
// 读它，产物就是推理链路要的画布。相比「离线拼好 mp4 再解码」省掉一次 H.264
// 编码 + 一次 5002 宽画布解码（后者本身就要 13 ms/帧，是原链路 49% 的耗时）。
//
// 输入既可以是六个离线片段，也可以是六路直播流（rtsp:// 等）—— libavformat 对
// 两者是同一套 API，差别只在打开选项，见 stitch_source.cpp 的 open_options。
//
// 为什么 NVDEC 在这里可用、解画布时不可用：CUVID 的 H.264 上限 4096x4096，
// 成品画布宽 5002 超限，而单路 4K 只有 3840 宽，正好装得下（HEVC 上限 8192）。
//
// 几何不在 C++ 里算：cpp/tools/build_stitch_lut.py 把标定烘成逐像素查找表
// （源坐标 + 权重），kernel 只做 gather。「哪个像素属于哪个三角形」的判定语义
// 来自 OpenCV 的 fillConvexPoly/getAffineTransform，在 CUDA 里重写是整条链路
// 唯一容易静默出错的地方（差一个边界像素表现为接缝错位而不报错）。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "swim/common.h"
#include "swim/frame_source.h"

namespace swim {

/// 拼接支持的最大路数。显存与 kernel 参数都按它预留，改大要同步 stitch.cu。
constexpr int kMaxLanes = 8;

/// 一路的查找表（显存）+ 当帧的 NV12 平面。
/// bbox 是该路在画布上的非零权重包围盒：一路只覆盖画布的一部分，存全画布会
/// 白占数倍空间。coords/weight 按 bbox 行主序，长度 bw*bh。
struct StitchLane {
  const float2*   coords = nullptr;   // [bh,bw] 源像素坐标（画布像素 -> 源）
  const uint16_t* weight = nullptr;   // [bh,bw] 融合权重，65535 = 1.0，0 = 未覆盖
  int bx = 0, by = 0, bw = 0, bh = 0;
  // 逐帧更新：NVDEC 解出的 NV12 两个平面（pitch 通常大于宽）
  const uint8_t* luma = nullptr;
  const uint8_t* chroma = nullptr;
  int luma_pitch = 0, chroma_pitch = 0;
};

/// 拼接查找表（显存驻留，只读）。由 cpp/tools/build_stitch_lut.py 生成。
class StitchLut {
 public:
  /// 读入并上传到显存；文件损坏或与源尺寸不符时抛异常。
  static std::unique_ptr<StitchLut> load(const std::string& path);
  ~StitchLut();

  StitchLut(const StitchLut&) = delete;
  StitchLut& operator=(const StitchLut&) = delete;

  int canvas_w() const { return canvas_w_; }
  int canvas_h() const { return canvas_h_; }
  int src_w()    const { return src_w_; }
  int src_h()    const { return src_h_; }
  int lanes()    const { return int(lanes_.size()); }
  /// 第 i 路的相机 id（与 mesh 顺序一致，决定该路配哪个视频文件）。
  const std::string& camera(int i) const { return names_[size_t(i)]; }
  /// 拷一份 lane 描述供调用方填 NV12 平面指针后交给 launch_stitch。
  const std::vector<StitchLane>& lanes_view() const { return lanes_; }

 private:
  StitchLut() = default;
  int canvas_w_ = 0, canvas_h_ = 0, src_w_ = 0, src_h_ = 0;
  void* blob_ = nullptr;                 // 一次 cudaMalloc 装下全部表
  std::vector<StitchLane> lanes_;
  std::vector<std::string> names_;
};

/// 拼接 kernel：逐画布像素累加各路的加权采样，直接写出 BGR uint8。
///
/// 单遍无原子、无中间 float 画布：每个线程负责一个画布像素，在寄存器里累加 6 路
/// 贡献后写一次。累加顺序固定为 lane 顺序，因此逐位可复现。
/// 权重已在烘表时全局归一化到和为 1（实测偏差 ±1.5e-5 = u16 的 1 个 LSB），
/// 所以不需要再除 alpha。
///
/// NV12 采样复刻离线参考的插值顺序：先把 4 个双线性 tap 各自转成 BGR（chroma
/// 点复制、bt601 limited->full、floor 取整），再在 BGR 上混合。反过来（先插值
/// chroma 再转色）在 chroma 边缘会偏最多 35 灰阶，实测 mean|d| 1.69 vs 1.08。
///
/// rot180=true 时画布整体转 180°（相机顺序、覆盖关系、采样值全不变）：只把这一
/// 个像素的**落点**改成 (cw-1-x, ch-1-y)。它与「烘表时把 mesh 顶点绕画布中心转
/// 180°」严格等价 —— 该映射在整数网格上是精确双射，不引入任何重采样 —— 但定向
/// 因此与 LUT 解耦：一份表两种定向都能跑，换定向不必重烘表、不必重启标定。
/// 代价为零：读写次数与地址跨度都不变，没有额外遍数，也没有中间画布。
void launch_stitch(const StitchLane* lanes, int n_lanes, uint8_t* canvas,
                   int canvas_w, int canvas_h, int src_w, int src_h,
                   bool rot180, cudaStream_t stream);

/// 六路视频拼接帧源。uri 语法见 open()。
class StitchSource {
 public:
  /// spec 有两种：
  ///   目录 —— 在其中按 lut 的相机 id 找离线片段（`*_<相机>.mp4`）
  ///   文件 —— 相机清单，每行 `<相机>=<地址>`，地址可以是 rtsp:// 等直播流
  /// lut_path 缺省取 models_dir/stitch.lut。ring 为画布环深度，须 >= 3。
  /// fps > 0 时覆盖各路自报帧率的最大值，同时用于离线路限速（见 stitch_source
  /// 的 pace）；相机已配成 30fps 而流里报 59.94 时靠它把时间轴摆正。
  /// rot180 把画布整体转 180°（尺寸不变），在拼接落点上完成，见 launch_stitch。
  static std::unique_ptr<FrameSource> open(const std::string& spec,
                                           const std::string& lut_path,
                                           double fps = 0, bool rot180 = false,
                                           int ring = 4);
};

}  // namespace swim
