// 帧源抽象：离线视频 / live 流，统一给出 GPU 上的画布帧。
//
// 设计要点：
//   - 内部持有 N 个 device 帧缓冲构成环形队列，next() 轮转复用，
//     全程不做 cudaMalloc/Free，避免运行时抖动。
//   - 解码在 CPU 侧完成后 H2D 上传，下游只见 GpuFrame，不感知后端差异。
//     两条 CPU 路径共用预取骨架，差别只在「怎么拿到一帧 BGR24」：
//     ffmpeg 管道（默认）实测 13.0 ms/帧，OpenCV(MSMF) 16.9 ms/帧。前者已贴住
//     ffmpeg CLI 自身的地板（同命令落 NUL 实测 10~13.5 ms/帧，取决于文件是否在
//     系统页缓存里），详见 proc.h 与 FfmpegSource 的注释。
//   - NVDEC 对本画布物理不可用：CUVID 的 H.264 8-bit max_width/height 上限是
//     4096，而画布宽 5002 超限（实测 ffmpeg 报 "Video width 5002 not within
//     range from 48 to 4096"）。HEVC 上限 8192 可以，但输入是 H.264。
//     所以没有 NVDEC 实现：这不是缺功能，是硬件限制。
#pragma once

#include <memory>
#include <string>

#include "swim/common.h"

namespace swim {

/// 解码后端偏好。取值与 --decoder 的 auto|cpu 一一对应：
///   Auto = ffmpeg 管道优先，探测失败自动回退 OpenCV；
///   Cpu  = 强制 OpenCV VideoCapture。本机 videoio 只有 MSMF，它的 YUV→BGR
///          换算与 swscale 不同（逐像素平均差约 4.5/255），因此该路径只作
///          兜底与排障，**数值结果不能用于与 Python 端逐值比对**。
/// 没有 Nvdec 取值：那条路已确认物理不可行（见文件头 4096 上限），留一个只会
/// 被忽略的枚举值等于让每个 switch 都要处理一个死分支。
enum class DecoderPref { Auto, Cpu };

/// 帧源基类。外部直供显存帧的场景（例如拼接程序把已在显存的画布喂进来）请新增
/// 一个 FrameSource 子类实现 next()，接口本身已足够，不必给基类加旁路入口。
class FrameSource {
 public:
  virtual ~FrameSource() = default;

  /// 取下一帧。返回 false 表示流结束（解码线程内的异常会在此 rethrow）。
  /// out.data 指向内部环形缓冲，**只在下一次 next() 之前有效**：环游标每次
  /// next() 前进一格，旧帧随即可能被解码线程覆写。
  /// 因此本接口**仅支持单消费者**（pipeline 的 infer_loop 是唯一调用者），
  /// 需要跨帧留存图像的一律自行拷走。
  virtual bool next(GpuFrame& out) = 0;

  virtual int    width()  const = 0;
  virtual int    height() const = 0;
  virtual double fps()    const = 0;
  /// 总帧数；live 流返回 -1。
  virtual int64_t total() const = 0;
  virtual const char* backend() const = 0;

  /// uri 为视频文件路径或流地址（rtsp://、rtmp:// 等）。
  /// prefetch=true 时内部起一个解码线程预取，使解码与推理真正重叠 ——
  /// 5002x2102 的 CPU 解码 13~17 ms/帧，不重叠会直接吃掉一半吞吐。
  /// ring 为环深度，须 >= 2（预取需要至少一格给消费者、一格给生产者）。
  /// fps > 0 时覆盖容器/流自报的帧率（时间轴由它定）：直播流常报错帧率，
  /// 实测 ZCam 的 rtsp 在 MSMF 下报 30.00 而真值 59.94，划水与速度会整体偏。
  static std::unique_ptr<FrameSource> open(const std::string& uri,
                                           DecoderPref pref = DecoderPref::Auto,
                                           double fps = 0, int ring = 4,
                                           bool prefetch = true);
};

}  // namespace swim
