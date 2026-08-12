// 帧源抽象：离线视频 / live 流 / 外部显存直供，统一给出 GPU 上的画布帧。
//
// 设计要点：
//   - 内部持有 N 个 device 帧缓冲构成环形队列，next() 轮转复用，
//     全程不做 cudaMalloc/Free，避免运行时抖动。
//   - CPU 解码路径（OpenCV/FFmpeg）解出后 H2D 上传；NVDEC 路径直接落显存。
//     本机容器未挂载 libnvcuvid，运行时探测失败会自动回退 CPU 路径，
//     下游对此无感（都拿到 GpuFrame）。
//   - RawSource 供拼接程序把已在显存的画布直接喂进来（零拷贝、零解码）。
#pragma once

#include <memory>
#include <string>

#include "swim/common.h"

namespace swim {

enum class DecoderPref { Auto, Nvdec, Cpu };

class FrameSource {
 public:
  virtual ~FrameSource() = default;

  /// 取下一帧。返回 false 表示流结束。out 指向内部环形缓冲，
  /// 在后续 ring 次 next() 调用内有效。
  virtual bool next(GpuFrame& out) = 0;

  virtual int    width()  const = 0;
  virtual int    height() const = 0;
  virtual double fps()    const = 0;
  /// 总帧数；live 流返回 -1。
  virtual int64_t total() const = 0;
  virtual const char* backend() const = 0;

  /// uri 为视频文件路径或流地址（rtsp://、rtmp:// 等）。
  /// prefetch=true 时内部起一个解码线程预取，使解码与推理真正重叠 ——
  /// 5002x2102 的 CPU 解码约 19 ms/帧，不重叠会直接吃掉一半吞吐。
  static std::unique_ptr<FrameSource> open(const std::string& uri,
                                           DecoderPref pref = DecoderPref::Auto,
                                           int ring = 4, bool prefetch = true);

  /// 外部直供：调用方自行把画布写进 device 指针后调用 push()。
  static std::unique_ptr<FrameSource> raw(int w, int h, double fps, int ring = 3);
  /// 仅 raw 源可用：拷入一帧（src 可为 host 或 device 指针）。
  virtual bool push(const void* src, bool src_on_device) { (void)src; (void)src_on_device; return false; }
};

}  // namespace swim
