#include "swim/frame_source.h"

#include <dlfcn.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace swim {
namespace {

/// device 端环形帧缓冲：一次性分配，之后只轮转，避免运行时分配抖动。
class Ring {
 public:
  Ring(int w, int h, int n) : w_(w), h_(h), bytes_(size_t(w) * h * 3) {
    bufs_.resize(n);
    for (auto& p : bufs_) SWIM_CUDA(cudaMalloc(&p, bytes_));
  }
  ~Ring() { for (auto* p : bufs_) cudaFree(p); }

  uint8_t* acquire() {
    auto* p = bufs_[cursor_];
    cursor_ = (cursor_ + 1) % bufs_.size();
    return static_cast<uint8_t*>(p);
  }
  size_t bytes() const { return bytes_; }

 private:
  int    w_, h_;
  size_t bytes_;
  std::vector<void*> bufs_;
  size_t cursor_ = 0;
};

/// OpenCV 解码 + H2D 上传。VideoCapture 同时支持文件与 rtsp/rtmp，
/// 因此离线与 live 共用同一实现，差别只在 total()。
///
/// prefetch=true 时起一个解码线程：解码(约 19 ms/帧 @5002x2102) 与推理重叠，
/// 否则两者串行会直接吃掉一半吞吐。环形缓冲深度决定可预取多少帧。
class CpuSource final : public FrameSource {
 public:
  CpuSource(const std::string& uri, int ring, bool prefetch)
      : ring_n_(ring), prefetch_(prefetch) {
    SWIM_CHECK(cap_.open(uri), "无法打开输入 " + uri);
    w_   = int(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    h_   = int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 30.0;
    const double n = cap_.get(cv::CAP_PROP_FRAME_COUNT);
    total_ = n > 0 ? int64_t(n) : -1;          // 流媒体读不到帧数
    SWIM_CHECK(w_ > 0 && h_ > 0, "输入尺寸非法");
    bytes_ = size_t(w_) * h_ * 3;

    // device 环 + 每格一块锁页 host 缓冲（H2D 走 DMA，比普通内存快约一倍）
    dev_.resize(ring_n_);
    pin_.resize(ring_n_);
    for (int i = 0; i < ring_n_; ++i) {
      SWIM_CUDA(cudaMalloc(&dev_[i], bytes_));
      SWIM_CUDA(cudaHostAlloc(&pin_[i], bytes_, cudaHostAllocDefault));
    }
    SWIM_CUDA(cudaStreamCreate(&stream_));
    if (prefetch_) worker_ = std::thread([this] { decode_loop(); });
  }

  ~CpuSource() override {
    stop_ = true;
    cv_full_.notify_all();
    cv_ready_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto* p : dev_) cudaFree(p);
    for (auto* p : pin_) cudaFreeHost(p);
    if (stream_) cudaStreamDestroy(stream_);
  }

  bool next(GpuFrame& out) override {
    if (!prefetch_) return decode_one(out);
    std::unique_lock lk(mu_);
    cv_ready_.wait(lk, [&] { return !queue_.empty() || done_; });
    if (queue_.empty()) return false;
    out = queue_.front();
    queue_.pop();
    cv_full_.notify_one();
    return true;
  }

  int     width()  const override { return w_; }
  int     height() const override { return h_; }
  double  fps()    const override { return fps_; }
  int64_t total()  const override { return total_; }
  const char* backend() const override {
    return prefetch_ ? "cpu-decode(prefetch)" : "cpu-decode";
  }

 private:
  /// 同步解码一帧并上传，供无预取路径使用。
  bool decode_one(GpuFrame& out) {
    if (!cap_.read(mat_) || mat_.empty()) return false;
    SWIM_CHECK(mat_.isContinuous() && mat_.type() == CV_8UC3, "帧格式非 BGR8 连续");
    const int slot = int(idx_ % ring_n_);
    std::memcpy(pin_[slot], mat_.data, bytes_);
    out.data = static_cast<uint8_t*>(dev_[slot]);
    out.w = w_; out.h = h_; out.index = idx_++;
    SWIM_CUDA(cudaMemcpyAsync(out.data, pin_[slot], bytes_,
                              cudaMemcpyHostToDevice, stream_));
    SWIM_CUDA(cudaStreamSynchronize(stream_));
    return true;
  }

  void decode_loop() {
    GpuFrame f;
    while (!stop_) {
      {   // 队列满则等消费者取走，形成背压（不会无限预取吃内存）
        std::unique_lock lk(mu_);
        cv_full_.wait(lk, [&] { return int(queue_.size()) < ring_n_ - 1 || stop_; });
        if (stop_) break;
      }
      if (!decode_one(f)) break;
      std::lock_guard lk(mu_);
      queue_.push(f);
      cv_ready_.notify_one();
    }
    std::lock_guard lk(mu_);
    done_ = true;
    cv_ready_.notify_all();
  }

  cv::VideoCapture cap_;
  cv::Mat          mat_;
  std::vector<void*> dev_, pin_;
  size_t       bytes_  = 0;
  int          ring_n_ = 4;
  bool         prefetch_ = true;
  cudaStream_t stream_ = nullptr;
  int      w_ = 0, h_ = 0;
  double   fps_ = 30.0;
  int64_t  total_ = -1, idx_ = 0;

  std::thread             worker_;
  std::mutex              mu_;
  std::condition_variable cv_ready_, cv_full_;
  std::queue<GpuFrame>    queue_;
  std::atomic<bool>       stop_{false};
  bool                    done_ = false;
};

/// 外部直供：拼接程序把画布写进来，无解码。
class RawSource final : public FrameSource {
 public:
  RawSource(int w, int h, double fps, int ring)
      : w_(w), h_(h), fps_(fps), ring_(std::make_unique<Ring>(w, h, ring)) {
    SWIM_CUDA(cudaStreamCreate(&stream_));
  }
  ~RawSource() override { if (stream_) cudaStreamDestroy(stream_); }

  bool push(const void* src, bool src_on_device) override {
    pending_ = ring_->acquire();
    SWIM_CUDA(cudaMemcpyAsync(pending_, src, ring_->bytes(),
                              src_on_device ? cudaMemcpyDeviceToDevice
                                            : cudaMemcpyHostToDevice, stream_));
    SWIM_CUDA(cudaStreamSynchronize(stream_));
    has_ = true;
    return true;
  }
  bool next(GpuFrame& out) override {
    if (!has_) return false;
    out.data = static_cast<uint8_t*>(pending_);
    out.w = w_; out.h = h_; out.index = idx_++;
    has_ = false;
    return true;
  }

  int     width()  const override { return w_; }
  int     height() const override { return h_; }
  double  fps()    const override { return fps_; }
  int64_t total()  const override { return -1; }
  const char* backend() const override { return "raw-device"; }

 private:
  int    w_, h_;
  double fps_;
  std::unique_ptr<Ring> ring_;
  cudaStream_t stream_  = nullptr;
  void*        pending_ = nullptr;
  bool         has_     = false;
  int64_t      idx_     = 0;
};

/// NVDEC 可用性探测：依赖驱动侧 libnvcuvid，容器常未挂载。
bool nvdec_available() {
  static const bool ok = [] {
    // 用 OpenCV 的 cudacodec 作为 NVDEC 入口时需 contrib 模块；
    // 这里只探测驱动库是否存在，缺失即回退，避免链接期硬依赖。
    if (void* h = dlopen("libnvcuvid.so.1", RTLD_LAZY)) { dlclose(h); return true; }
    if (void* h = dlopen("libnvcuvid.so", RTLD_LAZY))   { dlclose(h); return true; }
    printf("[Source] 未找到 libnvcuvid（NVDEC 不可用），回退 CPU 解码\n");
    return false;
  }();
  return ok;
}

}  // namespace

std::unique_ptr<FrameSource> FrameSource::open(const std::string& uri,
                                               DecoderPref pref, int ring,
                                               bool prefetch) {
  const bool want_nvdec = pref == DecoderPref::Nvdec ||
                          (pref == DecoderPref::Auto && nvdec_available());
  if (want_nvdec && !nvdec_available())
    throw std::runtime_error("指定了 --decoder nvdec 但 libnvcuvid 不可用");
  // NVDEC 路径待补：当前 OpenCV(apt 版) 未编译 cudacodec，统一走 CPU 解码。
  // 台式机若装了带 CUDA 的 OpenCV，可在此接 cv::cudacodec::createVideoReader。
  if (want_nvdec)
    printf("[Source] libnvcuvid 可用，但本构建未启用 cudacodec，仍走 CPU 解码\n");
  return std::make_unique<CpuSource>(uri, ring, prefetch);
}

std::unique_ptr<FrameSource> FrameSource::raw(int w, int h, double fps, int ring) {
  return std::make_unique<RawSource>(w, h, fps, ring);
}

}  // namespace swim
