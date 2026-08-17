// 六路 4K -> NVDEC -> CUDA 拼接 -> GpuFrame。设计取舍见 include/swim/stitch.h。
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <filesystem>

#include "swim/stitch.h"

namespace swim {
namespace {

/// 每路允许「同时被我们持有」的解码帧数。h264_cuvid 的 surface 池大小 =
/// 解码器自身需要 + extra_hw_frames；给不够会在 receive_frame 处卡住，
/// 给太多只是白占显存（每张 4K NV12 约 12 MB）。
/// 我们最多持有：预取队列 kLaneQueue + 画布环里各格扣着的 1 帧。
constexpr int kExtraHwFrames = 8;
/// 每路预取深度：解码与拼接重叠够用，再深只是延迟增大。
constexpr int kLaneQueue = 3;

std::string av_msg(const std::string& what, int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buf, sizeof buf);
  return what + " 失败: " + buf;
}

/// 一路 NVDEC 解码线程。解出的 AVFrame 是 AV_PIX_FMT_CUDA(NV12)，data[0]/data[1]
/// 直接就是显存指针，全程无 host 拷贝。帧由消费者 av_frame_free 释放，
/// 因此 surface 一直被占住直到拼接 kernel 读完（见 StitchSource 的 slot 回收）。
class NvdecLane {
 public:
  NvdecLane(const std::string& path, AVBufferRef* hw) : path_(path) {
    int rc = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
    SWIM_CHECK(rc >= 0, av_msg("打开 " + path, rc));
    rc = avformat_find_stream_info(fmt_, nullptr);
    SWIM_CHECK(rc >= 0, av_msg("探测 " + path, rc));
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    SWIM_CHECK(stream_ >= 0, path + " 没有视频流");
    AVStream* st = fmt_->streams[stream_];
    SWIM_CHECK(st->codecpar->codec_id == AV_CODEC_ID_H264,
               path + " 不是 H.264（NVDEC 这条路只支持 H.264）");

    const AVCodec* dec = avcodec_find_decoder_by_name("h264_cuvid");
    SWIM_CHECK(dec != nullptr, "FFmpeg 里没有 h264_cuvid 解码器");
    ctx_ = avcodec_alloc_context3(dec);
    SWIM_CHECK(ctx_ != nullptr, "分配解码器上下文失败");
    rc = avcodec_parameters_to_context(ctx_, st->codecpar);
    SWIM_CHECK(rc >= 0, av_msg("拷贝流参数", rc));
    ctx_->hw_device_ctx = av_buffer_ref(hw);
    // 不设 pkt_timebase 会打印 "Invalid pkt_timebase" 并按原样传时间戳
    ctx_->pkt_timebase = st->time_base;
    // 我们会把解码帧一直扣到拼接读完（见 StitchFrameSource 的 slot 回收），
    // 所以要让解码器多备这么多张 surface，否则它会等我们归还而卡住。
    // 用 extra_hw_frames 而不是已废弃的 "surfaces" 选项。
    ctx_->extra_hw_frames = kExtraHwFrames;
    rc = avcodec_open2(ctx_, dec, nullptr);
    SWIM_CHECK(rc >= 0, av_msg("打开 h264_cuvid", rc));

    w_ = st->codecpar->width;
    h_ = st->codecpar->height;
    const AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    fps_ = r.den > 0 ? double(r.num) / double(r.den) : 0.0;
    total_ = st->nb_frames > 0 ? st->nb_frames : -1;
    pkt_ = av_packet_alloc();
    SWIM_CHECK(pkt_ != nullptr, "分配 AVPacket 失败");
  }

  ~NvdecLane() {
    stop();
    while (!queue_.empty()) {                 // 释放未被消费的帧，归还 surface
      AVFrame* f = queue_.front();
      queue_.pop();
      av_frame_free(&f);
    }
    if (pkt_) av_packet_free(&pkt_);
    if (ctx_) avcodec_free_context(&ctx_);
    if (fmt_) avformat_close_input(&fmt_);
  }

  NvdecLane(const NvdecLane&) = delete;
  NvdecLane& operator=(const NvdecLane&) = delete;

  void start() { worker_ = std::thread([this] { loop(); }); }

  /// 停线程（幂等）。worker 可能阻塞在背压等待或 av_read_frame 上。
  void stop() noexcept {
    if (!worker_.joinable()) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_room_.notify_all();
    cv_ready_.notify_all();
    worker_.join();
  }

  /// 取下一帧（阻塞）。返回 nullptr 表示本路结束；调用方负责 av_frame_free。
  /// 解码线程的异常在此 rethrow。
  AVFrame* pop() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_ready_.wait(lk, [this] { return !queue_.empty() || done_ || stop_; });
    if (queue_.empty()) {
      if (err_) {
        auto e = err_;
        err_ = nullptr;
        std::rethrow_exception(e);
      }
      return nullptr;
    }
    AVFrame* f = queue_.front();
    queue_.pop();
    cv_room_.notify_one();
    return f;
  }

  int    width()  const { return w_; }
  int    height() const { return h_; }
  double fps()    const { return fps_; }
  int64_t total() const { return total_; }
  const std::string& path() const { return path_; }

 private:
  /// 解一帧到 out（已 ref）；返回 false 表示流结束。
  bool decode_one(AVFrame* out) {
    for (;;) {
      int rc = avcodec_receive_frame(ctx_, out);
      if (rc == 0) {
        SWIM_CHECK(out->format == AV_PIX_FMT_CUDA,
                   path_ + " 解出的不是 CUDA 帧（hw_device_ctx 没生效？）");
        return true;
      }
      SWIM_CHECK(rc == AVERROR(EAGAIN) || rc == AVERROR_EOF,
                 av_msg("接收解码帧", rc));
      if (rc == AVERROR_EOF) return false;
      if (av_read_frame(fmt_, pkt_) < 0) {     // 读完了，冲刷解码器
        if (flushed_) return false;
        flushed_ = true;
        avcodec_send_packet(ctx_, nullptr);
        continue;
      }
      if (pkt_->stream_index != stream_) {
        av_packet_unref(pkt_);
        continue;
      }
      rc = avcodec_send_packet(ctx_, pkt_);
      av_packet_unref(pkt_);
      SWIM_CHECK(rc >= 0, av_msg("送入解码器", rc));
    }
  }

  void loop() try {
    for (;;) {
      {   // 队列满则等消费者取走，形成背压
        std::unique_lock<std::mutex> lk(mu_);
        cv_room_.wait(lk, [this] {
          return int(queue_.size()) < kLaneQueue || stop_;
        });
        if (stop_) break;
      }
      AVFrame* f = av_frame_alloc();
      SWIM_CHECK(f != nullptr, "分配 AVFrame 失败");
      if (!decode_one(f)) {
        av_frame_free(&f);
        break;
      }
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(f);
      cv_ready_.notify_one();
    }
    finish(nullptr);
  } catch (...) {
    // 异常在非主线程逃逸会直接 terminate，存下来交给 pop() 在调用线程重抛
    finish(std::current_exception());
  }

  void finish(std::exception_ptr e) {
    std::lock_guard<std::mutex> lk(mu_);
    err_ = e;
    done_ = true;
    cv_ready_.notify_all();
  }

  std::string      path_;
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext*  ctx_ = nullptr;
  AVPacket*        pkt_ = nullptr;
  int     stream_ = -1, w_ = 0, h_ = 0;
  double  fps_ = 0;
  int64_t total_ = -1;
  bool    flushed_ = false;

  std::thread             worker_;
  std::mutex              mu_;
  std::condition_variable cv_ready_, cv_room_;
  std::queue<AVFrame*>    queue_;
  bool                    stop_ = false, done_ = false;
  std::exception_ptr      err_;
};

/// 六路 NVDEC + CUDA 拼接的帧源。
///
/// 同步纪律（与 PrefetchSource 同构，理由不同）：
///   - 拼接排在本类自己的 stream 上，next() 返回前 cudaEventSynchronize，
///     因为 Pipeline 不做任何跨 stream 等待（它只在自己的 stream 上排 kernel）。
///   - 画布环深 ring 决定「多久之后复用同一格」。Pipeline 每帧都会
///     cudaEventSynchronize(ev_count_)，那会迫使它 stream 上此前排的全部工作
///     （含上一帧的 crop_affine 与整帧 D2H）完成，所以领先不超过 1 帧；
///     ring>=3 即安全，默认 4 留余量。
///   - 六路的 AVFrame 必须活到拼接 kernel 读完才能 free：free 会把 surface 还给
///     解码器，提前还就是边解码边覆写正在被读的显存。因此每格记住它的 6 个
///     AVFrame，等该格的 event 之后才释放。
class StitchFrameSource final : public FrameSource {
 public:
  StitchFrameSource(const std::string& dir, const std::string& lut_path, int ring)
      : ring_n_(ring) {
    SWIM_CHECK(ring_n_ >= 3, "拼接画布环深须 >= 3");
    // 顺序要紧：av_hwdevice_ctx_create(AV_CUDA_USE_PRIMARY_CONTEXT) 要设置
    // primary context 的 flags，若 CUDA runtime API（cudaMalloc 等）已激活它就会
    // 失败。所以必须先建 hwdevice，再做任何 cudaMalloc（含 StitchLut::load）。
    const int rc = av_hwdevice_ctx_create(&hw_, AV_HWDEVICE_TYPE_CUDA, "0",
                                          nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
    SWIM_CHECK(rc >= 0, av_msg("创建 CUDA 硬解上下文", rc) +
                            "（拼接源必须在任何 CUDA 分配之前构造）");

    lut_ = StitchLut::load(lut_path);
    const int n = lut_->lanes();

    // 按 LUT 里的相机 id 找片段：一个相机恰好一个文件，多于一个必须报错 ——
    // 静默挑一个等于把错误的相机贴到网格上，症状是接缝错位而不是报错。
    namespace fs = std::filesystem;
    SWIM_CHECK(fs::is_directory(dir), "不是目录: " + dir);
    for (int i = 0; i < n; ++i) {
      const std::string& cam = lut_->camera(i);
      std::vector<std::string> hits;
      for (const auto& e : fs::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        // AppleDouble 残片（._name）会匹配后缀，且只有 4 KB 非视频内容
        if (name.rfind("._", 0) == 0) continue;
        if (e.path().extension() != ".mp4") continue;
        if (name.size() > cam.size() + 4 &&
            name.compare(name.size() - cam.size() - 4, cam.size(), cam) == 0)
          hits.push_back(e.path().string());
      }
      SWIM_CHECK(hits.size() == 1,
                 "在 " + dir + " 里为 " + cam + " 找到 " +
                     std::to_string(hits.size()) + " 个片段（应恰好 1 个）");
      lanes_.push_back(std::make_unique<NvdecLane>(hits[0], hw_));
    }

    // 六路必须同分辨率同帧率：拼接表按单一源尺寸烘的，混着不同尺寸会越界采样
    for (int i = 0; i < n; ++i) {
      SWIM_CHECK(lanes_[i]->width() == lut_->src_w() &&
                     lanes_[i]->height() == lut_->src_h(),
                 lanes_[i]->path() + " 是 " + std::to_string(lanes_[i]->width()) +
                     "x" + std::to_string(lanes_[i]->height()) + "，而 LUT 按 " +
                     std::to_string(lut_->src_w()) + "x" +
                     std::to_string(lut_->src_h()) + " 烘制");
      total_ = total_ < 0 ? lanes_[i]->total()
                          : std::min(total_, lanes_[i]->total());
      fps_ = std::max(fps_, lanes_[i]->fps());
    }
    if (fps_ <= 0) fps_ = 30.0;

    bytes_ = size_t(lut_->canvas_w()) * lut_->canvas_h() * 3;
    SWIM_CUDA(cudaStreamCreate(&stream_));
    slots_.resize(size_t(ring_n_));
    for (auto& s : slots_) {
      SWIM_CUDA(cudaMalloc(&s.canvas, bytes_));
      SWIM_CUDA(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
      // lane 描述每帧变（NV12 平面指针），锁页 host 一份 + device 一份，
      // H2D 只有几百字节，与拼接 kernel 同 stream 顺序执行
      SWIM_CUDA(cudaHostAlloc(&s.host_lanes, sizeof(StitchLane) * size_t(n),
                              cudaHostAllocDefault));
      SWIM_CUDA(cudaMalloc(&s.dev_lanes, sizeof(StitchLane) * size_t(n)));
      s.frames.resize(size_t(n), nullptr);
    }
    printf("[Stitch] %d 路 NVDEC -> 画布 %dx%d，环 %d x %.1f MB，%.2f fps 总帧 %lld\n",
           n, lut_->canvas_w(), lut_->canvas_h(), ring_n_, double(bytes_) / 1e6,
           fps_, static_cast<long long>(total_));
    for (auto& lane : lanes_) lane->start();
  }

  ~StitchFrameSource() override {
    for (auto& lane : lanes_) lane->stop();      // 先停线程，再回收显存
    if (stream_) cudaStreamSynchronize(stream_);
    for (auto& s : slots_) {
      for (auto*& f : s.frames)
        if (f) av_frame_free(&f);
      if (s.canvas) cudaFree(s.canvas);
      if (s.done) cudaEventDestroy(s.done);
      if (s.host_lanes) cudaFreeHost(s.host_lanes);
      if (s.dev_lanes) cudaFree(s.dev_lanes);
    }
    if (stream_) cudaStreamDestroy(stream_);
    lanes_.clear();                              // 解码器先于 hw_ 销毁
    lut_.reset();
    if (hw_) av_buffer_unref(&hw_);
  }

  bool next(GpuFrame& out) override {
    const int n = lut_->lanes();
    Slot& s = slots_[size_t(idx_ % ring_n_)];
    // 复用这一格前：等它上一轮的拼接读完源帧，然后把 surface 还给解码器
    SWIM_CUDA(cudaEventSynchronize(s.done));
    for (auto*& f : s.frames)
      if (f) av_frame_free(&f);

    for (int i = 0; i < n; ++i) {
      AVFrame* f = lanes_[size_t(i)]->pop();
      if (!f) {                                  // 任一路结束即整体结束
        for (int j = 0; j < i; ++j) av_frame_free(&s.frames[size_t(j)]);
        return false;
      }
      s.frames[size_t(i)] = f;
      StitchLane lane = lut_->lanes_view()[size_t(i)];
      lane.luma = f->data[0];
      lane.chroma = f->data[1];
      lane.luma_pitch = f->linesize[0];
      lane.chroma_pitch = f->linesize[1];
      s.host_lanes[i] = lane;
    }
    SWIM_CUDA(cudaMemcpyAsync(s.dev_lanes, s.host_lanes,
                              sizeof(StitchLane) * size_t(n),
                              cudaMemcpyHostToDevice, stream_));
    launch_stitch(s.dev_lanes, n, static_cast<uint8_t*>(s.canvas),
                  lut_->canvas_w(), lut_->canvas_h(), lut_->src_w(),
                  lut_->src_h(), stream_);
    SWIM_CUDA(cudaEventRecord(s.done, stream_));
    // 下游在别的 stream 上读它，而 Pipeline 不做跨 stream 等待，故在此等一次
    SWIM_CUDA(cudaEventSynchronize(s.done));

    out.data = static_cast<uint8_t*>(s.canvas);
    out.w = lut_->canvas_w();
    out.h = lut_->canvas_h();
    out.index = idx_++;
    return true;
  }

  int     width()  const override { return lut_->canvas_w(); }
  int     height() const override { return lut_->canvas_h(); }
  double  fps()    const override { return fps_; }
  int64_t total()  const override { return total_; }
  const char* backend() const override { return "nvdec-stitch"; }

 private:
  /// 画布环的一格：画布显存 + 拼接完成事件 + 该帧的 lane 描述与源帧。
  struct Slot {
    void*        canvas = nullptr;
    cudaEvent_t  done = nullptr;
    StitchLane*  host_lanes = nullptr;
    StitchLane*  dev_lanes = nullptr;
    std::vector<AVFrame*> frames;
  };

  AVBufferRef*                hw_ = nullptr;
  std::unique_ptr<StitchLut>  lut_;
  std::vector<std::unique_ptr<NvdecLane>> lanes_;
  std::vector<Slot>           slots_;
  cudaStream_t stream_ = nullptr;
  size_t  bytes_ = 0;
  int     ring_n_ = 4;
  double  fps_ = 0;
  int64_t total_ = -1, idx_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> StitchSource::open(const std::string& dir,
                                                const std::string& lut_path,
                                                int ring) {
  return std::make_unique<StitchFrameSource>(dir, lut_path, ring);
}

}  // namespace swim
