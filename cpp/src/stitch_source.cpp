// 六路 4K -> NVDEC -> CUDA 拼接 -> GpuFrame。设计取舍见 include/swim/stitch.h。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
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

#include "swim/stitch.h"

namespace swim {
namespace {

/// 每路允许「同时被我们持有」的解码帧数。h264_cuvid 的 surface 池大小 =
/// 解码器自身需要 + extra_hw_frames；给不够会在 receive_frame 处**卡住**
/// （不是报错），给太多只是白占显存（每张 4K NV12 约 12 MB）。
///
/// 我们同时扣着的上界：画布环 ring 格各 1 帧 + 预取队列 kLaneQueue + 正在解的 1。
/// 默认 ring=4 时是 4+3+1=8，但实测取 8 会在 live 流上把解码器卡死几秒
/// （相机 4K@60 的 DPB 比离线片段更深），所以留一倍余量。
constexpr int kExtraHwFrames = 16;
/// 每路预取深度：解码与拼接重叠够用，再深只是延迟增大。
constexpr int kLaneQueue = 3;
/// 直播流连续读失败多少次就判定要重连。单次失败通常是 5 s 的 socket 超时
/// （我们下发了 timeout=5s），而 30fps 的流五秒没数据已经是掉线 —— 把同一个死
/// socket 重试几十次只是白等。**重连的代价现在只有约 1 秒**（见 reopen()），
/// 不再是「整条链路停」，所以这个数从 20 降到 3：只用来容一次瞬时抖动。
constexpr int kMaxReadFails = 3;
/// 重连失败后的退避。相机重启要几十秒，退避太短只是空转（每次尝试本身还要等
/// 最多 5 s 的连接超时），太长现场等不起。
constexpr int kReconnectWaitMs = 1000;
/// 直播路 pop 的等待上限。超时就让调用方用上一帧顶住 —— 一路卡住不该把整块
/// 画布拖停。只兜「本路还没被标记不健康」的那一小段（首次读失败前最多 5 s）：
/// 一旦标记为不健康，pop 立刻返回 Stall，不再每帧陪它等这 300 ms。
constexpr int kLiveStallMs = 300;


std::string av_msg(const std::string& what, int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buf, sizeof buf);
  return what + " 失败: " + buf;
}

/// uri 是网络流吗（决定要不要下 rtsp/超时那组选项）。
bool is_stream(const std::string& uri) {
  for (const char* p : {"rtsp://", "rtsps://", "rtmp://", "http://", "https://",
                        "udp://", "srt://"})
    if (uri.rfind(p, 0) == 0) return true;
  return false;
}

/// codec -> cuvid 解码器名。NVDEC 只有这两种在本项目的分辨率下可用
/// （H.264 上限 4096，HEVC 8192；单路 4K 是 3840 宽，两者都装得下）。
const char* cuvid_name(AVCodecID id) {
  if (id == AV_CODEC_ID_H264) return "h264_cuvid";
  if (id == AV_CODEC_ID_HEVC) return "hevc_cuvid";
  return nullptr;
}

/// 打开流用的选项。文件不需要任何选项；网络流这两条是现场必需的：
///   rtsp_transport=tcp   UDP 在 4K 码率下丢包会花屏（前身项目的现场结论）
///   timeout              socket IO 超时，连不上时不要无限期卡住（旧脚本注释
///                        「连不上会卡住」就是缺这个）。新版 ffmpeg 把 rtsp 的
///                        `stimeout` 改名成 `timeout`，两个都下发以兼容旧库 ——
///                        识别不了的键会留在 dict 里被忽略，不是错误。
///
/// 刻意**不加** `fflags=nobuffer` 与 `reorder_queue_size=0`：它们看着能降延迟，
/// 实测（本机 4K@60 RTSP，400 帧）会让 RTP 重排缓冲失效 → `corrupt decoded
/// frame` + `error while decoding MB`，而且更慢（8.7 s vs 7.2 s）。真正的延迟
/// 控制在下游：每路只预取 kLaneQueue 帧，满了就背压。
void open_options(const std::string& uri, AVDictionary** opt) {
  if (!is_stream(uri)) return;
  av_dict_set(opt, "rtsp_transport", "tcp", 0);
  av_dict_set(opt, "timeout", "5000000", 0);       // 5 s，单位微秒
  av_dict_set(opt, "stimeout", "5000000", 0);      // 旧名，兼容老 ffmpeg
}

/// 一路 NVDEC 解码线程。解出的 AVFrame 是 AV_PIX_FMT_CUDA(NV12)，data[0]/data[1]
/// 直接就是显存指针，全程无 host 拷贝。帧由消费者 av_frame_free 释放，
/// 因此 surface 一直被占住直到拼接 kernel 读完（见 StitchSource 的 slot 回收）。
///
/// uri 既可以是本地文件也可以是 rtsp:// 等网络流 —— libavformat 对两者是同一套
/// API，差别只在打开时给的选项（见 open_options）。
///
/// **每路自建 hw device context**（都带 AV_CUDA_USE_PRIMARY_CONTEXT，所以底层仍是
/// 同一个 CUDA primary context，指针可被我们的 kernel 直接读）。共用一个
/// AVBufferRef 会让六路争同一把 hwctx 锁：满速解码的离线路几乎一直握着它，
/// 直播路的 av_read_frame 迟迟得不到调度 → socket 缓冲堆积 → 相机侧发送阻塞
/// → read 返回 ETIMEDOUT(-138)，表现为整条链路先掉到几 fps 再静默停住。
///
/// **直播路永不结束**：读失败连续 kMaxReadFails 次就在本线程内重开
/// demux + 解码器（hw device context 与 surface 池不动，所以重连只花约 1 秒），
/// 失败则退避后再试，直到 stop()。掉线期间 pop() 超时返回「本路无新帧」，
/// 由 StitchFrameSource 用上一帧顶住，其余五路照常出画。
class NvdecLane {
 public:
  /// pop 的三种结果。Stall 只会出现在直播路：它不是错误，也不是结束。
  enum class Got { Frame, End, Stall };

  explicit NvdecLane(const std::string& uri)
      : path_(uri), live_(is_stream(uri)) {
    int rc = av_hwdevice_ctx_create(&hw_, AV_HWDEVICE_TYPE_CUDA, "0", nullptr,
                                    AV_CUDA_USE_PRIMARY_CONTEXT);
    SWIM_CHECK(rc >= 0, av_msg("创建 CUDA 硬解上下文", rc));
    pkt_ = av_packet_alloc();
    SWIM_CHECK(pkt_ != nullptr, "分配 AVPacket 失败");
    std::string err;
    SWIM_CHECK(try_open(err), err);
  }

  ~NvdecLane() {
    stop();
    while (!queue_.empty()) {                 // 释放未被消费的帧，归还 surface
      AVFrame* f = queue_.front();
      queue_.pop();
      av_frame_free(&f);
    }
    if (pkt_) av_packet_free(&pkt_);
    close_input();
    if (hw_) av_buffer_unref(&hw_);        // 必须在解码器之后
  }

  NvdecLane(const NvdecLane&) = delete;
  NvdecLane& operator=(const NvdecLane&) = delete;

  /// pace=true 时离线路按 fps 限速（理由见 loop()）。fps 用画布帧率而不是本路
  /// 自报值：--fps 覆盖后两者可能不同，限速要跟着实际时间轴走。
  void start(bool pace, double fps) {
    pace_ = pace && !live_;
    pace_fps_ = fps > 0 ? fps : fps_;
    worker_ = std::thread([this] { loop(); });
  }

  /// 关掉 demux + 解码器，但保留已探到的 w/h/fps/codec，由 loop() 自己补开。
  /// 用在「探完参数到真正开始取帧」之间那段空窗：首次在新 GPU 上要先烘 engine
  /// （几分钟），这期间六路 RTSP 若已 PLAY 着没人读，TCP 缓冲堆满就会一路
  /// -138 超时 + 重连刷屏 —— 现场日志实测正是这个。相机只在真要消费时才拉。
  void park() {
    close_input();
    parked_ = true;
  }

  /// 停线程（幂等）。worker 可能阻塞在背压等待、av_read_frame 或重连退避上。
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

  /// 取下一帧。Frame 时 out 归调用方所有（负责 av_frame_free）。
  /// 离线路只会给出 Frame / End（队列空就阻塞等，丢帧会与基线对不上）。
  /// 直播路还可能给 Stall：等了 kLiveStallMs 仍无新帧，或本路正在重连。
  /// 解码线程的异常在此 rethrow。
  Got pop(AVFrame*& out) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto ready = [this] { return !queue_.empty() || done_ || stop_; };
    if (live_) {
      // 已知不健康时不再每帧陪它等 —— 直接顶上一帧，画布维持满帧率。
      if (!healthy_ && queue_.empty()) return Got::Stall;
      if (!cv_ready_.wait_for(lk, std::chrono::milliseconds(kLiveStallMs), ready))
        return Got::Stall;
    } else {
      cv_ready_.wait(lk, ready);
    }
    if (queue_.empty()) {
      if (err_) {
        auto e = err_;
        err_ = nullptr;
        std::rethrow_exception(e);
      }
      return Got::End;
    }
    out = queue_.front();
    queue_.pop();
    cv_room_.notify_one();
    return Got::Frame;
  }

  int    width()  const { return w_; }
  int    height() const { return h_; }
  double fps()    const { return fps_; }
  int64_t total() const { return total_; }
  bool    live()  const { return live_; }
  /// 实际用上的 cuvid 解码器名（h264_cuvid / hevc_cuvid），给启动摘要用。
  const char* codec() const { return codec_ ? codec_ : "?"; }
  int64_t reconnects() const { return reconnects_; }
  /// 解出并入队的累计帧数。无锁：只给 --show-fps 算「本路到帧率」，差一帧无碍。
  int64_t decoded() const { return decoded_; }
  int64_t dropped() const {                 // 只在 live_ 下非零
    std::lock_guard<std::mutex> lk(mu_);
    return dropped_;
  }
  const std::string& path() const { return path_; }

 private:
  /// 打开 demux + 解码器（可重复调用：先关掉旧的）。失败时把原因写进 err、
  /// 不留半开状态。首次由构造函数转成异常，重连时用来决定要不要退避重试。
  bool try_open(std::string& err) {
    close_input();
    AVDictionary* opt = nullptr;
    open_options(path_, &opt);
    int rc = avformat_open_input(&fmt_, path_.c_str(), nullptr, &opt);
    av_dict_free(&opt);
    if (rc < 0) return fail(av_msg("打开 " + path_, rc), err);
    rc = avformat_find_stream_info(fmt_, nullptr);
    if (rc < 0) return fail(av_msg("探测 " + path_, rc), err);
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_ < 0) return fail(path_ + " 没有视频流", err);
    // 显式丢弃其它流（ZCam 的 RTSP 带一路 1536 kb/s 的 PCM 音频）。只在
    // av_read_frame 里 unref 掉是不够的：libavformat 仍会为它们解析与排队，
    // 而我们永远不消费，积压会让 read 的节奏越来越不稳。
    for (unsigned i = 0; i < fmt_->nb_streams; ++i)
      if (int(i) != stream_) fmt_->streams[i]->discard = AVDISCARD_ALL;
    AVStream* st = fmt_->streams[stream_];

    // NVDEC 支持 H.264 与 HEVC，两者的 cuvid 解码器名不同。现场给什么编码就用
    // 对应的那个，别写死 —— 相机侧一个设置就能从 h264 切到 h265。
    const char* name = cuvid_name(st->codecpar->codec_id);
    if (!name)
      return fail(path_ + " 的编码不是 H.264/HEVC，NVDEC 这条路只支持这两种", err);
    codec_ = name;                            // 字面量，生命期是整个进程
    const AVCodec* dec = avcodec_find_decoder_by_name(name);
    if (!dec) return fail(std::string("FFmpeg 里没有 ") + name + " 解码器", err);
    ctx_ = avcodec_alloc_context3(dec);
    if (!ctx_) return fail("分配解码器上下文失败", err);
    rc = avcodec_parameters_to_context(ctx_, st->codecpar);
    if (rc < 0) return fail(av_msg("拷贝流参数", rc), err);
    ctx_->hw_device_ctx = av_buffer_ref(hw_);
    // 不设 pkt_timebase 会打印 "Invalid pkt_timebase" 并按原样传时间戳
    ctx_->pkt_timebase = st->time_base;
    // 我们会把解码帧一直扣到拼接读完（见 StitchFrameSource 的 slot 回收），
    // 所以要让解码器多备这么多张 surface，否则它会等我们归还而卡住。
    // 用 extra_hw_frames 而不是已废弃的 "surfaces" 选项。
    ctx_->extra_hw_frames = kExtraHwFrames;
    rc = avcodec_open2(ctx_, dec, nullptr);
    if (rc < 0) return fail(av_msg(std::string("打开 ") + name, rc), err);

    const int w = st->codecpar->width, h = st->codecpar->height;
    // 重连后分辨率变了不能接着用：LUT 按单一源尺寸烘的，混着会越界采样。
    // 现场把相机 movfmt 从 4K 改成 1080P 就会走到这里，明确报出来。
    if (w_ && (w != w_ || h != h_))
      return fail(path_ + " 重连后变成 " + std::to_string(w) + "x" +
                      std::to_string(h) + "，与首次的 " + std::to_string(w_) +
                      "x" + std::to_string(h_) + " 不一致（LUT 按前者烘制）",
                  err);
    w_ = w;
    h_ = h;
    const AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    fps_ = r.den > 0 ? double(r.num) / double(r.den) : 0.0;
    total_ = st->nb_frames > 0 ? st->nb_frames : -1;   // 直播流没有总帧数
    flushed_ = false;
    read_fails_ = 0;
    healthy_ = true;
    return true;
  }

  bool fail(std::string msg, std::string& err) {
    close_input();
    err = std::move(msg);
    return false;
  }

  void close_input() noexcept {
    if (ctx_) avcodec_free_context(&ctx_);
    if (fmt_) avformat_close_input(&fmt_);
    stream_ = -1;
  }

  /// 掉线重连（只在直播路、只在解码线程里调）。重开 demux + 解码器，
  /// hw device context 与 surface 池不动，所以一次成功的重连约 1 秒。
  /// why 非空时并进首行日志（首次打开失败时用得上，掉线路径留空）。
  /// 返回 false 表示外部要求停止。
  bool reopen(const std::string& why = {}) {
    healthy_ = false;
    printf("[Stitch] %s %s，开始重连（其余相机继续出画，本路用上一帧顶住）\n",
           path_.c_str(), why.empty() ? "断流" : why.c_str());
    fflush(stdout);
    for (int attempt = 1;; ++attempt) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        if (cv_room_.wait_for(lk, std::chrono::milliseconds(kReconnectWaitMs),
                              [this] { return stop_; }))
          return false;
      }
      std::string err;
      if (try_open(err)) {
        ++reconnects_;
        printf("[Stitch] %s 重连成功（第 %lld 次，尝试 %d 回）\n", path_.c_str(),
               static_cast<long long>(reconnects_), attempt);
        fflush(stdout);
        return true;
      }
      // 每次都打会淹掉日志（相机重启要几十秒），按 1/5/25… 递减频率报进展
      if (attempt == 1 || attempt % 5 == 0) {
        printf("[Stitch] %s 重连第 %d 次未成功: %s\n", path_.c_str(), attempt,
               err.c_str());
        fflush(stdout);
      }
    }
  }

  /// 解一帧到 out（已 ref）；返回 false 表示流结束（直播路只在 stop 时如此）。
  bool decode_one(AVFrame* out) {
    for (;;) {
      int rc = avcodec_receive_frame(ctx_, out);
      if (rc == 0) {
        SWIM_CHECK(out->format == AV_PIX_FMT_CUDA,
                   path_ + " 解出的不是 CUDA 帧（hw_device_ctx 没生效？）");
        read_fails_ = 0;
        healthy_ = true;
        return true;
      }
      SWIM_CHECK(rc == AVERROR(EAGAIN) || rc == AVERROR_EOF,
                 av_msg("接收解码帧", rc));
      if (rc == AVERROR_EOF) {
        // 直播路的 EOF 是相机断开（RTSP 会话结束），重连而不是收摊
        if (!live_) return false;
        if (!reopen()) return false;
        continue;
      }
      rc = av_read_frame(fmt_, pkt_);
      if (rc < 0) {
        // 文件读完就是结束；直播流的读失败通常只是 socket 超时（我们下发了
        // timeout=5s），把它当 EOF 会让整条链路在第一次网络抖动时静默停掉 ——
        // 实测正是这样：lane 每 5 秒"结束"一次，端到端掉到 0.5 fps。
        // 所以 live 下先重试几次容抖动，仍不行就重连（本路重开，其余路不受影响）。
        if (live_) {
          if (rc != AVERROR_EOF && ++read_fails_ < kMaxReadFails) {
            printf("[Stitch] %s 读取失败(%d/%d): %s\n", path_.c_str(),
                   read_fails_, kMaxReadFails, av_msg("read", rc).c_str());
            fflush(stdout);
            continue;
          }
          if (!reopen()) return false;
          continue;
        }
        if (flushed_) return false;
        flushed_ = true;
        avcodec_send_packet(ctx_, nullptr);     // 冲刷解码器里剩下的帧
        continue;
      }
      read_fails_ = 0;
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
    // park() 过的路在这里补开（探完参数就把流放掉了，见 park）。开不了就走
    // 重连那条路：现场相机上电顺序不齐是常态，不该让整条链路起不来。
    if (parked_) {
      parked_ = false;
      std::string err;
      if (!try_open(err) && !reopen(err)) return finish(nullptr);
    }
    // 混合来源（既有直播流又有离线文件）时，离线路必须按帧率限速。
    // 否则它们会以 200+ fps 满速解码，把 NVDEC 与 libav 内部占满，直播路的
    // av_read_frame 拿不到调度 → socket 缓冲堆积 → 相机侧发送阻塞 →
    // read 返回 ETIMEDOUT(-138)。已用最小复现确认：只读不解不受影响，
    // 一旦同进程有 5 路满速 cuvid 解码，直播路 5 s 就超时一次。
    const auto period = std::chrono::duration<double, std::milli>(
        pace_ && pace_fps_ > 0 ? 1000.0 / pace_fps_ : 0.0);
    auto next_at = std::chrono::steady_clock::now();
    for (;;) {
      if (pace_) {
        next_at += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_at);
      }
      // 离线文件在这里背压：队列满就等消费者取走。丢帧会让结果少帧、与基线对不上。
      // 直播流**不能**在这里等 —— 见下面 push 处的说明。
      if (!live_) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_room_.wait(lk, [this] {
          return int(queue_.size()) < kLaneQueue || stop_;
        });
        if (stop_) break;
      } else {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_) break;
      }
      AVFrame* f = av_frame_alloc();
      SWIM_CHECK(f != nullptr, "分配 AVFrame 失败");
      // 无条件解一帧：直播流靠这里的 socket 阻塞自然定速，不需要（也不能）自己睡
      if (!decode_one(f)) {
        av_frame_free(&f);
        break;
      }
      std::lock_guard<std::mutex> lk(mu_);
      // 直播流的丢帧放在**入队时**而不是入队前。相机不会等我们，只要有一刻不读
      // socket，TCP 接收缓冲就堆积、相机侧发送阻塞，最终 read 超时
      // （实测 AVERROR(ETIMEDOUT)=-138，表现为整条链路先掉到 3 fps 再静默停住）。
      // 所以 producer 永远不阻塞、不休眠：一直读一直解，满了就把最旧的一帧丢掉。
      // 实时场景要的是最新画面，不是每一帧。
      while (int(queue_.size()) >= kLaneQueue) {
        AVFrame* old = queue_.front();
        queue_.pop();
        av_frame_free(&old);              // 归还 surface，否则解码器会缺帧
        ++dropped_;
      }
      queue_.push(f);
      ++decoded_;
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
  AVBufferRef*     hw_ = nullptr;            // 本路独占的 hw device context
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext*  ctx_ = nullptr;
  AVPacket*        pkt_ = nullptr;
  int     stream_ = -1, w_ = 0, h_ = 0;
  const char* codec_ = nullptr;             // cuvid 解码器名（字面量，不拥有）
  double  fps_ = 0;
  int64_t total_ = -1;
  bool    flushed_ = false;
  bool    live_ = false;                    // 直播流：满队列丢旧帧而不是背压
  bool    parked_ = false;                  // park() 过，等 loop() 补开
  bool    pace_ = false;                    // 离线路按帧率限速（混合来源时）
  double  pace_fps_ = 0;                    // 限速用的帧率（画布帧率，可被 --fps 覆盖）
  int     read_fails_ = 0;                  // 连续读失败次数（live 下容忍抖动）
  // 跨线程：解码线程写，消费者（next）读。不健康时 pop 立即返回 Stall，
  // 不必每帧陪它等 kLiveStallMs。
  std::atomic<bool>    healthy_{true};
  std::atomic<int64_t> reconnects_{0};      // 成功重连次数（运维要看的数）

  std::thread             worker_;
  mutable std::mutex      mu_;
  std::condition_variable cv_ready_, cv_room_;
  std::queue<AVFrame*>    queue_;
  bool                    stop_ = false, done_ = false;
  int64_t                 dropped_ = 0;     // 因追不上而丢弃的帧数（只在 live_）
  std::atomic<int64_t>    decoded_{0};      // 解出并入队的帧数（--show-fps 用）
  std::exception_ptr      err_;
};

/// 每路的输入地址，按 LUT 的相机顺序。两种来源共用它，下游不再区分。
using LaneUris = std::vector<std::string>;

/// 目录 -> 每台相机一个片段。一个相机恰好一个文件，多于一个必须报错 ——
/// 静默挑一个等于把错误的相机贴到网格上，症状是接缝错位而不是报错。
LaneUris uris_from_dir(const std::string& dir, const StitchLut& lut) {
  namespace fs = std::filesystem;
  SWIM_CHECK(fs::is_directory(dir), "不是目录: " + dir);
  LaneUris out;
  for (int i = 0; i < lut.lanes(); ++i) {
    const std::string& cam = lut.camera(i);
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
    out.push_back(hits[0]);
  }
  return out;
}

/// 相机清单文件 -> 每台相机一个地址。每行 `<相机>=<地址>`，`#` 起注释，空行忽略。
/// 用显式的 `相机=地址` 而不是「按行序对应」：现场六台相机的 IP 尾数与 mesh 顺序
/// 不同（20260730 那批实测是 cam3 cam2 cam1 cam4 cam5 cam6），按行序写迟早错位，
/// 而错位的症状是接缝错乱、不报错。
///
/// **允许只给一部分相机**：没列出的那路留空，画布上它的区域是黑的。分批上线
/// （先装两台）与单相机联调都要靠这个，否则本地只有一台相机时根本跑不起来。
LaneUris uris_from_list(const std::string& path, const StitchLut& lut) {
  std::ifstream f(path);
  SWIM_CHECK(f.good(), "无法打开相机清单 " + path);
  std::map<std::string, std::string> map;
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (!line.empty() && line.back() == '\r') line.pop_back();   // 允许 CRLF
    // 剥 UTF-8 BOM：这个文件是给人在记事本里改的，交付包的模板就带 BOM
    // （见 docs/windows.md 的编码约定）。不剥掉，第一行的相机名会带三个不可见
    // 字节而匹配不上，报错还指向「格式不对」。
    if (lineno == 1 && line.size() >= 3 &&
        static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF)
      line.erase(0, 3);
    const size_t begin = line.find_first_not_of(" \t");
    if (begin == std::string::npos || line[begin] == '#') continue;
    const size_t eq = line.find('=', begin);
    SWIM_CHECK(eq != std::string::npos,
               path + ":" + std::to_string(lineno) + " 需要 `相机=地址` 形式");
    std::string key = line.substr(begin, eq - begin);
    std::string val = line.substr(eq + 1);
    while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
    const size_t vb = val.find_first_not_of(" \t");
    val = vb == std::string::npos ? "" : val.substr(vb);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    SWIM_CHECK(map.emplace(key, val).second,
               path + " 里 " + key + " 出现了两次");
  }
  LaneUris out;
  int active = 0;
  for (int i = 0; i < lut.lanes(); ++i) {
    auto it = map.find(lut.camera(i));
    out.push_back(it == map.end() ? std::string() : it->second);
    if (!out.back().empty()) ++active;
  }
  SWIM_CHECK(active > 0,
             path + " 里没有一台相机与 LUT 对得上（LUT 需要: " + [&] {
               std::string s;
               for (int i = 0; i < lut.lanes(); ++i)
                 s += (i ? " " : "") + lut.camera(i);
               return s;
             }() + "）");
  return out;
}

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
  StitchFrameSource(const std::string& spec, bool spec_is_list,
                    const std::string& lut_path, double fps, bool rot180,
                    int ring)
      : ring_n_(ring), rot180_(rot180) {
    SWIM_CHECK(ring_n_ >= 3, "拼接画布环深须 >= 3");
    // 顺序要紧：av_hwdevice_ctx_create(AV_CUDA_USE_PRIMARY_CONTEXT) 要设置
    // primary context 的 flags，若 CUDA runtime API（cudaMalloc 等）已激活它就会
    // 失败（-129）。每路各建一份（见 NvdecLane 的注释），所以这里先建一个临时的
    // 把 flags 定住，再 load LUT（它会 cudaMalloc）。临时的随即释放 ——
    // primary context 的 flags 一旦设定就一直保持。
    AVBufferRef* seed = nullptr;
    const int rc = av_hwdevice_ctx_create(&seed, AV_HWDEVICE_TYPE_CUDA, "0",
                                          nullptr, AV_CUDA_USE_PRIMARY_CONTEXT);
    SWIM_CHECK(rc >= 0, av_msg("创建 CUDA 硬解上下文", rc) +
                            "（拼接源必须在任何 CUDA 分配之前构造）");

    lut_ = StitchLut::load(lut_path);
    const int n = lut_->lanes();
    const LaneUris uris = spec_is_list ? uris_from_list(spec, *lut_)
                                       : uris_from_dir(spec, *lut_);
    // lanes_ 与 LUT 的 lane 一一对应，缺的那路存 nullptr（画布上是黑的）。
    // 保持下标对齐而不是压紧数组，是因为 lane i 的查找表就是 lut 的第 i 条。
    for (int i = 0; i < n; ++i) {
      const std::string& uri = uris[size_t(i)];
      if (uri.empty()) {
        printf("[Stitch] lane %d = %-6s <- (未配置，该区域留黑)\n", i,
               lut_->camera(i).c_str());
        lanes_.push_back(nullptr);
        continue;
      }
      // 打开前先报地址：六路 RTSP 里若有一台不通，卡住的是哪一台要能一眼看出
      // （avformat 的连接超时是 5 s，没有这行就只是黑屏等着）。
      printf("[Stitch] lane %d = %-6s <- %s ...\n", i, lut_->camera(i).c_str(),
             uri.c_str());
      fflush(stdout);
      auto lane = std::make_unique<NvdecLane>(uri);
      // 探到的实况：分辨率 / 帧率 / 编码 / 是流还是文件。现场最常问的
      // 「我拉的到底是哪一路、多少帧率」在这一行里全了。
      printf("[Stitch] lane %d = %-6s %dx%d %.2f fps %s %s\n", i,
             lut_->camera(i).c_str(), lane->width(), lane->height(),
             lane->fps(), lane->codec(), lane->live() ? "直播流" : "离线片段");
      fflush(stdout);
      lanes_.push_back(std::move(lane));
    }
    av_buffer_unref(&seed);

    // 各路必须同分辨率：拼接表按单一源尺寸烘的，混着不同尺寸会越界采样。
    // 帧率取各路最大值（用于划水/速度的时间轴）；直播流的 total 是 -1，
    // 用 min 会让整体退化成 -1 —— 那正是想要的（流没有总帧数）。
    bool live = false;
    for (int i = 0; i < n; ++i) {
      if (!lanes_[i]) continue;
      SWIM_CHECK(lanes_[i]->width() == lut_->src_w() &&
                     lanes_[i]->height() == lut_->src_h(),
                 lanes_[i]->path() + " 是 " + std::to_string(lanes_[i]->width()) +
                     "x" + std::to_string(lanes_[i]->height()) + "，而 LUT 按 " +
                     std::to_string(lut_->src_w()) + "x" +
                     std::to_string(lut_->src_h()) + " 烘制");
      if (lanes_[i]->total() <= 0) live = true;
      total_ = total_ < 0 ? lanes_[i]->total()
                          : std::min(total_, lanes_[i]->total());
      fps_ = std::max(fps_, lanes_[i]->fps());
    }
    if (live) total_ = -1;
    if (fps_ <= 0) fps_ = 30.0;
    // --fps 覆盖：相机侧改成 4KP29.97 后流里仍可能报 59.94（实测 ZCam 的
    // avg_frame_rate 跟不上 movfmt 切换），时间轴按错的帧率走会让速度整体翻倍。
    // 只改时间轴与离线路限速，不影响解码。
    if (fps > 0) {
      printf("[Stitch] --fps %.3f 覆盖源自报的 %.3f（时间轴与限速按前者）\n", fps, fps_);
      fps_ = fps;
    }

    bytes_ = size_t(lut_->canvas_w()) * lut_->canvas_h() * 3;
    last_.resize(size_t(n), nullptr);
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
    printf("[Stitch] %d 路 NVDEC -> 画布 %dx%d，环 %d x %.1f MB，%.2f fps 总帧 %lld%s\n",
           n, lut_->canvas_w(), lut_->canvas_h(), ring_n_, double(bytes_) / 1e6,
           fps_, static_cast<long long>(total_),
           live ? "（含直播流：跟不上时丢旧帧）" : "");
    // 混合来源时给离线路限速，理由见 NvdecLane::loop()。全离线时不限速
    // （那是批处理，越快越好）；全直播时也不需要（各路本来就由相机定速）。
    bool any_file = false;
    for (auto& lane : lanes_) any_file = any_file || (lane && !lane->live());
    pace_ = live && any_file;
    if (pace_) printf("[Stitch] 混合来源：离线路按帧率限速，避免饿死直播路\n");
    // **解码线程推迟到第一次 next() 才起**，直播路的 socket 也先放掉（park）。
    // 构造完到第一次取帧之间，主线程还要烘 TRT engine —— 换代显卡的首次启动是
    // 几分钟。这期间若六路 RTSP 已经 PLAY 着没人读，TCP 缓冲堆满就会一路
    // -138 超时 + 重连刷屏（现场日志实测），engine 转完还得等六路各重连一次。
    // 探参数（分辨率/帧率/编码）必须在这之前做完，所以是「先探后放」。
    for (auto& lane : lanes_)
      if (lane && lane->live()) lane->park();
    if (live)
      printf("[Stitch] 相机已探明，先松开连接；等模型就绪再开拉（避免空转堆缓冲）\n");
    fflush(stdout);
  }

  ~StitchFrameSource() override {
    // 直播流的运维三项：丢了多少帧（追不上相机帧率）、重连了几次、整帧靠旧帧
    // 顶住了多少帧（掉线时长的直接度量）。持续增长都说明现场有问题。
    int64_t drops = 0, recon = 0;
    for (auto& lane : lanes_)
      if (lane && lane->live()) {
        drops += lane->dropped();
        recon += lane->reconnects();
      }
    if (drops > 0)
      printf("[Stitch] 直播流累计丢弃 %lld 帧（追不上相机帧率时的正常行为）\n",
             static_cast<long long>(drops));
    if (recon > 0 || held_ > 0)
      printf("[Stitch] 掉线重连 %lld 次，整帧沿用旧画面 %lld 帧\n",
             static_cast<long long>(recon), static_cast<long long>(held_));
    for (auto& lane : lanes_)
      if (lane) lane->stop();                    // 先停线程，再回收显存
    if (stream_) cudaStreamSynchronize(stream_);
    for (auto*& f : last_)
      if (f) av_frame_free(&f);
    for (auto& s : slots_) {
      for (auto*& f : s.frames)
        if (f) av_frame_free(&f);
      if (s.canvas) cudaFree(s.canvas);
      if (s.done) cudaEventDestroy(s.done);
      if (s.host_lanes) cudaFreeHost(s.host_lanes);
      if (s.dev_lanes) cudaFree(s.dev_lanes);
    }
    if (stream_) cudaStreamDestroy(stream_);
    lanes_.clear();                    // 每路自带 hw ctx，随它一起释放
    lut_.reset();
  }

  bool next(GpuFrame& out) override {
    // 解码线程在这里才起（构造时只探参数）。放在 next() 而不是构造函数里，是为了
    // 让「烘 engine 的那几分钟」发生在相机没被 PLAY 的时候，见构造函数末尾。
    if (!started_) {
      started_ = true;
      printf("[Stitch] 模型已就绪，开始取帧（%d 路解码线程）\n", lut_->lanes());
      fflush(stdout);
      stat_decoded_.assign(lanes_.size(), 0);   // 到帧率的差分基准从此刻起算
      stat_ms_ = now_ms();
      for (auto& lane : lanes_)
        if (lane) lane->start(pace_, fps_);
    }
    const int n = lut_->lanes();
    Slot& s = slots_[size_t(idx_ % ring_n_)];
    // 复用这一格前：等它上一轮的拼接读完源帧，然后把 surface 还给解码器
    SWIM_CUDA(cudaEventSynchronize(s.done));
    for (auto*& f : s.frames)
      if (f) av_frame_free(&f);

    int fresh = 0;                       // 本帧真正拿到新帧的路数
    for (int i = 0; i < n; ++i) {
      StitchLane lane = lut_->lanes_view()[size_t(i)];
      NvdecLane* src = lanes_[size_t(i)].get();
      AVFrame* f = nullptr;
      if (src) {
        // Stall（只可能是直播路）时用本路上一帧顶住：一台相机掉线不该让整块
        // 画布停住。还没有可顶的帧（开机时相机未就绪）就让这一路留黑，
        // 与「未配置的相机」同一条路 —— 绝不在这里空转等，否则 q/ESC 停不下来。
        const NvdecLane::Got got = src->pop(f);
        if (got == NvdecLane::Got::End) {   // 任一在用的路真结束即整体结束
          for (int j = 0; j < i; ++j)
            if (s.frames[size_t(j)]) av_frame_free(&s.frames[size_t(j)]);
          return false;
        }
        if (got == NvdecLane::Got::Frame) ++fresh;
        else if (last_[size_t(i)]) f = ref_of(last_[size_t(i)]);
      }
      if (!f) {
        // 未配置的相机：权重给 0，kernel 里这一路对累加无贡献（画布留黑）。
        // 指针置空是刻意的 —— 权重为 0 时 kernel 不会取样，留着野指针更危险。
        lane.weight = nullptr;
        lane.luma = lane.chroma = nullptr;
        lane.bw = lane.bh = 0;
        s.host_lanes[i] = lane;
        continue;
      }
      s.frames[size_t(i)] = f;
      // 留一份引用当「上一帧」。引用的是同一张 surface（不拷像素），代价是每路
      // 多扣一张 NV12（4K 约 12 MB），kExtraHwFrames 已含这份余量。
      if (src && src->live()) {
        if (last_[size_t(i)]) av_frame_free(&last_[size_t(i)]);
        last_[size_t(i)] = ref_of(f);
      }
      lane.luma = f->data[0];
      lane.chroma = f->data[1];
      lane.luma_pitch = f->linesize[0];
      lane.chroma_pitch = f->linesize[1];
      s.host_lanes[i] = lane;
    }
    // 一帧都没拿到新画面（唯一那台相机正在重连、或开机时还没出第一帧）：
    // 按帧率放行，别空转刷同一张画布把 CPU 跑满。pop 的 300 ms 超时只在
    // 「本路还没被标记不健康」时才会等，所以这一条是必需的兜底。
    if (fresh == 0) {
      ++held_;
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(
          fps_ > 0 ? 1000.0 / fps_ : 33.0));
    }
    SWIM_CUDA(cudaMemcpyAsync(s.dev_lanes, s.host_lanes,
                              sizeof(StitchLane) * size_t(n),
                              cudaMemcpyHostToDevice, stream_));
    launch_stitch(s.dev_lanes, n, static_cast<uint8_t*>(s.canvas),
                  lut_->canvas_w(), lut_->canvas_h(), lut_->src_w(),
                  lut_->src_h(), rot180_, stream_);
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

  /// 每路的**到帧率**（上次调用以来解出的帧数 / 墙钟）+ 三项累计计数。
  /// 端到端帧率掉下来时，这一行当场把责任分清：某一路明显低于其余路 = 那台相机
  /// 或它那段网络的问题；六路一起低 = 下游（NVDEC 吞吐 / 推理）跟不上。
  /// 从前这三项只在退出时打一次，掉帧发生在运行中却看不见，只能事后猜。
  std::string status() const override {
    const double t = now_ms();
    const double dt = (t - stat_ms_) / 1000.0;
    stat_ms_ = t;
    std::string s = "| 到帧";
    char buf[32];
    int64_t drops = 0, recon = 0;
    for (size_t i = 0; i < lanes_.size(); ++i) {
      const int64_t d = lanes_[i] ? lanes_[i]->decoded() : 0;
      snprintf(buf, sizeof buf, " %.1f",
               dt > 0 ? double(d - stat_decoded_[i]) / dt : 0.0);
      stat_decoded_[i] = d;
      s += buf;
      if (lanes_[i] && lanes_[i]->live()) {
        drops += lanes_[i]->dropped();
        recon += lanes_[i]->reconnects();
      }
    }
    snprintf(buf, sizeof buf, " 丢%lld 顶%lld 重连%lld",
             static_cast<long long>(drops), static_cast<long long>(held_),
             static_cast<long long>(recon));
    return s + buf;
  }

 private:
  /// 给同一张 surface 再加一个引用（不拷像素）。用于「掉线时顶住的上一帧」：
  /// 它与画布环里的那一份指向同一块显存，各自 av_frame_free 独立计数。
  static AVFrame* ref_of(const AVFrame* src) {
    AVFrame* f = av_frame_alloc();
    SWIM_CHECK(f != nullptr, "分配 AVFrame 失败");
    const int rc = av_frame_ref(f, src);
    if (rc < 0) {
      av_frame_free(&f);
      SWIM_CHECK(false, av_msg("引用上一帧", rc));
    }
    return f;
  }

  /// 画布环的一格：画布显存 + 拼接完成事件 + 该帧的 lane 描述与源帧。
  struct Slot {
    void*        canvas = nullptr;
    cudaEvent_t  done = nullptr;
    StitchLane*  host_lanes = nullptr;
    StitchLane*  dev_lanes = nullptr;
    std::vector<AVFrame*> frames;
  };

  std::unique_ptr<StitchLut>  lut_;
  std::vector<std::unique_ptr<NvdecLane>> lanes_;
  std::vector<AVFrame*>       last_;   // 每路最后一帧（只直播路留），掉线时顶住
  std::vector<Slot>           slots_;
  cudaStream_t stream_ = nullptr;
  size_t  bytes_ = 0;
  int     ring_n_ = 4;
  bool    rot180_ = false;             // 画布整体转 180°，在拼接落点上完成
  double  fps_ = 0;
  int64_t total_ = -1, idx_ = 0;
  int64_t held_ = 0;                   // 整帧都是「顶住的旧帧」的次数
  // status() 的差分基准。const 方法里要更新，故 mutable —— 它只被后处理线程
  // （--show-fps 的打印处）调用，单读者，不需要加锁。
  mutable double               stat_ms_ = 0;
  mutable std::vector<int64_t> stat_decoded_;
  bool    pace_ = false;               // 混合来源：离线路按帧率限速
  bool    started_ = false;            // 解码线程已起（首次 next() 时置位）
};

}  // namespace

std::unique_ptr<FrameSource> StitchSource::open(const std::string& spec,
                                                const std::string& lut_path,
                                                double fps, bool rot180,
                                                int ring) {
  // 目录 = 一批离线片段，文件 = 相机清单（每行 `相机=地址`，地址可为 rtsp://）。
  // 两者都只是「怎么给六个地址」，NvdecLane 内部不区分。
  const bool is_list = !std::filesystem::is_directory(spec);
  return std::make_unique<StitchFrameSource>(spec, is_list, lut_path, fps,
                                             rot180, ring);
}

}  // namespace swim
