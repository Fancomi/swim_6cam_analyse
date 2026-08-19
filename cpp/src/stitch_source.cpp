// 六路 4K -> NVDEC -> CUDA 拼接 -> GpuFrame。设计取舍见 include/swim/stitch.h。
#include <algorithm>
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
/// 直播流连续读失败多少次才认定断流。单次失败通常只是 socket 超时（我们下发了
/// timeout=5s），当场判 EOF 会让整条链路在第一次网络抖动时静默停掉。
constexpr int kMaxReadFails = 20;

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
class NvdecLane {
 public:
  explicit NvdecLane(const std::string& uri)
      : path_(uri), live_(is_stream(uri)) {
    int rc = av_hwdevice_ctx_create(&hw_, AV_HWDEVICE_TYPE_CUDA, "0", nullptr,
                                    AV_CUDA_USE_PRIMARY_CONTEXT);
    SWIM_CHECK(rc >= 0, av_msg("创建 CUDA 硬解上下文", rc));
    AVDictionary* opt = nullptr;
    open_options(uri, &opt);
    rc = avformat_open_input(&fmt_, uri.c_str(), nullptr, &opt);
    av_dict_free(&opt);
    SWIM_CHECK(rc >= 0, av_msg("打开 " + uri, rc));
    rc = avformat_find_stream_info(fmt_, nullptr);
    SWIM_CHECK(rc >= 0, av_msg("探测 " + uri, rc));
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    SWIM_CHECK(stream_ >= 0, uri + " 没有视频流");
    // 显式丢弃其它流（ZCam 的 RTSP 带一路 1536 kb/s 的 PCM 音频）。只在
    // av_read_frame 里 unref 掉是不够的：libavformat 仍会为它们解析与排队，
    // 而我们永远不消费，积压会让 read 的节奏越来越不稳。
    for (unsigned i = 0; i < fmt_->nb_streams; ++i)
      if (int(i) != stream_) fmt_->streams[i]->discard = AVDISCARD_ALL;
    AVStream* st = fmt_->streams[stream_];

    // NVDEC 支持 H.264 与 HEVC，两者的 cuvid 解码器名不同。现场给什么编码就用
    // 对应的那个，别写死 —— 相机侧一个设置就能从 h264 切到 h265。
    const char* name = cuvid_name(st->codecpar->codec_id);
    SWIM_CHECK(name != nullptr,
               uri + " 的编码不是 H.264/HEVC，NVDEC 这条路只支持这两种");
    const AVCodec* dec = avcodec_find_decoder_by_name(name);
    SWIM_CHECK(dec != nullptr, std::string("FFmpeg 里没有 ") + name + " 解码器");
    ctx_ = avcodec_alloc_context3(dec);
    SWIM_CHECK(ctx_ != nullptr, "分配解码器上下文失败");
    rc = avcodec_parameters_to_context(ctx_, st->codecpar);
    SWIM_CHECK(rc >= 0, av_msg("拷贝流参数", rc));
    ctx_->hw_device_ctx = av_buffer_ref(hw_);
    // 不设 pkt_timebase 会打印 "Invalid pkt_timebase" 并按原样传时间戳
    ctx_->pkt_timebase = st->time_base;
    // 我们会把解码帧一直扣到拼接读完（见 StitchFrameSource 的 slot 回收），
    // 所以要让解码器多备这么多张 surface，否则它会等我们归还而卡住。
    // 用 extra_hw_frames 而不是已废弃的 "surfaces" 选项。
    ctx_->extra_hw_frames = kExtraHwFrames;
    rc = avcodec_open2(ctx_, dec, nullptr);
    SWIM_CHECK(rc >= 0, av_msg(std::string("打开 ") + name, rc));

    w_ = st->codecpar->width;
    h_ = st->codecpar->height;
    const AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    fps_ = r.den > 0 ? double(r.num) / double(r.den) : 0.0;
    total_ = st->nb_frames > 0 ? st->nb_frames : -1;   // 直播流没有总帧数
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
    if (hw_) av_buffer_unref(&hw_);        // 必须在解码器之后
  }

  NvdecLane(const NvdecLane&) = delete;
  NvdecLane& operator=(const NvdecLane&) = delete;

  void start(bool pace) {
    pace_ = pace && !live_;
    worker_ = std::thread([this] { loop(); });
  }

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
  bool    live()  const { return live_; }
  int64_t dropped() const {                 // 只在 live_ 下非零
    std::lock_guard<std::mutex> lk(mu_);
    return dropped_;
  }
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
      rc = av_read_frame(fmt_, pkt_);
      if (rc < 0) {
        // 文件读完就是结束；直播流的读失败通常只是 socket 超时（我们下发了
        // timeout=5s），把它当 EOF 会让整条链路在第一次网络抖动时静默停掉 ——
        // 实测正是这样：lane 每 5 秒"结束"一次，端到端掉到 0.5 fps。
        // 所以 live 下重试，只有连续多次才认定断流。
        if (live_ && rc != AVERROR_EOF && ++read_fails_ < kMaxReadFails) {
          if (read_fails_ == 1)
            printf("[Stitch] %s 读取失败(%d)，重试中: %s\n", path_.c_str(), rc,
                   av_msg("read", rc).c_str());
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
    // 混合来源（既有直播流又有离线文件）时，离线路必须按帧率限速。
    // 否则它们会以 200+ fps 满速解码，把 NVDEC 与 libav 内部占满，直播路的
    // av_read_frame 拿不到调度 → socket 缓冲堆积 → 相机侧发送阻塞 →
    // read 返回 ETIMEDOUT(-138)。已用最小复现确认：只读不解不受影响，
    // 一旦同进程有 5 路满速 cuvid 解码，直播路 5 s 就超时一次。
    const auto period = std::chrono::duration<double, std::milli>(
        pace_ && fps_ > 0 ? 1000.0 / fps_ : 0.0);
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
  double  fps_ = 0;
  int64_t total_ = -1;
  bool    flushed_ = false;
  bool    live_ = false;                    // 直播流：满队列丢旧帧而不是背压
  bool    pace_ = false;                    // 离线路按帧率限速（混合来源时）
  int     read_fails_ = 0;                  // 连续读失败次数（live 下容忍抖动）

  std::thread             worker_;
  mutable std::mutex      mu_;
  std::condition_variable cv_ready_, cv_room_;
  std::queue<AVFrame*>    queue_;
  bool                    stop_ = false, done_ = false;
  int64_t                 dropped_ = 0;     // 因追不上而丢弃的帧数（只在 live_）
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
                    const std::string& lut_path, int ring)
      : ring_n_(ring) {
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
      printf("[Stitch] lane %d = %-6s <- %s\n", i, lut_->camera(i).c_str(),
             uri.c_str());
      lanes_.push_back(std::make_unique<NvdecLane>(uri));
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
    printf("[Stitch] %d 路 NVDEC -> 画布 %dx%d，环 %d x %.1f MB，%.2f fps 总帧 %lld%s\n",
           n, lut_->canvas_w(), lut_->canvas_h(), ring_n_, double(bytes_) / 1e6,
           fps_, static_cast<long long>(total_),
           live ? "（含直播流：跟不上时丢旧帧）" : "");
    // 混合来源时给离线路限速，理由见 NvdecLane::loop()。全离线时不限速
    // （那是批处理，越快越好）；全直播时也不需要（各路本来就由相机定速）。
    bool any_file = false;
    for (auto& lane : lanes_) any_file = any_file || (lane && !lane->live());
    const bool pace = live && any_file;
    if (pace) printf("[Stitch] 混合来源：离线路按帧率限速，避免饿死直播路\n");
    for (auto& lane : lanes_)
      if (lane) lane->start(pace);
  }

  ~StitchFrameSource() override {
    // 直播流丢了多少帧是运维要看的数：持续增长说明这台机器追不上相机帧率
    int64_t drops = 0;
    for (auto& lane : lanes_)
      if (lane && lane->live()) drops += lane->dropped();
    if (drops > 0)
      printf("[Stitch] 直播流累计丢弃 %lld 帧（追不上相机帧率时的正常行为）\n",
             static_cast<long long>(drops));
    for (auto& lane : lanes_)
      if (lane) lane->stop();                    // 先停线程，再回收显存
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
    lanes_.clear();                    // 每路自带 hw ctx，随它一起释放
    lut_.reset();
  }

  bool next(GpuFrame& out) override {
    const int n = lut_->lanes();
    Slot& s = slots_[size_t(idx_ % ring_n_)];
    // 复用这一格前：等它上一轮的拼接读完源帧，然后把 surface 还给解码器
    SWIM_CUDA(cudaEventSynchronize(s.done));
    for (auto*& f : s.frames)
      if (f) av_frame_free(&f);

    for (int i = 0; i < n; ++i) {
      StitchLane lane = lut_->lanes_view()[size_t(i)];
      if (!lanes_[size_t(i)]) {
        // 未配置的相机：权重给 0，kernel 里这一路对累加无贡献（画布留黑）。
        // 指针置空是刻意的 —— 权重为 0 时 kernel 不会取样，留着野指针更危险。
        lane.weight = nullptr;
        lane.luma = lane.chroma = nullptr;
        lane.bw = lane.bh = 0;
        s.host_lanes[i] = lane;
        continue;
      }
      AVFrame* f = lanes_[size_t(i)]->pop();
      if (!f) {                                  // 任一在用的路结束即整体结束
        for (int j = 0; j < i; ++j)
          if (s.frames[size_t(j)]) av_frame_free(&s.frames[size_t(j)]);
        return false;
      }
      s.frames[size_t(i)] = f;
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

std::unique_ptr<FrameSource> StitchSource::open(const std::string& spec,
                                                const std::string& lut_path,
                                                int ring) {
  // 目录 = 一批离线片段，文件 = 相机清单（每行 `相机=地址`，地址可为 rtsp://）。
  // 两者都只是「怎么给六个地址」，NvdecLane 内部不区分。
  const bool is_list = !std::filesystem::is_directory(spec);
  return std::make_unique<StitchFrameSource>(spec, is_list, lut_path, ring);
}

}  // namespace swim
