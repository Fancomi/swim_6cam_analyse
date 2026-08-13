#include "swim/frame_source.h"

#ifdef _WIN32
// NOMINMAX：windows.h 的 min/max 宏会打断 std:: 与 OpenCV 的模板调用
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace swim {
namespace {

/// CUVID 的 H.264 8bit max_width/height。画布 5002 宽超限（实测 ffmpeg 报
/// "Video width 5002 not within range from 48 to 4096"），所以本文件没有 NVDEC
/// 实现：这不是缺功能，是硬件限制。HEVC 上限 8192 可以，但输入是 H.264。
constexpr int kNvdecMax = 4096;

/// 只读子进程管道（本文件只读不写；写侧见 main.cpp 的 Writer）。
///
/// Windows 的像素通道不用 _popen：它给的匿名管道缓冲只有几 KB，ffmpeg 每写满
/// 就得等我们来取，解码与搬运串成一串。实测同一条命令 _popen 约 16.7 ms/帧，
/// 自己 CreatePipe 给 128 MB 缓冲后 13.0 ms/帧 —— 已贴住 ffmpeg CLI 本身的
/// 13.5 ms/帧地板（`-f rawvideo -pix_fmt bgr24` 落 NUL 实测）。
/// 文本通道（ffprobe，几十字节）没有吞吐需求，仍用 _popen。
class Pipe {
 public:
  /// text=true 按行读文本（ffprobe）；否则读二进制像素 ——
  /// 后者在 Windows 必须二进制模式，文本模式会改写 0x0D/0x0A 毁掉像素。
  explicit Pipe(const std::string& cmd, bool text = false) {
#ifdef _WIN32
    if (text) { f_ = _popen(cmd.c_str(), "rt"); return; }
    spawn(cmd, 128u << 20);
#else
    (void)text;
    f_ = popen(cmd.c_str(), "r");
#endif
  }

  ~Pipe() {
    if (!f_) return;
#ifdef _WIN32
    if (!proc_) { _pclose(f_); return; }
    kill(proc_);                      // 顺序见 kill()：先处置子进程再撤读端
    fclose(f_);
#else
    pclose(f_);
#endif
  }

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;

  FILE* get() const { return f_; }
  explicit operator bool() const { return f_ != nullptr; }

 private:
#ifdef _WIN32
  /// 起子进程并把它的 stdout 接到一根 buf 字节的管道上。失败则 f_ 保持 nullptr。
  /// buf 只是内核给管道的容量上限提示，不预提交物理内存 ——
  /// 实际占用取决于「已写入尚未被读走」的字节数，稳态下就是几个 chunk。
  void spawn(const std::string& cmd, DWORD buf) {
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};   // 写端要可继承
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, buf)) return;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);   // 读端不给子进程

    STARTUPINFOA si{};
    si.cb = sizeof si;
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);     // 让 ffmpeg 的报错照旧可见
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> line(cmd.begin(), cmd.end());     // CreateProcessA 要可写缓冲
    line.push_back('\0');
    if (!CreateProcessA(nullptr, line.data(), nullptr, nullptr, TRUE, 0, nullptr,
                        nullptr, &si, &pi)) {
      CloseHandle(rd);
      CloseHandle(wr);
      return;
    }
    CloseHandle(wr);                                    // 只留子进程那份，否则读不到 EOF
    CloseHandle(pi.hThread);

    // 把 HANDLE 包成 FILE*，让 read_into 里的 fread 与 Linux 分支共用一份代码。
    // fd/FILE* 接管 rd 的所有权，之后只能由 fclose 释放（不能再 CloseHandle）。
    // 这两步失败时子进程已经起来了，必须就地收掉：析构看 f_==nullptr 就直接返回，
    // 指望它清理等于漏一个进程。
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(rd),
                                   _O_RDONLY | _O_BINARY);
    if (fd < 0)           { CloseHandle(rd); kill(pi.hProcess); return; }
    f_ = _fdopen(fd, "rb");
    if (!f_)              { _close(fd);      kill(pi.hProcess); return; }
    proc_ = pi.hProcess;                                // 交给析构
  }

  /// 处置子进程。**必须在 fclose(读端) 之前调**：先关读端的话，仍在写的 ffmpeg
  /// 会拿到管道错误并把 "Error muxing a packet / Error writing trailer" 一串刷到
  /// stderr —— 提前结束（--max-frames）时那是必然发生的假报错，会盖住真日志。
  /// 读到 EOF 时子进程已在退出，给 100 ms 让它自己走完；仍在写就直接终止。
  static void kill(HANDLE h) {
    if (WaitForSingleObject(h, 100) != WAIT_OBJECT_0) TerminateProcess(h, 1);
    CloseHandle(h);
  }
  HANDLE proc_ = nullptr;
#endif
  FILE* f_ = nullptr;
};

/// 解码预取骨架：device 环 + 每格锁页 host 缓冲 + 预取线程 + 背压。
/// 子类只需实现 read_into()「把一帧 BGR24 写进 dst」，H2D、环轮转、线程与
/// 异常传递全在这里，两条 CPU 解码路径不重复一行。
///
/// prefetch=true 时解码(13~17 ms/帧 @5002x2102)与推理重叠，否则串行会吃掉
/// 一半吞吐；环深度决定最多预取多少帧（同时也是背压上限）。
class PrefetchSource : public FrameSource {
 public:
  // 子类析构已 shutdown()，这里只是兜底（幂等），保证 worker 停了才回收缓冲
  ~PrefetchSource() override {
    shutdown();
    if (stream_) cudaStreamSynchronize(stream_);   // 可能还有 H2D 在飞
    for (auto* e : ev_) cudaEventDestroy(e);
    for (auto* p : dev_) cudaFree(p);
    for (auto* p : pin_) cudaFreeHost(p);
    if (stream_) cudaStreamDestroy(stream_);
  }

  bool next(GpuFrame& out) override {
    if (!prefetch_) {
      if (!fill(out)) return false;
      wait_ready(out);
      return true;
    }
    std::unique_lock<std::mutex> lk(mu_);
    // 谓词必须带 stop_：否则析构时若队列空且 done_ 未置，消费者永久卡住
    cv_ready_.wait(lk, [this] { return !queue_.empty() || done_ || stop_; });
    if (queue_.empty()) {
      if (err_) {                     // 把解码线程的异常搬到调用线程再抛
        auto e = err_;
        err_ = nullptr;
        std::rethrow_exception(e);
      }
      return false;
    }
    out = queue_.front();
    queue_.pop();
    cv_full_.notify_one();
    lk.unlock();                      // H2D 的等待不该占着锁
    wait_ready(out);
    return true;
  }

  int     width()  const override { return w_; }
  int     height() const override { return h_; }
  double  fps()    const override { return fps_; }
  int64_t total()  const override { return total_; }

 protected:
  /// 子类构造末尾调用：此时宽高已知才能算 bytes_，也才允许起线程
  /// （线程要调用子类的 read_into，早于子类构造完成即为未定义行为）。
  void start(int w, int h, double fps, int64_t total, int ring, bool prefetch) {
    SWIM_CHECK(w > 0 && h > 0, "输入尺寸非法");
    // ring<2 时背压谓词 queue_.size() < ring-1 恒为假 → 解码线程永久等待
    SWIM_CHECK(ring >= 2, "ring 须 >= 2");
    w_ = w; h_ = h;
    fps_ = fps > 0 ? fps : 30.0;      // 流媒体常读不到帧率
    total_ = total > 0 ? total : -1;
    ring_n_ = ring;
    prefetch_ = prefetch;
    bytes_ = size_t(w_) * h_ * 3;

    // device 环 + 每格一块锁页 host 缓冲（H2D 走 DMA，比普通内存快约一倍）
    // 每格配一个 event：H2D 只在这里入队，由 next() 在消费侧等 ——
    // 解码线程不等拷贝完成，30 MB 的 H2D(约 3 ms)就与下一帧解码重叠掉了。
    dev_.resize(ring_n_);
    pin_.resize(ring_n_);
    ev_.resize(ring_n_);
    for (int i = 0; i < ring_n_; ++i) {
      SWIM_CUDA(cudaMalloc(&dev_[i], bytes_));
      SWIM_CUDA(cudaHostAlloc(&pin_[i], bytes_, cudaHostAllocDefault));
      SWIM_CUDA(cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming));
    }
    SWIM_CUDA(cudaStreamCreate(&stream_));
    if (prefetch_) worker_ = std::thread([this] { decode_loop(); });
  }

  /// 子类析构第一件事必须调用它（幂等）：停掉线程，之后才能安全销毁子类自己的
  /// 解码器/管道 —— worker 调的是子类的 read_into，子类先没了就是纯虚调用崩溃。
  /// worker 可能正阻塞在 read_into 的 fread/cap_.read 上，要等它读完当前一帧
  /// （离线文件是毫秒级；live 流断流时取决于底层超时，与原实现一致）。
  void shutdown() {
    if (!worker_.joinable()) return;
    {   // stop_ 必须在锁内置位：否则 worker 刚求完谓词、还没 wait 时
        // notify 会丢失（lost wakeup），join 永久阻塞
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_full_.notify_all();
    cv_ready_.notify_all();
    worker_.join();
  }

  /// 填一帧 BGR24（恰好 bytes_ 字节）进 dst；返回 false 表示流结束。
  virtual bool read_into(void* dst) = 0;

  bool   prefetch() const { return prefetch_; }
  size_t bytes_ = 0;                  // 一帧字节数，供 read_into 用

 private:
  /// 解一帧到锁页缓冲并把 H2D 入队（不等它完成）。idx_ 只被「worker 或 next()」
  /// 其中一方推进（prefetch_ 决定是哪一方），故无需加锁。
  bool fill(GpuFrame& out) {
    const int slot = int(idx_ % ring_n_);
    // 该格上一轮的 H2D 必须已完成，否则会边传边覆写锁页缓冲。
    // 环深 >= 2 且消费者已等过 event，正常情况下这里不阻塞。
    SWIM_CUDA(cudaEventSynchronize(ev_[slot]));
    if (!read_into(pin_[slot])) {
      if (idx_ == 0) printf("[Source] 首帧读取失败：输入为空或解码器无输出\n");
      return false;
    }
    out.data = static_cast<uint8_t*>(dev_[slot]);
    out.w = w_; out.h = h_; out.index = idx_++;
    SWIM_CUDA(cudaMemcpyAsync(out.data, pin_[slot], bytes_,
                              cudaMemcpyHostToDevice, stream_));
    SWIM_CUDA(cudaEventRecord(ev_[slot], stream_));
    return true;
  }

  /// 消费侧等这一帧的 H2D 落地（下游在别的 stream 上用它）。
  void wait_ready(const GpuFrame& f) {
    SWIM_CUDA(cudaEventSynchronize(ev_[size_t(f.index % ring_n_)]));
  }

  void decode_loop() try {
    GpuFrame f;
    for (;;) {
      {   // 队列满则等消费者取走，形成背压（不会无限预取吃内存）
        std::unique_lock<std::mutex> lk(mu_);
        cv_full_.wait(lk, [this] {
          return int(queue_.size()) < ring_n_ - 1 || stop_;
        });
        if (stop_) break;
      }
      if (!fill(f)) break;
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(f);
      cv_ready_.notify_one();
    }
    mark_done(nullptr);
  } catch (...) {
    // 异常在非主线程逃逸会直接 std::terminate（进程无声死掉），
    // 所以存下来交给 next() 在调用线程 rethrow。
    mark_done(std::current_exception());
  }

  void mark_done(std::exception_ptr e) {
    std::lock_guard<std::mutex> lk(mu_);
    err_  = e;
    done_ = true;
    cv_ready_.notify_all();
  }

  std::vector<void*> dev_, pin_;
  std::vector<cudaEvent_t> ev_;             // 每格一个：标记该格 H2D 完成
  cudaStream_t stream_   = nullptr;
  int          ring_n_   = 4;
  bool         prefetch_ = true;
  int      w_ = 0, h_ = 0;
  double   fps_ = 30.0;
  int64_t  total_ = -1, idx_ = 0;

  std::thread             worker_;
  std::mutex              mu_;              // 保护以下四个成员
  std::condition_variable cv_ready_, cv_full_;
  std::queue<GpuFrame>    queue_;
  bool                    stop_ = false, done_ = false;
  std::exception_ptr      err_;
};

/// ffprobe 探到的流信息。ok=false 表示不可用（ffprobe 缺失/字段缺失/尺寸非法），
/// 调用方据此回退 OpenCV。
struct Probe {
  bool    ok = false;
  int     w = 0, h = 0;
  double  fps = 0;
  int64_t total = -1;
};

/// 用 ffprobe 探流。输出为每行一个值，顺序 = -show_entries 的声明顺序
/// （实测与命令行里字段先后无关，故按固定顺序解析是安全的）：
///   width / height / r_frame_rate(num/den) / nb_frames(可能是 N/A)
Probe probe(const std::string& uri) {
  const std::string cmd =
      "ffprobe -v error -select_streams v:0 -show_entries "
      "stream=width,height,r_frame_rate,nb_frames "
      "-of default=noprint_wrappers=1:nokey=1 -i \"" + uri + "\"";
  Pipe p(cmd, /*text=*/true);
  if (!p) return {};

  char line[256];
  std::string v[4];
  int n = 0;
  while (n < 4 && fgets(line, sizeof line, p.get())) {
    std::string s(line);
    // 去掉行尾 \r\n（Windows 下 ffprobe 输出 CRLF）与首尾空白
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
      s.pop_back();
    if (!s.empty()) v[n++] = s;
  }
  if (n < 4) return {};

  Probe r;
  try {
    r.w = std::stoi(v[0]);
    r.h = std::stoi(v[1]);
    const size_t slash = v[2].find('/');          // r_frame_rate 是 num/den
    const double num = std::stod(v[2].substr(0, slash));
    const double den = slash == std::string::npos ? 1.0
                                                  : std::stod(v[2].substr(slash + 1));
    r.fps = den > 0 ? num / den : 0;
    r.total = v[3] == "N/A" ? -1 : std::stoll(v[3]);   // 流媒体无总帧数
  } catch (...) {
    return {};                                    // 字段不是数字，按不可用处理
  }
  r.ok = r.w > 0 && r.h > 0;
  return r;
}

/// ffmpeg 管道解码：子进程解到 rawvideo bgr24，本进程 fread 直接读进锁页缓冲，
/// 比 OpenCV 路径省掉一次整帧 memcpy（5002x2102 每帧 30 MB）。
/// 实测 13.0 ms/帧，对比 OpenCV(MSMF) 的 16.9 ms/帧；ffmpeg CLI 自身地板是
/// 13.5 ms/帧（解码 11.2 + swscale 转 bgr24 2.3），即本实现已榨干这条路。
/// 附带好处：解码在独立进程，与本进程的 GPU 推理天然并行。
class FfmpegSource final : public PrefetchSource {
 public:
  FfmpegSource(const std::string& uri, const Probe& pr, int ring, bool prefetch)
      // -noautorotate：保证输出帧尺寸恒等于 ffprobe 报的 stream 宽高。带旋转元数据
      // 时自动旋转会输出 h×w（字节数相同！），按 w×h 解读就是整帧错切且无从察觉。
      : pipe_("ffmpeg -hide_banner -loglevel error -nostdin -noautorotate -i \"" +
              uri + "\" -map 0:v:0 -f rawvideo -pix_fmt bgr24 pipe:1") {
    SWIM_CHECK(bool(pipe_), "无法启动 ffmpeg 解码管道（PATH 里有 ffmpeg 吗？）");
    start(pr.w, pr.h, pr.fps, pr.total, ring, prefetch);
  }
  // 先停线程再关管道：worker 可能正阻塞在 fread 上
  ~FfmpegSource() override { shutdown(); }

  const char* backend() const override {
    return prefetch() ? "ffmpeg-pipe(prefetch)" : "ffmpeg-pipe";
  }

 private:
  bool read_into(void* dst) override {
    // 管道会短读：一次 fread 不保证读满一帧，必须循环累加。
    // 单次请求上限 kChunk 是实测调出来的：一次性请求整帧 31.5 MB 反而只有
    // 约 760 MB/s，按 1 MB 分次请求约 1.4 GB/s。
    auto* p = static_cast<uint8_t*>(dst);
    for (size_t got = 0; got < bytes_;) {
      const size_t want = std::min(bytes_ - got, kChunk);
      const size_t k = fread(p + got, 1, want, pipe_.get());
      if (k == 0) return false;       // EOF 或出错；不完整帧按流结束丢弃
      got += k;
    }
    return true;
  }

  static constexpr size_t kChunk = 1u << 20;
  Pipe pipe_;
};

/// OpenCV 解码 + H2D。VideoCapture 同时支持文件与 rtsp/rtmp，因此离线与 live
/// 共用同一实现，差别只在 total()。作为 ffmpeg 探测失败时的兜底：
/// 本机 vcpkg 版 OpenCV 的 videoio 只有 MSMF（未编 ffmpeg 特性），实测
/// 16.9 ms/帧，比 ffmpeg 管道慢约 30%，且多一次整帧 memcpy。
///
/// 注意这条路径**不只是慢，像素值也不同**：MSMF 的 YUV→BGR 矩阵与 swscale
/// 不一致，同一帧逐像素平均差 4.5、最大 30（拟合 ff ≈ 0.965*msmf + 6.8），
/// 会带来检测置信度与 track 数的小幅漂移。而 ffmpeg 管道路径与 Python 的
/// cv2.VideoCapture（opencv-python 自带 ffmpeg）逐字节相同。
/// 所以：与 Python 对照数值时必须走 Auto，本路径只用于兜底/排障。
class CpuSource final : public PrefetchSource {
 public:
  CpuSource(const std::string& uri, int ring, bool prefetch) {
    SWIM_CHECK(cap_.open(uri), "无法打开输入 " + uri);
    const double n = cap_.get(cv::CAP_PROP_FRAME_COUNT);
    start(int(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
          int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
          cap_.get(cv::CAP_PROP_FPS),
          n > 0 ? int64_t(n) : -1,          // 流媒体读不到帧数
          ring, prefetch);
  }
  ~CpuSource() override { shutdown(); }     // 先停线程再释放 cap_

  const char* backend() const override {
    return prefetch() ? "opencv(prefetch)" : "opencv";
  }

 private:
  bool read_into(void* dst) override {
    if (!cap_.read(mat_) || mat_.empty()) return false;
    SWIM_CHECK(mat_.isContinuous() && mat_.type() == CV_8UC3, "帧格式非 BGR8 连续");
    // 尺寸必须与首帧一致：分辨率中途变化时 mat_ 会小于 bytes_，
    // 不查就是照着 bytes_ 越界读（原实现的隐患）。
    SWIM_CHECK(mat_.total() * 3 == bytes_, "帧尺寸与首帧不一致");
    std::memcpy(dst, mat_.data, bytes_);      // VideoCapture 只能写自己的 Mat
    return true;
  }

  cv::VideoCapture cap_;
  cv::Mat          mat_;
};

/// device 端环形帧缓冲：一次性分配，之后只轮转，避免运行时分配抖动。
/// 仅 RawSource 用（预取路径的环在 PrefetchSource 里，还带配对的锁页缓冲）。
class Ring {
 public:
  Ring(int w, int h, int n) : bytes_(size_t(w) * h * 3) {
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
  size_t bytes_;
  std::vector<void*> bufs_;
  size_t cursor_ = 0;
};

/// 外部直供：拼接程序把已在显存的画布喂进来。零解码，一次 D2D 拷贝
/// （拷贝是必要的：调用方的缓冲随时会被下一帧覆写，环缓冲负责隔离）。
/// 单生产者（push）+ 单消费者（next），用一把 mutex 保护极短的临界区即可。
class RawSource final : public FrameSource {
 public:
  RawSource(int w, int h, double fps, int ring) : w_(w), h_(h), fps_(fps) {
    // ring>=2：消费者手上那格不能同时被 push 覆写
    SWIM_CHECK(w > 0 && h > 0 && ring >= 2, "raw 源参数非法（ring 须 >= 2）");
    ring_ = std::make_unique<Ring>(w, h, ring);
    SWIM_CUDA(cudaStreamCreate(&stream_));
  }
  ~RawSource() override {
    if (stream_) cudaStreamDestroy(stream_);
    if (dropped_) printf("[Source] raw 源丢帧 %lld（消费慢于 push）\n",
                         static_cast<long long>(dropped_));
  }

  bool push(const void* src, bool src_on_device) override {
    // 拷贝在锁外做：cudaMemcpy 30 MB 约 3 ms，不该阻塞 next()
    auto* dst = ring_->acquire();
    SWIM_CUDA(cudaMemcpyAsync(dst, src, ring_->bytes(),
                              src_on_device ? cudaMemcpyDeviceToDevice
                                            : cudaMemcpyHostToDevice, stream_));
    SWIM_CUDA(cudaStreamSynchronize(stream_));
    std::lock_guard<std::mutex> lk(mu_);
    const bool overwrote = has_;      // 上一帧还没被取走，本次覆盖它
    pending_ = dst;
    has_ = true;
    if (overwrote) ++dropped_;
    return !overwrote;                // 调用方可据此限速
  }

  bool next(GpuFrame& out) override {
    std::lock_guard<std::mutex> lk(mu_);
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
  std::mutex   mu_;                   // 保护 pending_/has_/idx_
  void*        pending_ = nullptr;
  bool         has_     = false;
  int64_t      idx_     = 0, dropped_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> FrameSource::open(const std::string& uri,
                                              DecoderPref pref, int ring,
                                              bool prefetch) {
  const Probe pr = pref == DecoderPref::Cpu ? Probe{} : probe(uri);

  // Nvdec 取值只为兼容 --decoder nvdec，这里把「为什么被忽略」说清楚。
  if (pref == DecoderPref::Nvdec) {
    if (pr.ok && (pr.w > kNvdecMax || pr.h > kNvdecMax))
      printf("[Source] nvdec 忽略：%dx%d 超出 CUVID H.264 8bit 上限 %d，"
             "改走 CPU 解码\n", pr.w, pr.h, kNvdecMax);
    else
      printf("[Source] nvdec 忽略：本构建未启用 cudacodec，改走 CPU 解码\n");
  }

  // 首选 ffmpeg 管道（13.0 ms/帧，且与 Python 的 cv2 逐字节一致）；
  // 探测不到才回退 OpenCV/MSMF（16.9 ms/帧，像素值与 swscale 有差异）。
  if (pr.ok) return std::make_unique<FfmpegSource>(uri, pr, ring, prefetch);
  if (pref != DecoderPref::Cpu)
    printf("[Source] ffprobe 探测失败，回退 OpenCV 解码（更慢，且像素值与 ffmpeg 不同）\n");
  return std::make_unique<CpuSource>(uri, ring, prefetch);
}

std::unique_ptr<FrameSource> FrameSource::raw(int w, int h, double fps, int ring) {
  return std::make_unique<RawSource>(w, h, fps, ring);
}

}  // namespace swim
