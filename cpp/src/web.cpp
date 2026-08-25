#include "swim/web.h"

// winsock2.h 必须在 windows.h 之前：WIN32_LEAN_AND_MEAN 排除了老 winsock.h，
// 但不会替我们引入 winsock2，顺序错了会拿到「重定义 sockaddr」一串错误。
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
using socket_t = SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t INVALID_SOCKET = -1;
#define closesocket ::close
#define SD_BOTH SHUT_RDWR
#endif

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "swim/common.h"

namespace swim {

/// 单页前端，定义在 web_ui.cpp（HTML/CSS/JS 与 C++ 分文件，改样式不必翻这里）。
extern const char kIndexHtml[];

namespace {

// ── socket 小工具 ─────────────────────────────────────────────────────────
int last_err() {
#ifdef _WIN32
  return WSAGetLastError();
#else
  return errno;
#endif
}

/// 收发都设超时：卡住的客户端不能把连接线程永久钉在 send 里，否则析构 join 不回来。
void set_timeout(socket_t s, int opt, int ms) {
#ifdef _WIN32
  DWORD v = DWORD(ms);
  ::setsockopt(s, SOL_SOCKET, opt, reinterpret_cast<const char*>(&v), sizeof v);
#else
  timeval v{ms / 1000, (ms % 1000) * 1000};
  ::setsockopt(s, SOL_SOCKET, opt, &v, sizeof v);
#endif
}

bool send_all(socket_t s, const void* p, size_t n) {
  const char* q = static_cast<const char*>(p);
  while (n) {
    const int chunk = int(n < (1u << 18) ? n : (1u << 18));
    const int k = ::send(s, q, chunk, 0);
    if (k <= 0) return false;
    q += k;
    n -= size_t(k);
  }
  return true;
}
bool send_str(socket_t s, const std::string& v) {
  return send_all(s, v.data(), v.size());
}

/// 一次性响应（含 Connection: close，浏览器不会复用这条连接）。
bool send_body(socket_t s, const char* type, const std::string& body) {
  return send_str(s, std::string("HTTP/1.1 200 OK\r\nContent-Type: ") + type +
                         "\r\nCache-Control: no-store\r\nContent-Length: " +
                         std::to_string(body.size()) +
                         "\r\nConnection: close\r\n\r\n") &&
         send_str(s, body);
}

// ── 「只保留最新一份」的广播格 ─────────────────────────────────────────────
/// 生产者 set() 覆盖上一份并递增序号，消费者按自己见过的序号等下一份。
/// 语义就是「慢客户端丢帧」：set 只做一次赋值 + notify，绝不阻塞（它跑在后处理
/// 线程上，阻塞会把背压一路顶到解码线程）。
///
/// 两侧都用回调在锁内操作值，于是单一实现同时服务两种取法：整帧（唯一消费者，
/// swap 走缓冲、零拷贝）与 JPEG（多消费者，必须各拷一份）。
template <typename T>
class Latest {
 public:
  template <typename F>
  void set(F&& fill) {
    {
      std::lock_guard lk(mu_);
      fill(v_);
      ++seq_;
    }
    cv_.notify_all();
  }
  /// 等到序号超过 seq，在锁内调 take(v_) 取值并更新 seq。返回 false = 已停止。
  template <typename F>
  bool wait(uint64_t& seq, F&& take) {
    std::unique_lock lk(mu_);
    cv_.wait(lk, [&] { return seq_ > seq || stop_; });
    if (stop_) return false;
    take(v_);
    seq = seq_;
    return true;
  }
  void stop() {
    { std::lock_guard lk(mu_); stop_ = true; }
    cv_.notify_all();
  }
  /// 当前序号。新消费者应从这里起等：门控（无人观看即不发布）会把上一次的值
  /// 留在格子里，从 0 起等会先收到那份陈旧帧（跟随流换人后尤其明显）。
  uint64_t seq() const {
    std::lock_guard lk(mu_);
    return seq_;
  }

 private:
  mutable std::mutex      mu_;
  std::condition_variable cv_;
  T        v_{};
  uint64_t seq_  = 0;
  bool     stop_ = false;
};

/// 一帧待编码的画面（BGR packed）。
struct Raw {
  std::vector<uint8_t> bgr;
  int w = 0, h = 0;
};

/// 一路 MJPEG：调用方发布原始画面，编码线程缩放 + JPEG 再广播给各连接。
///
/// 缩放与编码刻意放在独立线程：5002 宽画布缩到 1280 要 4 ms、JPEG 1.5 ms，
/// 摆在后处理线程上会吃掉六分之一的帧预算；发布侧只付一次 memcpy（31.5 MB
/// 约 1.4 ms）。无人观看时线程停在条件变量上，不占 CPU。
class MjpegStream {
 public:
  MjpegStream(int quality, int max_width)
      : max_w_(max_width), par_{cv::IMWRITE_JPEG_QUALITY, quality},
        th_([this] { encode_loop(); }) {}
  ~MjpegStream() {
    stop();
    if (th_.joinable()) th_.join();
  }
  /// 唤醒编码线程与所有正在推流的连接线程。析构前必须先调它再 join 连接 ——
  /// 否则连接线程还停在 jpg_.wait 上，join 永远回不来。幂等。
  void stop() {
    raw_.stop();
    jpg_.stop();
  }

  void publish(const uint8_t* bgr, int w, int h) {
    if (!viewers_ || w <= 0 || h <= 0) return;   // 没人看就连拷贝都不做
    const size_t n = size_t(w) * h * 3;
    raw_.set([&](Raw& r) {
      r.bgr.assign(bgr, bgr + n);                // 容量够时不重新分配
      r.w = w;
      r.h = h;
    });
  }
  bool wanted() const { return viewers_ > 0; }

  /// 在连接线程里推流到客户端结束。
  void serve(socket_t s) {
    if (!send_str(s, "HTTP/1.1 200 OK\r\nCache-Control: no-store\r\n"
                     "Connection: close\r\nContent-Type: "
                     "multipart/x-mixed-replace; boundary=" +
                         std::string(kBoundary) + "\r\n\r\n"))
      return;
    ++viewers_;
    // 已有别人在看 -> 从 0 起等，立刻拿到当前那一帧，不必空等一帧；自己是第一个
    // -> 从当前序号起等，否则先收到的是门控期间（无人观看即不发布）留在格子里
    // 的陈旧帧 —— 跟随流换人时那一帧属于上一个人，很扎眼。
    uint64_t seq = viewers_ > 1 ? 0 : jpg_.seq();
    std::vector<uint8_t> buf;
    char hdr[128];
    for (;;) {
      if (!jpg_.wait(seq, [&](const std::vector<uint8_t>& v) { buf = v; })) break;
      if (buf.empty()) continue;
      snprintf(hdr, sizeof hdr,
               "--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
               kBoundary, buf.size());
      if (!send_str(s, hdr) || !send_all(s, buf.data(), buf.size()) ||
          !send_str(s, "\r\n"))
        break;
    }
    --viewers_;
  }

 private:
  void encode_loop() {
    uint64_t seq = 0;
    Raw r;
    cv::Mat small;
    std::vector<uint8_t> out;
    // swap 而非拷贝：整帧 31.5 MB，且这里是 raw_ 的唯一消费者。
    // 换出去的是上一轮的缓冲，于是两块缓冲被反复复用，运行期不再分配。
    const auto take = [&r](Raw& v) { std::swap(r, v); };
    while (raw_.wait(seq, take)) {
      if (r.w <= 0 || r.h <= 0 || r.bgr.size() != size_t(r.w) * r.h * 3) continue;
      const cv::Mat m(r.h, r.w, CV_8UC3, r.bgr.data());
      const cv::Mat* src = &m;
      if (r.w > max_w_) {
        // INTER_AREA：4 倍下采样用 LINEAR 会明显走样（细线闪烁），
        // 而这条路在独立线程上，多花的 4 ms 不占帧预算
        cv::resize(m, small, {max_w_, std::max(1, r.h * max_w_ / r.w)}, 0, 0,
                   cv::INTER_AREA);
        src = &small;
      }
      if (!cv::imencode(".jpg", *src, out, par_)) continue;
      jpg_.set([&](std::vector<uint8_t>& v) { v = out; });
    }
  }

  static constexpr const char* kBoundary = "swimframe";
  int                          max_w_;
  Latest<Raw>                  raw_;
  Latest<std::vector<uint8_t>> jpg_;
  std::vector<int>             par_;
  std::atomic<int>             viewers_{0};
  std::thread                  th_;
};

void open_browser(const std::string& url) {
#ifdef _WIN32
  ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
  const int rc = std::system(("xdg-open '" + url + "' >/dev/null 2>&1 &").c_str());
  (void)rc;
#endif
}

}  // namespace

// ── 服务实现 ──────────────────────────────────────────────────────────────
struct WebServer::Impl {
  WebOptions  opt;
  std::string meta, url;
  socket_t    ls = INVALID_SOCKET;
  MjpegStream canvas, follow;
  Latest<std::string> stats;
  std::atomic<int>    stats_viewers{0};
  std::atomic<int>    sel{-1};
  std::atomic<bool>   stop{false};
  std::thread         acc;

  /// 每条连接一个线程。done 由线程自己置位，accept 循环顺手回收，
  /// 于是长跑（浏览器反复刷新）不会攒下无限多个 thread 对象。
  struct Conn {
    std::thread       th;
    std::atomic<bool> done{false};
  };
  std::mutex conn_mu;
  std::vector<std::unique_ptr<Conn>> conns;

  Impl(const WebOptions& o, std::string m)
      : opt(o), meta(std::move(m)), canvas(o.quality, o.width),
        follow(o.quality, o.follow_width) {}

  void accept_loop();
  void handle(socket_t s);
  void reap(bool all);
};

void WebServer::Impl::reap(bool all) {
  std::vector<std::unique_ptr<Conn>> dead;
  {
    std::lock_guard lk(conn_mu);
    for (auto& c : conns)
      if (all || c->done) dead.push_back(std::move(c));
    conns.erase(std::remove(conns.begin(), conns.end(), nullptr), conns.end());
  }
  for (auto& c : dead)
    if (c->th.joinable()) c->th.join();
}

void WebServer::Impl::accept_loop() {
  for (;;) {
    const socket_t c = ::accept(ls, nullptr, nullptr);
    if (c == INVALID_SOCKET) break;              // 析构关掉监听 socket 即走这里
    if (stop) { closesocket(c); break; }
    reap(false);
    auto conn = std::make_unique<Conn>();
    Conn* raw = conn.get();
    raw->th = std::thread([this, c, raw] {
      try {
        handle(c);
      } catch (...) {
      }
      ::shutdown(c, SD_BOTH);
      closesocket(c);
      raw->done = true;
    });
    std::lock_guard lk(conn_mu);
    conns.push_back(std::move(conn));
  }
}

void WebServer::Impl::handle(socket_t s) {
  set_timeout(s, SO_RCVTIMEO, 5000);
  // 发送超时兼作「客户端不读」的判据：MJPEG 一帧几十 KB，5 s 发不出去就是死连接
  set_timeout(s, SO_SNDTIMEO, 5000);

  // 只需要请求行，但必须把头读完（否则某些客户端的后续写入会撞上 RST）
  std::string req;
  char buf[2048];
  for (;;) {
    const int n = ::recv(s, buf, sizeof buf, 0);
    if (n <= 0) return;
    req.append(buf, size_t(n));
    if (req.find("\r\n\r\n") != std::string::npos) break;
    if (req.size() > (1u << 16)) return;          // 异常大的请求头直接丢
  }
  const size_t p1 = req.find(' ');
  const size_t p2 = req.find(' ', p1 == std::string::npos ? 0 : p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos) return;
  const std::string target = req.substr(p1 + 1, p2 - p1 - 1);
  const size_t q = target.find('?');
  const std::string path  = target.substr(0, q);
  const std::string query = q == std::string::npos ? "" : target.substr(q + 1);

  if (path == "/" || path == "/index.html") {
    send_body(s, "text/html; charset=utf-8", kIndexHtml);
  } else if (path == "/meta") {
    send_body(s, "application/json", meta);
  } else if (path == "/canvas.mjpg") {
    canvas.serve(s);
  } else if (path == "/follow.mjpg") {
    follow.serve(s);
  } else if (path == "/select") {
    const size_t k = query.find("id=");
    sel = k == std::string::npos ? -1 : atoi(query.c_str() + k + 3);
    printf("[Web] 选中 track %d\n", sel.load());
    fflush(stdout);
    send_str(s, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
  } else if (path == "/stats") {
    if (!send_str(s, "HTTP/1.1 200 OK\r\nCache-Control: no-store\r\n"
                     "Connection: close\r\nContent-Type: text/event-stream\r\n\r\n"))
      return;
    ++stats_viewers;
    uint64_t seq = stats_viewers > 1 ? 0 : stats.seq();   // 同 MjpegStream::serve
    std::string v;
    const auto take = [&v](const std::string& x) { v = x; };
    while (stats.wait(seq, take))
      if (!send_str(s, "data: " + v + "\n\n")) break;
    --stats_viewers;
  } else {
    send_str(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                "Connection: close\r\n\r\n");
  }
}

WebServer::WebServer(const WebOptions& opt, std::string meta_json)
    : im_(std::make_unique<Impl>(opt, std::move(meta_json))) {
#ifdef _WIN32
  static std::once_flag wsa_once;
  std::call_once(wsa_once, [] {
    WSADATA d;
    SWIM_CHECK(WSAStartup(MAKEWORD(2, 2), &d) == 0, "WSAStartup 失败");
  });
#endif
  im_->ls = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  SWIM_CHECK(im_->ls != INVALID_SOCKET,
             "创建 socket 失败 (" + std::to_string(last_err()) + ")");
  const int on = 1;
  ::setsockopt(im_->ls, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&on), sizeof on);
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port   = htons(static_cast<uint16_t>(opt.port));
  SWIM_CHECK(::inet_pton(AF_INET, opt.bind.c_str(), &sa.sin_addr) == 1,
             "--web-bind 不是合法 IPv4 地址: " + opt.bind);
  SWIM_CHECK(::bind(im_->ls, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0,
             "绑定 " + opt.bind + ":" + std::to_string(opt.port) + " 失败 (" +
                 std::to_string(last_err()) + ")，端口可能已被占用");
  SWIM_CHECK(::listen(im_->ls, 16) == 0,
             "listen 失败 (" + std::to_string(last_err()) + ")");

  im_->url = "http://" + opt.bind + ":" + std::to_string(opt.port) + "/";
  im_->acc = std::thread([this] { im_->accept_loop(); });
  printf("[Web] %s%s\n", im_->url.c_str(),
         opt.bind == "127.0.0.1" ? "" : "  （已开在全部网口，同网段可见）");
  fflush(stdout);
  if (opt.open_browser) open_browser(im_->url);
}

WebServer::~WebServer() {
  im_->stop = true;
  // 顺序要紧：先唤醒所有阻塞在 wait 上的连接线程（SSE 与两路 MJPEG），
  // 再关监听 socket 让 accept 返回，最后 join 连接线程。反过来会死等。
  im_->stats.stop();
  im_->canvas.stop();
  im_->follow.stop();
  if (im_->ls != INVALID_SOCKET) {
    closesocket(im_->ls);
    im_->ls = INVALID_SOCKET;
  }
  if (im_->acc.joinable()) im_->acc.join();
  im_->reap(true);
}

void WebServer::publish_canvas(const uint8_t* bgr, int w, int h) {
  im_->canvas.publish(bgr, w, h);
}
void WebServer::publish_follow(const uint8_t* bgr, int w, int h) {
  im_->follow.publish(bgr, w, h);
}
void WebServer::publish_stats(const std::string& json) {
  im_->stats.set([&](std::string& v) { v = json; });
}
bool WebServer::wants_canvas() const { return im_->canvas.wanted(); }
bool WebServer::wants_follow() const { return im_->follow.wanted(); }
bool WebServer::wants_stats()  const { return im_->stats_viewers > 0; }
int  WebServer::selected() const { return im_->sel; }
const std::string& WebServer::url() const { return im_->url; }

}  // namespace swim
