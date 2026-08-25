// 内嵌 HTTP 服务：把画布与分析结果推给浏览器（零第三方依赖，只用系统 socket）。
//
// 两路 MJPEG（multipart/x-mixed-replace）+ 一条 SSE：
//   GET /               单页前端（编译期嵌入，源文件是 cpp/web/index.html）
//   GET /meta           画布尺寸、ppm、骨架边表、网格口径（一次性）
//   GET /canvas.mjpg    全景画布流
//   GET /follow.mjpg    选中运动员的跟随视角
//   GET /stats          SSE，每帧一条 JSON（全场统计 + 每人明细）
//   GET /select?id=N    选中 track（-1 取消），204 无内容
//
// 为什么不用 WebSocket：叠加层与统计都是「只要最新一帧」的语义，SSE 天然满足且
// 浏览器端零解析代码；画布走 MJPEG 则由 <img> 原生解码，省掉一整套帧协议、
// SHA-1 握手与 base64。代价是画面与叠加层可能差一帧（33 ms），肉眼不可分。
//
// 线程模型：accept 一个线程 + 每连接一个线程 + 每路 MJPEG 一个编码线程（无人
// 观看时休眠，不烧 CPU）。publish_* 只做一次 memcpy 便返回，**绝不阻塞调用方**
// （它跑在后处理线程上），慢客户端只会丢帧，不会把背压顶回流水线。
#pragma once

#include <memory>
#include <string>

namespace swim {

struct WebOptions {
  /// 默认只绑回环：这个服务无认证无 TLS，绑 0.0.0.0 等于把实时泳池画面对同网段
  /// 全部开放。要给别的机器看时显式传 --web-bind 0.0.0.0。
  std::string bind = "127.0.0.1";
  int  port         = 8080;
  /// 两路流的横向像素上限（超过则等比缩小；纵向由源比例定）。缩放在编码线程做，
  /// 不占后处理线程 —— 5002 宽画布缩到 1280 要 4 ms，放在热路径上会吃掉帧预算。
  int  width        = 1280;   // 全景画布
  int  follow_width = 1280;   // 个人跟随（窗宽约 1225 px，给足不缩放的余量）
  int  quality      = 75;     // JPEG 质量 1..100
  bool open_browser = true;
};

/// HTTP 服务。构造即开始监听（失败抛异常），析构关闭并 join 所有线程。
class WebServer {
 public:
  /// meta_json 是 /meta 的完整响应体（几何与绘制口径，运行期不变）。
  WebServer(const WebOptions& opt, std::string meta_json);
  ~WebServer();
  WebServer(const WebServer&) = delete;
  WebServer& operator=(const WebServer&) = delete;

  /// 发布一帧 BGR（packed HWC）。内部拷走，返回后调用方可复用缓冲。
  void publish_canvas(const uint8_t* bgr, int w, int h);
  void publish_follow(const uint8_t* bgr, int w, int h);
  /// 发布一帧统计 JSON（SSE 的 data 体，不含换行）。
  void publish_stats(const std::string& json);

  /// 当前有客户端在看吗。没人看时调用方可跳过缩放/裁切/序列化，省掉每帧 1 ms。
  bool wants_canvas() const;
  bool wants_follow() const;
  bool wants_stats() const;
  /// 前端选中的 track；-1 = 未选。
  int  selected() const;

  const std::string& url() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> im_;
};

}  // namespace swim
