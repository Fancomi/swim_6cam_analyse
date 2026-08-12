// GPU 推理流水线：detect + crop + pose 全程显存驻留，一次 D2H 只取关键点。
//
// 三段线程并行（段间有界队列，深度小以形成自然背压）：
//   [解码线程]  FrameSource -> GpuFrame          (H2D 或 NVDEC，独立 stream)
//   [推理线程]  preprocess -> detect -> filter -> crop -> pose -> simcc_decode
//               全部在同一 CUDA stream 上排队，只在取 count/关键点时同步
//   [后处理线程] 跟踪 -> 划水/速度 -> 回调（渲染或落盘）
//
// 为什么 detect 与 pose 不再拆两个线程：二者有数据依赖（pose 的 batch 取决于
// detect 的框数），拆开只会引入跨线程同步，收益为负。同 stream 内 GPU 自身
// 已经把 kernel 与推理排满。
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "swim/frame_source.h"
#include "swim/kernels.h"
#include "swim/metrics.h"
#include "swim/trt_engine.h"

namespace swim {

/// 有界阻塞队列：满则阻塞生产者（背压），close() 后消费者取空即退出。
template <typename T>
class Channel {
 public:
  explicit Channel(size_t cap) : cap_(cap) {}

  bool push(T v) {
    std::unique_lock lk(mu_);
    cv_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push(std::move(v));
    cv_empty_.notify_one();
    return true;
  }
  bool pop(T& out) {
    std::unique_lock lk(mu_);
    cv_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;               // 已 close 且取空
    out = std::move(q_.front());
    q_.pop();
    cv_full_.notify_one();
    return true;
  }
  void close() {
    std::lock_guard lk(mu_);
    closed_ = true;
    cv_empty_.notify_all();
    cv_full_.notify_all();
  }
  size_t size() const { std::lock_guard lk(mu_); return q_.size(); }

 private:
  mutable std::mutex      mu_;
  std::condition_variable cv_empty_, cv_full_;
  std::queue<T>           q_;
  size_t                  cap_;
  bool                    closed_ = false;
};

struct PipelineOptions {
  std::string models_dir  = "cpp/models";  // 放 detect.onnx / pose.onnx
  std::string engine_dir  = "cpp/models";  // engine 缓存目录
  int    detect_size      = 640;           // 必须与 detect.onnx 的输入尺寸一致
                                           // (yolo_swim_detect 训练 imgsz=640)
  float  conf_thr         = 0.25f;
  float  kpt_thr          = 0.4f;
  int    max_persons      = kMaxPersons;
  float  containment      = 0.7f;         // 包含率去重阈值（交集/自身面积）
  float  track_iou        = 0.3f;
  int    track_max_lost   = 30;
  float  ppm              = 100.f;         // 画布每米像素数
  int    queue_depth      = 3;
  bool   fp16             = true;
  int64_t max_frames      = 0;             // 0 = 不限
  /// 需要图像（渲染/预览）时置 true：每帧多一次整帧 D2H。
  /// 纯分析模式保持 false —— 全链路只回读关键点，约 5 KB/帧。
  bool   need_image       = false;
};

class Pipeline {
 public:
  /// 每帧结果的回调（在后处理线程调用，需自身线程安全）。
  using Sink = std::function<void(const FrameResult&)>;

  Pipeline(const PipelineOptions& opt, double fps);
  ~Pipeline();

  /// 阻塞运行到流结束或达到 max_frames。
  void run(FrameSource& src, const Sink& sink);

  const Timers& timers() const { return timers_; }
  int64_t frames_done() const { return frames_done_; }

 private:
  /// 一帧的推理产物（关键点已在 CPU）。
  struct Raw {
    int64_t             index = -1;
    std::vector<Person> persons;
    /// 仅 need_image 时有效：指向锁页帧缓冲中的一格。
    uint8_t*            bgr = nullptr;
    int                 w = 0, h = 0;
  };

  void infer_loop(FrameSource& src);
  void post_loop(const Sink& sink);
  /// 单帧 GPU 链路，返回本帧人数。
  int  infer_frame(const GpuFrame& f);

  PipelineOptions opt_;
  double          fps_;
  Timers          timers_;

  std::unique_ptr<TrtEngine> det_, pose_;
  cudaStream_t               stream_ = nullptr;

  // 预分配的 device 中间缓冲（运行期不再分配）
  float* d_boxes_   = nullptr;   // [max_persons, 4]
  float* d_conf_    = nullptr;   // [max_persons]
  int*   d_count_   = nullptr;   // 有效框数
  float* d_centers_ = nullptr;   // [max_persons, 2] 采样窗中心
  float* d_scales_  = nullptr;   // [max_persons, 2] 采样窗尺度
  float* d_kpts_    = nullptr;   // [max_persons, K, 2]
  float* d_scores_  = nullptr;   // [max_persons, K]
  // 锁页 host 侧回读缓冲
  int*   h_count_   = nullptr;
  float* h_boxes_   = nullptr;
  float* h_conf_    = nullptr;
  float* h_kpts_    = nullptr;
  float* h_scores_  = nullptr;
  // need_image 时的整帧回读环（深度与队列一致，避免消费者还在用就被覆写）
  std::vector<uint8_t*> h_frames_;
  size_t                frame_bytes_ = 0;
  size_t                frame_cursor_ = 0;

  Channel<Raw>              chan_;
  Tracker                   tracker_;
  MetricsTracker            metrics_;
  std::atomic<int64_t>       frames_done_{0};
  std::atomic<bool>          stop_{false};
};

}  // namespace swim
