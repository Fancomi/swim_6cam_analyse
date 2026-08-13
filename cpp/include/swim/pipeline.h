// GPU 推理流水线：detect + crop + pose 全程显存驻留，一次 D2H 只取关键点。
//
// 三段线程并行（段间有界队列，深度小以形成自然背压）：
//   [解码线程]  FrameSource -> GpuFrame          (解码 + H2D，独立 stream)
//   [推理线程]  preprocess -> detect -> filter -> crop -> pose -> simcc_decode
//               全部在同一 CUDA stream 上排队
//   [后处理线程] 跟踪 -> 划水/速度 -> 回调（渲染或落盘）
//
// 跨帧重叠：每帧的回读缓冲取自一个环，推理线程排完 cudaMemcpyAsync 只记一个
// cudaEvent 就去处理下一帧，由后处理线程在真正读之前 eventSynchronize。
// 于是「GPU 算第 n+1 帧」与「CPU 跟踪第 n 帧」天然并行。唯一必须当帧等待的是
// 框数（pose 的 batch 取决于它，4 字节）。
//
// 为什么 detect 与 pose 不再拆两个线程：二者有数据依赖（pose 的 batch 取决于
// detect 的框数），拆开只会引入跨线程同步，收益为负。同 stream 内 GPU 自身
// 已经把 kernel 与推理排满。
#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "swim/frame_source.h"
#include "swim/kernels.h"
#include "swim/metrics.h"
#include "swim/trt_engine.h"

namespace swim {

/// 合法泳姿（与 cli.py 的 --stroke-type choices 一致）。一处定义，参数校验、
/// 错误信息与 --help 共用，避免三处各写一份而漂移。
constexpr const char* kStrokeTypes =
    "freestyle backstroke butterfly breaststroke unknown";

inline bool valid_stroke(const std::string& s) {
  const std::string all = std::string(" ") + kStrokeTypes + " ";
  return !s.empty() && all.find(" " + s + " ") != std::string::npos;
}

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
  /// 泳姿：只影响"左右是否分开计数"（对齐 metrics.py 的 split_sides）
  std::string  stroke_type = "freestyle";
  StrokeSignal signal      = StrokeSignal::ElbowAngle;
  int    queue_depth      = 3;
  bool   fp16             = true;
  int64_t max_frames      = 0;             // 0 = 不限
  /// 需要图像（渲染/预览）时置 true：每帧多一次整帧 D2H。
  /// 纯分析模式保持 false —— 全链路只回读关键点，约 9 KB/帧。
  bool   need_image       = false;

  /// 参数区间自检（构造前调用；越界参数会在 kernel 里表现为越界或死循环，
  /// 在这里一次性挡掉比在 GPU 上崩溃好定位）。
  void validate() const;
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
  /// 回读环的一格：锁页 host 缓冲 + 该格 D2H 的完成事件。
  /// 有了事件，推理线程排完拷贝即可继续下一帧，由消费者在真正用之前等待，
  /// 从而让「GPU 算下一帧」与「CPU 处理上一帧」重叠。
  template <typename T>
  struct Slot {
    T*          host  = nullptr;
    cudaEvent_t ready = nullptr;
  };

  /// 一帧的推理产物（指向回读环，尚未落地；消费者等 ready 后再读）。
  struct Raw {
    int64_t      index = -1;
    int          n     = 0;          // 本帧人数
    Slot<float>* blob  = nullptr;    // 框+关键点，n>0 时非空
    Slot<uint8_t>* frame = nullptr;  // 整帧 BGR，仅 need_image 时非空
    int          w = 0, h = 0;
  };

  void infer_loop(FrameSource& src);
  void post_loop(const Sink& sink);
  /// 单帧 GPU 链路，结果异步写入 blob，返回本帧人数。
  int  infer_frame(const GpuFrame& f, Slot<float>& blob);
  /// 渲染路径的整帧回读环：run() 开始时一次性分配，运行期不再分配。
  void alloc_frame_ring(int w, int h);

  PipelineOptions opt_;
  double          fps_;
  Timers          timers_;

  std::unique_ptr<TrtEngine> det_, pose_;
  cudaStream_t               stream_ = nullptr;

  // 预分配的 device 中间缓冲（运行期不再分配）。
  // boxes/conf/kpts/scores 连成一块 d_blob_，回读只需一次 D2H（约 9 KB）。
  float* d_blob_    = nullptr;
  float* d_boxes_   = nullptr;   // [max_persons, 4]  ┐
  float* d_conf_    = nullptr;   // [max_persons]     ├ 指向 d_blob_ 内部
  float* d_kpts_    = nullptr;   // [max_persons,K,2] │
  float* d_scores_  = nullptr;   // [max_persons,K]   ┘
  int*   d_count_   = nullptr;   // 有效框数
  float* d_centers_ = nullptr;   // [max_persons, 2] 采样窗中心
  float* d_scales_  = nullptr;   // [max_persons, 2] 采样窗尺度
  int*   h_count_   = nullptr;   // 唯一必须当帧同步的回读（决定 pose batch）
  size_t blob_n_    = 0;         // d_blob_/h_blob_ 的 float 个数

  // 深度 = 队列容量 + 消费者手上 1 + 生产者正在写 1，少一格会覆写在用的数据
  std::vector<Slot<float>>   blob_ring_;
  std::vector<Slot<uint8_t>> frame_ring_;
  size_t                     frame_bytes_ = 0;
  size_t                     cursor_      = 0;   // 两个环共用（仅推理线程访问）

  Channel<Raw>              chan_;
  Tracker                   tracker_;
  MetricsTracker            metrics_;
  cudaEvent_t               ev_count_ = nullptr;   // 框数 D2H 完成
  std::atomic<int64_t>      frames_done_{0};
  std::atomic<bool>         stop_{false};
  std::exception_ptr        err_;                  // 后处理线程的异常，join 后重抛
};

}  // namespace swim
