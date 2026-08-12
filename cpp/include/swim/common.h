// 公共类型、CUDA/TRT 错误检查、计时器。
//
// 全链路的核心约束：图像数据进 GPU 后不再回 CPU。
// 只有三处小量 D2H：有效框数(4B)、框(≤300*5*4B)、关键点(≤40*17*3*4B)。
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace swim {

// ── 常量（与 cpp/models/pose_meta.json 及训练配置一致）─────────────────────
constexpr int   kNumKpts     = 17;      // COCO17
constexpr int   kPoseW       = 192;     // RTMPose 输入宽
constexpr int   kPoseH       = 256;     // RTMPose 输入高
constexpr float kSimccRatio  = 2.0f;    // simcc_split_ratio
constexpr float kBoxPadding  = 1.25f;   // GetBBoxCenterScale 的 padding
constexpr int   kMaxDet      = 300;     // yolo26 end2end 的 max_det
constexpr int   kMaxPersons  = 40;      // pose 的最大 batch（显存按此预留）

// RTMPose data_preprocessor 的归一化参数（RGB 顺序）
constexpr std::array<float, 3> kMean{123.675f, 116.28f, 103.53f};
constexpr std::array<float, 3> kStd{58.395f, 57.12f, 57.375f};

// ── 错误检查 ──────────────────────────────────────────────────────────────
#define SWIM_CUDA(call)                                                       \
  do {                                                                        \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess)                                                     \
      throw std::runtime_error(std::string("CUDA ") + #call + " -> " +         \
                               cudaGetErrorString(e_));                        \
  } while (0)

#define SWIM_CHECK(cond, msg)                                                 \
  do {                                                                        \
    if (!(cond)) throw std::runtime_error(std::string("检查失败: ") + (msg));   \
  } while (0)

// ── 数据结构 ──────────────────────────────────────────────────────────────

/// 显存中的一帧画布（BGR uint8，packed HWC）。所有权由 FrameSource 持有。
struct GpuFrame {
  uint8_t* data   = nullptr;   // device 指针，尺寸 w*h*3
  int      w      = 0;
  int      h      = 0;
  int64_t  index  = -1;        // 帧号（live 流下为递增序号）
  bool     valid() const { return data && w > 0 && h > 0; }
};

/// 一个检测框 + 其关键点。坐标均为画布像素。
struct Person {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0;
  int   track_id = -1;
  int   strokes  = 0;                          // 累计划水次数（后处理填）
  float speed    = std::numeric_limits<float>::quiet_NaN();   // m/s
  std::array<float, kNumKpts * 2> kpts{};      // x,y 交替
  std::array<float, kNumKpts>     scores{};

  float cx() const { return (x1 + x2) * 0.5f; }
  float cy() const { return (y1 + y2) * 0.5f; }
  float w()  const { return x2 - x1; }
  float h()  const { return y2 - y1; }
};

/// 一帧的完整结果（已在 CPU）。bgr 仅在开启渲染时非空（指向内部锁页缓冲，
/// 在回调返回前有效）。
struct FrameResult {
  int64_t             index = -1;
  std::vector<Person> persons;
  const uint8_t*      bgr = nullptr;
  int                 w = 0, h = 0;
};

// ── 分段计时（线程安全，用于耗时拆解）────────────────────────────────────
class Timers {
 public:
  void add(const std::string& key, double ms) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& s = stat_[key];
    s.first += ms;
    s.second += 1;
  }
  /// 按累计耗时降序打印，附占比。
  void report(const std::string& title, double total_ms, int64_t frames) const;
  void clear() { std::lock_guard<std::mutex> lk(mu_); stat_.clear(); }

 private:
  mutable std::mutex mu_;
  std::map<std::string, std::pair<double, int64_t>> stat_;   // key -> (总ms, 次数)
};

/// RAII 计时：析构时把耗时写入 Timers。
class ScopedTimer {
 public:
  ScopedTimer(Timers& t, std::string key)
      : t_(t), key_(std::move(key)), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    using namespace std::chrono;
    t_.add(key_, duration<double, std::milli>(steady_clock::now() - t0_).count());
  }

 private:
  Timers&     t_;
  std::string key_;
  std::chrono::steady_clock::time_point t0_;
};

#define SWIM_TIME(timers, key) swim::ScopedTimer _st_##__LINE__((timers), (key))

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace swim
