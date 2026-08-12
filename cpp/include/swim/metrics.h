// 跟踪与运动指标：IoU 贪心跟踪、划水计数、瞬时速度。
//
// 与 Python 版（src/swim_analyse/tracking.py、metrics.py）算法一致：
//   跟踪   包含率去重已在 detect 端完成（yolo26 end2end），此处只做逐帧 IoU 贪心匹配
//   划水   肩-肘-腕夹角 -> 中值+SG 平滑 -> 局部自适应阈值的波谷计数
//   速度   框中心位移 / ppm，每 0.2 s 一个采样点，窗口 2 s
//
// 在线约束：live 场景不能等全序列结束。因此划水与速度都用滑动窗口增量计算，
// 只保留最近 kHistorySec 秒的历史。
#pragma once

#include <cmath>
#include <deque>
#include <unordered_map>
#include <vector>

#include "swim/common.h"

namespace swim {

/// 逐帧 IoU 贪心匹配。丢失超过 max_lost 帧才删除 track。
class Tracker {
 public:
  Tracker(float iou_thr = 0.3f, int max_lost = 30)
      : iou_thr_(iou_thr), max_lost_(max_lost) {}

  /// 就地给 persons 填 track_id（顺序不变）。
  void update(std::vector<Person>& persons);
  size_t active() const { return tracks_.size(); }

 private:
  struct Track { float x1, y1, x2, y2; int lost; };
  std::unordered_map<int, Track> tracks_;
  float iou_thr_;
  int   max_lost_;
  int   next_id_ = 0;
};

/// 单个 track 的运动指标（滑动窗口，适合 live）。
class MetricsTracker {
 public:
  /// fps 用于把帧数换算成秒；ppm 为画布每米像素数。
  MetricsTracker(double fps, float ppm) : fps_(fps), ppm_(ppm) {}

  /// 喂入一帧某个 track 的观测。返回该 track 当前的累计划水次数。
  void update(int track_id, int64_t frame, const Person& p);

  int   strokes(int track_id) const;
  /// 最近一次速度（m/s）；无数据返回 NaN。
  float speed(int track_id) const;
  /// 清理长时间未更新的 track，避免 live 场景内存增长。
  void prune(int64_t current_frame, int64_t max_idle);

 private:
  struct State {
    std::deque<std::pair<int64_t, float>> angle_l, angle_r;   // (帧号, 肘角度)
    std::deque<std::pair<int64_t, float>> center;             // (帧号, cx) 用于速度
    std::deque<std::pair<int64_t, float>> center_y;
    int      strokes_l = 0, strokes_r = 0;
    int64_t  last_peak_l = -1000, last_peak_r = -1000;
    int64_t  last_frame = -1;
    float    speed = std::nanf("");
  };
  /// 在窗口内做平滑 + 波谷检测，命中则计数。
  void detect_valley(std::deque<std::pair<int64_t, float>>& buf, int64_t frame,
                     int& count, int64_t& last_peak) const;
  void update_speed(State& s, int64_t frame) const;

  std::unordered_map<int, State> st_;
  double fps_;
  float  ppm_;
};

}  // namespace swim
