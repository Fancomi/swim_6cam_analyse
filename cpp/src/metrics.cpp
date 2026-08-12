#include "swim/metrics.h"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace swim {
namespace {

// COCO17 索引（与 src/swim_analyse/pose.py 一致）
constexpr int kShoL = 5, kShoR = 6, kElbL = 7, kElbR = 8, kWriL = 9, kWriR = 10;
constexpr float kKptThr = 0.4f;

// 滑动窗口长度：够放下局部自适应阈值所需的 6 秒，外加余量
constexpr double kWindowSec   = 8.0;
constexpr double kMinStrokeSec = 0.67;   // 相邻划水最小间隔
constexpr double kSpeedWinSec  = 2.0;    // 速度窗口
constexpr float  kProminence   = 0.2f;   // 局部幅度的占比，作 prominence 下限

float iou(float ax1, float ay1, float ax2, float ay2,
          float bx1, float by1, float bx2, float by2) {
  const float ix = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float iy = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = ix * iy;
  const float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

/// 肩-肘-腕夹角（弧度）；任一点置信度不足返回 NaN。
float elbow_angle(const Person& p, int sho, int elb, int wri) {
  if (std::min({p.scores[sho], p.scores[elb], p.scores[wri]}) < kKptThr)
    return std::nanf("");
  const float ax = p.kpts[sho * 2] - p.kpts[elb * 2];
  const float ay = p.kpts[sho * 2 + 1] - p.kpts[elb * 2 + 1];
  const float bx = p.kpts[wri * 2] - p.kpts[elb * 2];
  const float by = p.kpts[wri * 2 + 1] - p.kpts[elb * 2 + 1];
  const float na = std::hypot(ax, ay), nb = std::hypot(bx, by);
  if (na < 1e-6f || nb < 1e-6f) return std::nanf("");
  return std::acos(std::clamp((ax * bx + ay * by) / (na * nb), -1.f, 1.f));
}

/// 长度 5 的中值滤波（就地，短窗足够抑制单帧跳变）。
void median5(std::vector<float>& v) {
  if (v.size() < 5) return;
  std::vector<float> out = v;
  for (size_t i = 2; i + 2 < v.size(); ++i) {
    float w[5] = {v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2]};
    std::nth_element(w, w + 2, w + 5);
    out[i] = w[2];
  }
  v.swap(out);
}

}  // namespace

void Tracker::update(std::vector<Person>& persons) {
  // 枚举所有 (track, det) 的 IoU，按从高到低贪心配对
  struct Cand { float iou; int tid; int di; };
  std::vector<Cand> cands;
  cands.reserve(tracks_.size() * persons.size());
  for (const auto& [tid, t] : tracks_)
    for (size_t i = 0; i < persons.size(); ++i) {
      const auto& p = persons[i];
      const float v = iou(t.x1, t.y1, t.x2, t.y2, p.x1, p.y1, p.x2, p.y2);
      if (v >= iou_thr_) cands.push_back({v, tid, int(i)});
    }
  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) { return a.iou > b.iou; });

  std::vector<int> assign(persons.size(), -1);
  std::vector<char> used_det(persons.size(), 0);
  std::unordered_map<int, char> matched;         // 本帧匹配上的既有 track
  for (const auto& c : cands) {
    if (matched.count(c.tid) || used_det[c.di]) continue;
    assign[c.di] = c.tid;
    matched[c.tid] = 1;
    used_det[c.di] = 1;
  }

  // 先处理未匹配的既有 track：累加丢失计数，超阈值才删除。
  // 必须在给 persons 赋新 id 之前做 —— 否则新建的 id 已写进 tracks_，
  // 就分不清"本帧匹配上的"与"本帧新建的"，lost 永远累加不起来。
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (matched.count(it->first)) { it->second.lost = 0; ++it; continue; }
    it = (++it->second.lost > max_lost_) ? tracks_.erase(it) : std::next(it);
  }

  // 再落地本帧结果：匹配上的沿用 id 并更新框，未匹配的检测框开新 id
  for (size_t i = 0; i < persons.size(); ++i) {
    auto& p = persons[i];
    p.track_id = assign[i] >= 0 ? assign[i] : next_id_++;
    tracks_[p.track_id] = {p.x1, p.y1, p.x2, p.y2, 0};
  }
}

void MetricsTracker::update(int tid, int64_t frame, const Person& p) {
  auto& s = st_[tid];
  s.last_frame = frame;

  const float al = elbow_angle(p, kShoL, kElbL, kWriL);
  const float ar = elbow_angle(p, kShoR, kElbR, kWriR);
  if (!std::isnan(al)) s.angle_l.emplace_back(frame, al);
  if (!std::isnan(ar)) s.angle_r.emplace_back(frame, ar);
  s.center.emplace_back(frame, p.cx());
  s.center_y.emplace_back(frame, p.cy());

  // 丢弃窗口外的历史
  const int64_t keep = int64_t(kWindowSec * fps_);
  auto trim = [&](std::deque<std::pair<int64_t, float>>& d) {
    while (!d.empty() && frame - d.front().first > keep) d.pop_front();
  };
  trim(s.angle_l); trim(s.angle_r); trim(s.center); trim(s.center_y);

  detect_valley(s.angle_l, frame, s.strokes_l, s.last_peak_l);
  detect_valley(s.angle_r, frame, s.strokes_r, s.last_peak_r);
  update_speed(s, frame);
}

void MetricsTracker::detect_valley(std::deque<std::pair<int64_t, float>>& buf,
                                   int64_t frame, int& count,
                                   int64_t& last_peak) const {
  // 相邻划水的最小间隔（与 Python 版的 find_peaks distance 同义）
  const int64_t min_gap = int64_t(kMinStrokeSec * fps_);
  if (frame - last_peak < min_gap) return;
  // 至少要够 median5 + 左右各 1 个邻域
  if (buf.size() < 7) return;

  std::vector<float> v;
  v.reserve(buf.size());
  for (const auto& kv : buf) v.push_back(kv.second);
  median5(v);

  // 判定"倒数第 3 个点"是否为局部极小：median5 只平滑 i∈[2, n-3]，
  // 末尾两点未被平滑，因此判定点取 n-3（其自身与右邻 n-2 都在有效区内）。
  const size_t c = v.size() - 3;
  if (c < 1) return;
  const float cur = v[c];
  // 只与左右各 1 个邻居比较。原先比左2右2共4个邻居过严：
  // 水花导致的单帧抖动经 median5 已抑制，多比邻居只会漏掉真实波谷。
  if (!(cur <= v[c - 1] && cur <= v[c + 1])) return;

  // 局部自适应阈值：用窗口内的均值与幅度，避免全局阈值淹没浅震荡段
  const auto [mn, mx] = std::minmax_element(v.begin(), v.end());
  const float span = *mx - *mn;
  if (span < 1e-6f) return;
  const float mean = std::accumulate(v.begin(), v.end(), 0.f) / v.size();
  if (cur > mean) return;                              // 必须低于均值
  if (*mx - cur < span * kProminence) return;          // 突出度不足

  ++count;
  last_peak = buf[c].first;
}

void MetricsTracker::update_speed(State& s, int64_t frame) const {
  const int64_t win = int64_t(kSpeedWinSec * fps_);
  // 找窗口内最早的样本，与当前样本算位移
  size_t i = 0;
  while (i + 1 < s.center.size() && frame - s.center[i].first > win) ++i;
  if (s.center.size() - i < 2) return;
  const double dt = double(s.center.back().first - s.center[i].first) / fps_;
  if (dt < 1e-6) return;
  const float dx = s.center.back().second - s.center[i].second;
  const float dy = s.center_y.back().second - s.center_y[i].second;
  s.speed = float(std::hypot(dx, dy) / ppm_ / dt);
}

int MetricsTracker::strokes(int tid) const {
  auto it = st_.find(tid);
  // 自由泳左右手交替，取较小值（与 Python 版一致，抗单侧漏检）
  return it == st_.end() ? 0 : std::min(it->second.strokes_l, it->second.strokes_r);
}

float MetricsTracker::speed(int tid) const {
  auto it = st_.find(tid);
  return it == st_.end() ? std::nanf("") : it->second.speed;
}

void MetricsTracker::prune(int64_t cur, int64_t max_idle) {
  for (auto it = st_.begin(); it != st_.end();)
    it = (cur - it->second.last_frame > max_idle) ? st_.erase(it) : std::next(it);
}

}  // namespace swim
