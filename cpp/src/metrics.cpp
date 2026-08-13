#include "swim/metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace swim {
namespace {

// COCO17 索引（与 src/swim_analyse/pose.py 一致）
constexpr int kNose = 0, kShoL = 5, kShoR = 6, kElbL = 7, kElbR = 8,
              kWriL = 9, kWriR = 10;

// 以下常量与 src/swim_analyse/metrics.py 的同名量一一对应
constexpr double kMinStrokeSec = 0.67;   // MIN_STROKE_INTERVAL
constexpr double kZcMinSec     = 0.30;   // ZC_MIN_INTERVAL
constexpr double kAdaptiveSec  = 6.00;   // ADAPTIVE_WINDOW
constexpr double kSpeedWinSec  = 2.00;   // SPEED_WINDOW_SEC
constexpr double kSpeedDtSec   = 0.20;   // SPEED_SAMPLE_DT
constexpr float  kPromRatio    = 0.20f;  // ADAPTIVE_PROMINENCE_RATIO
constexpr float  kHystRatio    = 0.15f;  // HYSTERESIS_RATIO
constexpr float  kEps          = 1e-6f;

constexpr int kMed = 5;    // MEDIAN_WINDOW
constexpr int kSg  = 11;   // SG_WINDOW（SG_POLYORDER = 3）
// SG(11,3) 平滑核。对称窗 2m+1 的三阶（与二阶同解）最小二乘平滑系数有闭式：
//   c_i = 3(3m^2 + 3m - 1 - 5i^2) / ((2m+3)(2m+1)(2m-1))
// m=5 时分母 13*11*9 = 1287 = 3*429，整数分子/429 即下表（与 Savitzky & Golay
// 1964 Table I 的 11 点二次/三次平滑行一致）。定值常量，无需每帧解最小二乘。
constexpr float kSgK[kSg] = {-36 / 429.f, 9 / 429.f, 44 / 429.f, 69 / 429.f,
                             84 / 429.f, 89 / 429.f, 84 / 429.f, 69 / 429.f,
                             44 / 429.f, 9 / 429.f, -36 / 429.f};
// 判定点相对最新样本的滞后 = 两级滤波各自的右侧邻域之和。取这个滞后后，判定点
// 的平滑值只依赖已到达的样本，与离线全序列的平滑结果逐位相同（无边界补齐误差）
constexpr int kLag = kMed / 2 + kSg / 2;   // 7

float iou(float ax1, float ay1, float ax2, float ay2,
          float bx1, float by1, float bx2, float by2) {
  const float ix = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float iy = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = ix * iy;
  const float ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

/// 肩-肘-腕夹角（弧度）；任一点置信度不足返回 NaN。
float elbow_angle(const Person& p, int sho, int elb, int wri, float thr) {
  if (std::min({p.scores[sho], p.scores[elb], p.scores[wri]}) < thr)
    return std::nanf("");
  const float ax = p.kpts[sho * 2] - p.kpts[elb * 2];
  const float ay = p.kpts[sho * 2 + 1] - p.kpts[elb * 2 + 1];
  const float bx = p.kpts[wri * 2] - p.kpts[elb * 2];
  const float by = p.kpts[wri * 2 + 1] - p.kpts[elb * 2 + 1];
  const float na = std::hypot(ax, ay), nb = std::hypot(bx, by);
  if (na < kEps || nb < kEps) return std::nanf("");
  return std::acos(std::clamp((ax * bx + ay * by) / (na * nb), -1.f, 1.f));
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
  // stable_sort：IoU 相等时保持枚举序，配对结果与运行次数无关（决定论输出）
  std::stable_sort(cands.begin(), cands.end(),
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

MetricsTracker::MetricsTracker(double fps, float ppm, float kpt_thr, bool split,
                               StrokeSignal signal)
    : fps_(fps), ppm_(ppm), kpt_thr_(kpt_thr), split_(split), signal_(signal) {
  gap_stroke_ = std::max<int64_t>(1, int64_t(fps * kMinStrokeSec));
  gap_zc_     = std::max<int64_t>(1, int64_t(fps * kZcMinSec));
  // 波谷的拓扑 prominence 要向右走到回升点才成立，故判定点额外右留一个划水间隔
  // 的前视；过零判定本身是因果的，不需要前视。
  look_ = signal == StrokeSignal::ElbowAngle ? gap_stroke_ : 0;
  // 局部自适应统计窗与 Python 同宽：max(distance*2, fps*ADAPTIVE_WINDOW) 收敛到
  // 奇数；同时保证放得下"平滑邻域 + 前视 + 左右各一个比较点"
  const int64_t w = std::max(gap_stroke_ * 2, int64_t(fps * kAdaptiveSec));
  cap_ = std::max(2 * int64_t(kLag) + look_ + 3, w % 2 ? w : w - 1);
  keep_speed_ = kSpeedWinSec * fps;
  speed_step_ = std::max<int64_t>(1, int64_t(std::lround(kSpeedDtSec * fps)));
  med_.reserve(size_t(cap_));
  sm_.reserve(size_t(cap_));
}

void MetricsTracker::update(int tid, int64_t frame, const Person& p) {
  auto& s = st_[tid];
  if (s.first_frame < 0) s.first_frame = frame;
  s.last_frame = frame;

  float l = 0.f, r = 0.f;
  raw_signals(p, s, l, r);
  if (split_) {
    push(s.ch[0], l);
    push(s.ch[1], r);
  } else {
    // 左右均值；只一侧可信取该侧，两侧都缺失保持 NaN（交给 push 插值补齐）
    const bool bl = !std::isnan(l), br = !std::isnan(r);
    push(s.ch[0], bl && br ? (l + r) * 0.5f : (bl ? l : r));
  }
  const int n_ch = split_ ? 2 : 1;
  const float dir = signal_ == StrokeSignal::WristXHead ? s.dir : 1.f;
  for (int i = 0; i < n_ch; ++i) detect(s.ch[i], dir);

  s.center.push_back({frame, p.cx(), p.cy()});
  while (!s.center.empty() &&
         double(frame - s.center.front().f) > keep_speed_ + double(speed_step_))
    s.center.pop_front();
  update_speed(s, frame);
}

void MetricsTracker::raw_signals(const Person& p, State& s,
                                 float& l, float& r) const {
  if (signal_ == StrokeSignal::ElbowAngle) {
    l = elbow_angle(p, kShoL, kElbL, kWriL, kpt_thr_);
    r = elbow_angle(p, kShoR, kElbR, kWriR, kpt_thr_);
    return;
  }
  // wrist_x_head：手腕沿 x 相对鼻子的位移，方向符号在 detect 里统一施加。
  // Python 用全序列首末鼻子 x 定方向，在线只能用"首个可信鼻子 x -> 当前"，
  // 因此开局几帧方向可能翻转，随位移累积后稳定。
  l = r = std::nanf("");
  if (p.scores[kNose] < kpt_thr_) return;
  const float nx = p.kpts[kNose * 2];
  if (std::isnan(s.nose_x0)) s.nose_x0 = nx;
  s.dir = nx < s.nose_x0 ? -1.f : 1.f;
  if (p.scores[kWriL] >= kpt_thr_) l = p.kpts[kWriL * 2] - nx;
  if (p.scores[kWriR] >= kpt_thr_) r = p.kpts[kWriR * 2] - nx;
}

void MetricsTracker::push(Chan& c, float x) const {
  if (std::isnan(x)) { ++c.hole; return; }   // 空洞先记账，等下一个有效样本插值
  if (c.hole >= cap_) {                      // 断档超过一整窗：旧样本已无参考价值
    c.pos += c.hole;
    c.hole = 0;
    c.v.clear();
  } else if (c.hole) {
    // 线性插值补齐（等价 _fill_nan 的 np.interp）；序列开头的空洞用首值填平
    const float a = c.v.empty() ? x : c.v.back();
    const float k = (x - a) / float(c.hole + 1);
    for (int64_t i = 1; i <= c.hole; ++i) c.v.push_back(a + k * float(i));
    c.pos += c.hole;
    c.hole = 0;
  }
  c.v.push_back(x);
  ++c.pos;
  while (int64_t(c.v.size()) > cap_) c.v.pop_front();
}

void MetricsTracker::detect(Chan& c, float dir) {
  const int n = int(c.v.size());
  const int m = n - 2 * kLag;             // 平滑值支撑完整的样本数
  if (m < 3 + int(look_)) return;         // 开局样本不足，无可判定位置
  // 可判定的最新位置：末端要留出右邻点与 prominence 前视
  const int64_t p = c.pos - kLag - 1 - look_;
  if (p <= c.last_eval) return;           // 本帧信号缺失，或该位置已判过

  // 中值(5) -> SG(11,3)：只算支撑完整的下标，故不需要任何边界补齐，判定点的
  // 平滑值与离线全序列（scipy）逐位一致。med_ 按原始下标寻址，有效区 [2, n-3]。
  med_.resize(size_t(n));
  for (int i = kMed / 2; i + kMed / 2 < n; ++i) {
    float w[kMed] = {c.v[size_t(i - 2)], c.v[size_t(i - 1)], c.v[size_t(i)],
                     c.v[size_t(i + 1)], c.v[size_t(i + 2)]};
    std::nth_element(w, w + kMed / 2, w + kMed);
    med_[size_t(i)] = w[kMed / 2];
  }
  // sm_[t] 对应原始下标 kLag + t；顺便统计窗口均值与峰峰值（_local_stats）。
  // dir 在此处施加：中值与 SG 都与取负可交换，先滤后乘等价。
  sm_.resize(size_t(m));
  float mean = 0.f, mn = std::numeric_limits<float>::max(), mx = -mn;
  for (int t = 0; t < m; ++t) {
    float acc = 0.f;
    for (int k = 0; k < kSg; ++k) acc += kSgK[k] * med_[size_t(t + kMed / 2 + k)];
    acc *= dir;
    sm_[size_t(t)] = acc;
    mean += acc;
    mn = std::min(mn, acc);
    mx = std::max(mx, acc);
  }
  mean /= float(m);
  const float span = mx - mn;
  if (span < kEps) { c.last_eval = p; return; }   // 信号无波动（同 Python 的 ptp 门限）
  // 波谷的 height/prominence 用窗内统计（自适应，同 Python 的 _local_stats）；
  // 过零的滞回门限用累计全程幅度（Python 是全序列 np.ptp，非局部量）
  c.gmin = std::min(c.gmin, mn);
  c.gmax = std::max(c.gmax, mx);

  // 逐个判定新暴露的位置：正常每帧 1 个，缺帧补齐后可能一次暴露多个
  for (int64_t q = std::max(c.last_eval + 1, c.begin() + kLag + 1); q <= p; ++q) {
    const int t = int(q - c.begin()) - kLag;      // 由上面的边界保证落在 [1, m-2]
    if (signal_ == StrokeSignal::ElbowAngle)
      count_valley(c, q, t, mean, span);
    else
      count_cross(c, q, t, (c.gmax - c.gmin) * kHystRatio);
  }
  c.last_eval = p;
}

void MetricsTracker::count_valley(Chan& c, int64_t p, int t,
                                  float mean, float span) const {
  if (p - c.last_ev < gap_stroke_) return;   // distance = fps * 0.67
  const int m = int(sm_.size());
  const float cur = sm_[size_t(t)];
  // 取负后即为波峰，故此处判局部极小。scipy 的平台峰取平台中点，此处只认
  // 左严格下降、右不上升的首点 —— SG 输出为 float，严格相等的平台实际不会出现
  if (!(cur < sm_[size_t(t - 1)] && cur <= sm_[size_t(t + 1)])) return;
  if (cur > mean) return;                    // height：取负后须 >= 局部均值
  // 拓扑 prominence（scipy peak_prominences 的定义，搜索范围限于本窗口）：
  // 向两侧走到第一个比该谷更低的样本或窗口边界，各取区间内最大值，
  // prominence = min(左最大, 右最大) - 谷值
  float lmax = cur, rmax = cur;
  for (int i = t - 1; i >= 0 && sm_[size_t(i)] >= cur; --i)
    lmax = std::max(lmax, sm_[size_t(i)]);
  for (int i = t + 1; i < m && sm_[size_t(i)] >= cur; ++i)
    rmax = std::max(rmax, sm_[size_t(i)]);
  if (std::min(lmax, rmax) - cur < std::max(span * kPromRatio, kEps)) return;

  ++c.count;
  c.last_ev = p;
}

void MetricsTracker::count_cross(Chan& c, int64_t p, int t, float delta) const {
  const float v = sm_[size_t(t)];
  // 首个判定位置只初始化滞回状态：Python 用 below = smoothed[0] < 0，而它的
  // 第 0 次迭代必然不触发（v < 0 与 v > delta 互斥），此处语义等价
  if (!c.below_init) { c.below_init = true; c.below = v < 0.f; return; }
  if (c.below) {
    if (v > delta) {                         // 由负转正：记一次划水
      if (p - c.last_ev >= gap_zc_) { ++c.count; c.last_ev = p; }
      c.below = false;
    }
  } else if (v < -delta) {
    c.below = true;
  }
}

void MetricsTracker::update_speed(State& s, int64_t frame) const {
  // 每 SPEED_SAMPLE_DT 秒一个采样格；格内不重算，即 Python 的逐帧前向填充
  const int64_t slot = (frame - s.first_frame) / speed_step_;
  if (slot <= s.speed_slot) return;
  s.speed_slot = slot;

  const int64_t g = s.first_frame + slot * speed_step_;   // 采样点对应帧号
  const Pt* a = nullptr;                                  // 窗口内最早的观测
  const Pt* b = nullptr;                                  // 窗口内最晚的观测
  for (const auto& q : s.center) {
    if (q.f > g) break;
    if (double(g - q.f) > keep_speed_) continue;          // 早于 2 s 窗口
    if (!a) a = &q;
    b = &q;
  }
  if (!a || a == b) return;                               // 窗口内不足两点：保持旧值
  const double dt = double(b->f - a->f) / fps_;
  if (dt < 1e-6) return;
  s.speed = float(std::hypot(b->x - a->x, b->y - a->y) / ppm_ / dt);
}

int MetricsTracker::strokes(int tid) const {
  auto it = st_.find(tid);
  if (it == st_.end()) return 0;
  const auto& s = it->second;
  // split：自由泳/仰泳左右手交替，取较小值抗单侧漏检；否则只有一路左右均值信号
  return split_ ? std::min(s.ch[0].count, s.ch[1].count) : s.ch[0].count;
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
