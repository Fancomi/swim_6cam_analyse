#include "swim/stats.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>

namespace swim {
namespace {

/// printf 风格追加到 string。JSON 全是数字与短键名，用一次性缓冲比 ostream 短且快。
void put(std::string& s, const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  const int n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (n > 0) s.append(buf, size_t(n) < sizeof buf ? size_t(n) : sizeof buf - 1);
}

/// JSON 没有 NaN/Inf 字面量：非有限值一律 null，否则前端 JSON.parse 直接抛。
void put_num(std::string& s, double v, int prec = 2) {
  if (std::isfinite(v)) put(s, "%.*f", prec, v);
  else                  s += "null";
}

/// 里程采样间隔（秒）。与速度的 0.2 s 采样格同量级：更密只会把关键点抖动
/// 累加成虚假里程（静止的人也会「游」出几十米）。
constexpr double kDistDtSec = 0.2;
/// 吞吐显示的采样窗（秒）。太短读数乱跳，太长看不出卡顿。
constexpr double kFpsWinSec = 1.0;
/// 跟随窗：纵向 = 一条泳道 + 上下各留这么多米（看得见手臂出水与身后水花）。
/// 锁泳道而不是按框的倍数，是因为泳姿会让 bbox 忽大忽小、按框定高会不停变焦。
constexpr double kFollowPadM  = 0.5;
/// 跟随窗宽高比。看板里那一格是很扁的（约 3.5:1），按 16:9 裁会剩大片黑边。
constexpr double kFollowAspect = 3.5;
/// 横向延迟跟随的时间常数（秒）。镜头以一阶低通追人：bbox 抖动被滤掉，
/// 快速游动时落后约 tau*v（2 m/s 时约 1.2 m，仍在窗内三成偏移以内）。
constexpr double kFollowTauSec = 0.6;
/// 参与「近期最快」排名的最短观察时长（秒）。刚入场的人里程还没积起来，
/// 不设门限的话一两次关键点抖动就能把他排到第一。
constexpr double kRecentMinSec = 3.0;

}  // namespace

StatsBoard::StatsBoard(int canvas_w, int canvas_h, float ppm, double fps,
                       float kpt_thr)
    : cw_(canvas_w), ch_(canvas_h), ppm_(ppm), kpt_thr_(kpt_thr), fps_(fps) {}

int StatsBoard::lane_of(float cy) const {
  const double margin = kGridMarginM * ppm_;
  const double pitch  = kLanePitchM * ppm_;
  const int k = int(std::floor((double(cy) - margin) / pitch)) + 1;
  return k >= 1 && k <= kNumLanes ? k : 0;   // 0 = 池岸/池外
}

double StatsBoard::secs(const Track& t) const {
  return t.last > t.first ? double(t.last - t.first) / fps_ : 0.0;
}

void StatsBoard::update(const FrameResult& fr, double elapsed_sec, int sel) {
  index_   = fr.index;
  elapsed_ = elapsed_sec;
  cur_     = fr.persons;

  // 实时吞吐：按固定时间窗测，与 --show-fps 的口径一致（累计平均会掩盖卡顿）
  if (fps_t0_ == 0.0) { fps_t0_ = elapsed_sec; fps_i0_ = fr.index; }
  if (elapsed_sec - fps_t0_ >= kFpsWinSec) {
    cur_fps_ = double(fr.index - fps_i0_) / (elapsed_sec - fps_t0_);
    fps_t0_  = elapsed_sec;
    fps_i0_  = fr.index;
  }

  const int64_t dist_step = int64_t(kDistDtSec * fps_ + 0.5);
  now_real_ = 0;
  for (const auto& p : fr.persons) {
    // ghost 占位框不进任何统计（与 --dump / [Summary] 同口径），只进画面
    if (p.lost) { ++ghosts_; continue; }
    ++persons_;
    ++now_real_;
    auto& t = sum_[p.track_id];
    if (t.first < 0) { t.first = fr.index; t.sx = p.cx(); t.sy = p.cy(); t.sf = fr.index; }
    t.last    = fr.index;
    t.strokes = p.strokes;
    t.speed   = p.speed;
    t.lane    = lane_of(p.cy());
    if (std::isfinite(p.speed) && p.speed > t.vmax) t.vmax = p.speed;
    if (fr.index - t.sf >= std::max<int64_t>(dist_step, 1)) {
      t.dist_m += std::hypot(double(p.cx() - t.sx), double(p.cy() - t.sy)) / ppm_;
      t.sx = p.cx(); t.sy = p.cy(); t.sf = fr.index;
    }
    // 每秒给里程留一格快照（环形，只留最近十来格）。近期均速由此得来 ——
    // 见 recent_v()；不留逐帧历史是因为 sum_ 从不回收。
    const int64_t sec_step = std::max<int64_t>(int64_t(fps_ + 0.5), 1);
    if (t.rn == 0 || fr.index - t.rf[(t.rn - 1) % kRecentSlots] >= sec_step) {
      const int k = t.rn % kRecentSlots;
      t.rf[k] = fr.index;
      t.rd[k] = float(t.dist_m);
      ++t.rn;
    }
  }
  track_camera(sel);
}

// 近 kRecentSec 秒的均速：拿环里最老那格与当下作差。与 vavg 同口径（里程 ÷ 时长），
// 所以两者可直接比；样本不足 kRecentMinSec 秒时不给数（返回 NaN）。
double StatsBoard::recent_v(const Track& t) const {
  if (t.rn == 0) return NAN;
  const int k  = t.rn >= kRecentSlots ? t.rn % kRecentSlots : 0;
  const double dt = double(t.last - t.rf[k]) / fps_;
  return dt >= kRecentMinSec ? (t.dist_m - t.rd[k]) / dt : NAN;
}

// 跟随镜头：纵向锁泳道（泳姿会让 bbox 忽大忽小，按框定高会不停变焦），
// 横向一阶低通做延迟跟随（同理，且人游得快时镜头略微落后反而更像转播机位）。
void StatsBoard::track_camera(int sel) {
  const Person* p = nullptr;
  for (const auto& q : cur_)
    if (q.track_id == sel) { p = &q; break; }
  if (!p) { fsel_ = -1; return; }        // 未选或本帧不在场 -> 前端显示「已离场」
  // 纵向目标 = 所在泳道的中心线；在池岸（lane 0）时退回人本身的 y
  const int lane = lane_of(p->cy());
  const float ty = lane ? float((kGridMarginM + (lane - 0.5) * kLanePitchM) * ppm_)
                        : p->cy();
  if (fsel_ != sel) { fsel_ = sel; fx_ = p->cx(); fy_ = ty; return; }   // 刚选中即对准
  // alpha = 1 - exp(-dt/tau)：按秒定手感，换帧率或掉帧都是同一条时间常数
  const float a = float(1.0 - std::exp(-1.0 / (std::max(fps_, 1.0) * kFollowTauSec)));
  fx_ += a * (p->cx() - fx_);
  fy_ += a * (ty - fy_);                 // 纵向也过一遍：换道时是平移而非跳变
}

int StatsBoard::total_strokes() const {
  int n = 0;
  for (const auto& [id, t] : sum_) n += t.strokes;
  return n;
}

void StatsBoard::write_json(const std::string& path) const {
  std::ofstream f(path);
  SWIM_CHECK(f.good(), "无法写入 " + path);
  f << "{\n";
  bool first = true;
  for (const auto& [id, t] : sum_) {
    if (!first) f << ",\n";
    first = false;
    f << "  \"" << id << "\": {\"strokes\": " << t.strokes << ", \"speed\": ";
    if (std::isfinite(t.speed)) f << t.speed; else f << "null";
    f << "}";
  }
  f << "\n}\n";
  SWIM_CHECK(f.good(), "写入未完成 " + path);
}

bool StatsBoard::follow_rect(int id, int& x, int& y, int& w, int& h) const {
  if (id < 0 || id != fsel_) return false;
  // 高 = 一条泳道 + 上下留白，宽按 kFollowAspect 铺开（看板那一格很扁，
  // 按 16:9 裁会剩大片黑边）；两者都不超过画布。
  float hh = std::min(float((kLanePitchM + 2 * kFollowPadM) * ppm_), float(ch_));
  float ww = std::min(hh * float(kFollowAspect), float(cw_));
  w = int(ww) & ~1;                       // 取偶：给编码器省一次奇数宽的补齐
  h = int(hh) & ~1;
  x = std::min(std::max(int(fx_) - w / 2, 0), cw_ - w);
  y = std::min(std::max(int(fy_) - h / 2, 0), ch_ - h);
  return w > 0 && h > 0;
}

void StatsBoard::append_person_stats(std::string& s, int id, const Track& t) const {
  const double sec  = secs(t);
  const double vavg = sec > 0.5 ? t.dist_m / sec : NAN;
  const double dps  = t.strokes > 0 ? t.dist_m / t.strokes : NAN;
  put(s, "\"id\":%d,\"lane\":%d,\"strokes\":%d,\"sec\":%.1f,\"dist\":%.1f",
      id, t.lane, t.strokes, sec, t.dist_m);
  s += ",\"v\":";     put_num(s, t.speed);
  s += ",\"vmax\":";  put_num(s, t.vmax);
  // 均速用「里程/在场时长」而非瞬时速度的平均：后者会被丢检期间的空档抬高
  s += ",\"vavg\":";  put_num(s, vavg);
  s += ",\"spm\":";   put_num(s, sec > 1.0 ? t.strokes * 60.0 / sec : NAN, 1);
  s += ",\"dps\":";   put_num(s, dps);
  // 划水指数 = 均速 × 每划距离，泳界通用的效率指标（同样速度下越大越省力）
  s += ",\"si\":";    put_num(s, vavg * dps);
}

std::string StatsBoard::json(int sel) const {
  std::string s;
  s.reserve(4096);
  // 全场口径全部只看**本帧在场的真检出者**。刻意不报「累计人次 / 累计 track /
  // 划水合计 / 全场累计里程」：第一个是逐帧人数的和（没有物理意义），其余三个
  // 会被 ID 切换灌水，教练看到的只是噪声。
  int    lanes[kNumLanes + 1] = {};
  double vsum = 0.0, spm_sum = 0.0, dps_sum = 0.0, hot_v = 0.0;
  int    nv = 0, nspm = 0, ndps = 0, hot_id = -1;
  for (const auto& p : cur_) {
    if (p.lost) continue;                       // 在场人数/占位数在 update 里已数过
    ++lanes[lane_of(p.cy())];
    if (std::isfinite(p.speed)) { vsum += p.speed; ++nv; }
    const auto it = sum_.find(p.track_id);
    if (it == sum_.end()) continue;
    // 「最快」一律用近 kRecentSec 秒均速，不用瞬时：后者每帧换人，读数与自动
    // 接管的镜头都会乱闪。看板与自动接管共用这一个 hot_id / hot_v。
    const double rv = recent_v(it->second);
    if (std::isfinite(rv) && rv > hot_v) { hot_v = rv; hot_id = p.track_id; }
    if (it->second.strokes <= 0) continue;
    const double sec = secs(it->second);
    if (sec > 1.0) { spm_sum += it->second.strokes * 60.0 / sec; ++nspm; }
    dps_sum += it->second.dist_m / it->second.strokes;
    ++ndps;
  }

  put(s, "{\"i\":%lld,\"t\":%.1f,\"fps\":%.1f,\"all\":{",
      static_cast<long long>(index_), elapsed_, cur_fps_);
  put(s, "\"now\":%d,\"hotid\":%d,\"hotsec\":%d", now_real_, hot_id, kRecentSec);
  s += ",\"vavg\":";  put_num(s, nv ? vsum / nv : NAN);
  s += ",\"hotv\":";  put_num(s, hot_id >= 0 ? hot_v : NAN);
  s += ",\"spm\":";   put_num(s, nspm ? spm_sum / nspm : NAN, 1);
  s += ",\"dps\":";   put_num(s, ndps ? dps_sum / ndps : NAN);
  s += ",\"lanes\":[";
  for (int k = 1; k <= kNumLanes; ++k) put(s, k > 1 ? ",%d" : "%d", lanes[k]);
  s += "]}";

  // 逐人明细：前端据此重绘框/骨架/标签，所以框、关键点、分数都要带上。
  // 坐标一位小数已远超 5002 px 画布上的显示精度，能把体积压掉约三成。
  s += ",\"p\":[";
  bool first = true;
  for (const auto& p : cur_) {
    if (!first) s += ',';
    first = false;
    put(s, "{\"id\":%d,\"b\":[%.1f,%.1f,%.1f,%.1f],\"c\":%.2f,\"s\":%d,\"g\":%d",
        p.track_id, p.x1, p.y1, p.x2, p.y2, p.conf, p.strokes, p.lost);
    s += ",\"v\":"; put_num(s, p.speed);
    s += ",\"k\":[";
    for (int k = 0; k < kNumKpts; ++k)
      put(s, k ? ",%.1f,%.1f,%.2f" : "%.1f,%.1f,%.2f", p.kpts[k * 2],
          p.kpts[k * 2 + 1], p.scores[k]);
    s += "]}";
  }
  s += "]";

  if (sel >= 0) {
    const auto it = sum_.find(sel);
    if (it != sum_.end()) {
      s += ",\"sel\":{";
      append_person_stats(s, it->first, it->second);
      // 跟随窗矩形：前端据它把同一批关键点画到跟随视图上（不必再传一份坐标）。
      // 拿不到（该 id 本帧不在场）就不给 rect，前端据此显示「已离场」。
      int x, y, w, h;
      if (follow_rect(sel, x, y, w, h))
        put(s, ",\"rect\":[%d,%d,%d,%d]", x, y, w, h);
      s += "}";
    }
  }
  s += "}";
  return s;
}

std::string StatsBoard::meta_json() const {
  std::string s;
  put(s, "{\"w\":%d,\"h\":%d,\"ppm\":%g,\"fps\":%.4f,\"kpt_thr\":%g,\"nk\":%d",
      cw_, ch_, double(ppm_), fps_, double(kpt_thr_), kNumKpts);
  put(s, ",\"grid\":{\"margin\":%g,\"pitch\":%g,\"subdiv\":%d,\"lanes\":%d}",
      kGridMarginM, kLanePitchM, kLaneSubdiv, kNumLanes);
  s += ",\"skel\":[";
  bool first = true;
  for (const auto& e : kSkeleton) {
    put(s, first ? "[%d,%d]" : ",[%d,%d]", e[0], e[1]);
    first = false;
  }
  s += "]}";
  return s;
}

}  // namespace swim
