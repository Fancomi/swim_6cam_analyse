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
/// 换道迟滞（米）：越过分道绳后还要再深入这么多才算换了道。压着绳游时 bbox 中心
/// 会在绳上来回，不设迟滞道号每帧跳，道内占位号也跟着乱换。
constexpr double kLaneHystM = 0.35;
/// 占位号的保留时长（秒）。一丢检就腾号会让遮挡后回来的人换个号，比号码空着更
/// 难认；比 `--track-max-lost`（默认 30 帧）留宽些即可，track 销毁后 id 不会再来。
constexpr double kSlotHoldSec = 1.5;

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

// 迟滞：要离开 prev 道，得越过它的分道绳再深入 kLaneHystM 米。压着绳游时 bbox
// 中心会在绳上来回，直接用 lane_of 会让道号每帧跳，道内占位号也跟着乱换。
int StatsBoard::lane_at(float cy, int prev) const {
  const int k = lane_of(cy);
  if (k == prev || prev < 1 || prev > kNumLanes) return k;
  const double hyst = kLaneHystM * ppm_;
  const double top  = (kGridMarginM + (prev - 1) * kLanePitchM) * ppm_ - hyst;
  const double bot  = (kGridMarginM + prev * kLanePitchM) * ppm_ + hyst;
  return double(cy) >= top && double(cy) <= bot ? prev : k;
}

// 占位号：下标即号码，找第一个 -1 就是「补最小空号」，所以不需要另立号池。
// 每帧对在场者都调一次（按 id 命中即返回原号），于是号码被回收后又出现的人
// 会自动重新拿号，不必在别处维护「我的号还有效吗」。
int StatsBoard::take_slot(int lane, int id) {
  auto& v = occ_[lane];
  for (size_t i = 0; i < v.size(); ++i) if (v[i] == id) return int(i) + 1;
  for (size_t i = 0; i < v.size(); ++i) if (v[i] < 0) { v[i] = id; return int(i) + 1; }
  v.push_back(id);
  return int(v.size());
}

void StatsBoard::free_slot(int lane, int id) {
  auto& v = occ_[lane];
  for (auto& x : v) if (x == id) { x = -1; break; }
  while (!v.empty() && v.back() < 0) v.pop_back();
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
    // 泳道与「道内占位号」：换道即还旧号、取新号，原始 track id 不变。
    // 号码只服务显示，任何统计都按 track id 走，所以穿行几条道也不会串。
    // 池岸（lane 0）不给号 —— 那里不是泳道，前端退回显示原始 id。
    const int lane = lane_at(p.cy(), t.lane);
    if (lane != t.lane) { free_slot(t.lane, p.track_id); t.lane = lane; }
    t.slot = lane ? take_slot(lane, p.track_id) : 0;
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
  // 回收离场者的占位号：丢检满 kSlotHoldSec 才腾，短暂遮挡不换号（换号比号码
  // 空着更难认）。扫的是占位表而不是 sum_ —— 后者从不回收、越跑越长，而占位表
  // 的规模就是「此刻各道有几个人」。就地置 -1 而不是走 free_slot：那个会 pop_back
  // 尾部空位，边遍历边缩会让引用失效，所以缩表放循环后。
  const int64_t hold = int64_t(kSlotHoldSec * fps_ + 0.5);
  for (int lane = 1; lane <= kNumLanes; ++lane) {
    auto& v = occ_[lane];
    for (int& id : v) {
      if (id < 0) continue;
      const auto it = sum_.find(id);
      if (it == sum_.end() || fr.index - it->second.last > hold) id = -1;
    }
    while (!v.empty() && v.back() < 0) v.pop_back();
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

// 显示编号只是把占位表里的结果读出来 —— 谁都不重算，于是「窗口/落盘 mp4 的标签」
// 与「看板下发的 ln/sl」必然是同一个号。
bool StatsBoard::display_no(int id, int& lane, int& slot) const {
  const auto it = sum_.find(id);
  if (it == sum_.end() || it->second.slot <= 0) return false;
  lane = it->second.lane;
  slot = it->second.slot;
  return true;
}

void StatsBoard::append_person_stats(std::string& s, int id, const Track& t) const {
  const double sec  = secs(t);
  const double vavg = sec > 0.5 ? t.dist_m / sec : NAN;
  const double dps  = t.strokes > 0 ? t.dist_m / t.strokes : NAN;
  put(s, "\"id\":%d,\"lane\":%d,\"slot\":%d,\"strokes\":%d,\"sec\":%.1f,\"dist\":%.1f",
      id, t.lane, t.slot, t.strokes, sec, t.dist_m);
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
  // hot_v 从 -1 起比而不是 0：均速恒 >= 0，从 0 起比会把「在场但没游动的人」排除掉，
  // 于是泡在水里没人游时 hotid 恒为 -1、自动接管找不到落点。语义因此是
  // 「在场且观察够 kRecentMinSec 秒的人里均速最高的那个（可以是 0）」。
  double vsum = 0.0, spm_sum = 0.0, dps_sum = 0.0, hot_v = -1.0;
  int    nv = 0, nspm = 0, ndps = 0, hot_id = -1;
  for (const auto& p : cur_) {
    if (p.lost) continue;                       // 在场人数/占位数在 update 里已数过
    const auto it = sum_.find(p.track_id);
    if (it == sum_.end()) continue;
    // 分道占用按**带迟滞的道号**数（Track::lane），与标签上显示的道号同一个值 ——
    // 用 lane_of 现算会让压绳游的人在两道之间来回计数，与他标签上的道对不上。
    ++lanes[it->second.lane];
    if (std::isfinite(p.speed)) { vsum += p.speed; ++nv; }
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
  // id 是原始 track id（选人、跟随、统计一律用它）；ln/sl 是显示用的「x 道 y 号」，
  // 由服务端算好下发 —— 占位表是有状态的，前端逐帧重算不出同一个号。
  s += ",\"p\":[";
  bool first = true;
  for (const auto& p : cur_) {
    if (!first) s += ',';
    first = false;
    const auto it = sum_.find(p.track_id);
    const int ln = it == sum_.end() ? 0 : it->second.lane;
    const int sl = it == sum_.end() ? 0 : it->second.slot;
    put(s, "{\"id\":%d,\"ln\":%d,\"sl\":%d,\"b\":[%.1f,%.1f,%.1f,%.1f]"
           ",\"c\":%.2f,\"s\":%d,\"g\":%d",
        p.track_id, ln, sl, p.x1, p.y1, p.x2, p.y2, p.conf, p.strokes, p.lost);
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
