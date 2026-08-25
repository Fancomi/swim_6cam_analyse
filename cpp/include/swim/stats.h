// 统计看板：把每帧的 FrameResult 汇总成「全场 + 个人」两级指标，并序列化成
// 前端要的 JSON。**只做汇总与序列化，不碰 HTTP、不碰 OpenCV**。
//
// 与 metrics.cpp 的分工：那边算划水与瞬时速度（与 Python 侧逐项对齐，属同步契约）；
// 这里只做纯累计派生量（里程、均速、峰值、泳道、SPM），不参与任何对齐口径，
// 因此改这里不会动 scripts/test.sh 的基线数字。
//
// 它同时接管了 main.cpp 原先手写的收尾汇总（track -> 划水/末速 的 map、人次、
// ghost 数、--json 落盘），于是「累计什么」只有一处定义。
#pragma once

#include <map>
#include <string>
#include <vector>

#include "swim/common.h"

namespace swim {

class StatsBoard {
 public:
  /// canvas_w/h 为画布像素尺寸，ppm 为每米像素数，fps 为时间轴帧率，
  /// kpt_thr 供前端按同一门限决定画哪些关键点。
  StatsBoard(int canvas_w, int canvas_h, float ppm, double fps, float kpt_thr);

  /// 喂入一帧（在后处理线程调用，与 json()/follow_rect() 同线程）。
  /// elapsed_sec 是进程已运行秒数，用于实时吞吐显示；sel 是前端选中的 track
  /// （<0 = 未选），跟随镜头的平滑状态在此推进 —— 不放 follow_rect() 里是因为
  /// 那个一帧要被调两次（推跟随流 + 写进 /stats），状态会多走一步。
  void update(const FrameResult& fr, double elapsed_sec, int sel = -1);

  /// 本帧统计 JSON。sel 为前端选中的 track（<0 = 未选）。
  std::string json(int sel) const;
  /// 运行期不变的几何与绘制口径（画布尺寸、ppm、骨架边表、网格常数）。
  std::string meta_json() const;

  /// 选中者的跟随裁切窗（纵向锁泳道、横向延迟跟随，钳在画布内）。
  /// id 与最近一次 update() 的 sel 不符、或那人本帧不在场，返回 false。
  bool follow_rect(int id, int& x, int& y, int& w, int& h) const;

  /// 显示编号「x 道 y 号」。取到即填 lane/slot 返回 true；那人在池岸（不属于任何
  /// 泳道）或号已随离场回收时返回 false，由调用方决定退化文案。
  /// 给 OpenCV 那条渲染路用 —— 看板走 json() 里下发的 ln/sl，同一份状态。
  bool display_no(int id, int& lane, int& slot) const;

  // ── 收尾汇总（口径与改造前逐字节一致）──────────────────────────────────
  int64_t persons() const { return persons_; }    ///< 累计人次（只计真检出）
  int64_t ghosts()  const { return ghosts_; }     ///< 累计 ghost 占位框
  size_t  tracks()  const { return sum_.size(); }
  int     now_real() const { return now_real_; }  ///< 本帧真检出人数（--show-fps 用）
  int     total_strokes() const;
  /// 写 --json：{"<tid>": {"strokes": N, "speed": F|null}}，按 tid 升序。
  void write_json(const std::string& path) const;

 private:
  /// 「近期均速」的观察窗（秒）。看板的「最快」与自动接管**共用这一个口径** ——
  /// 瞬时速度每帧换人，读数与镜头都会乱闪。5 秒够覆盖两三个划水周期。
  static constexpr int kRecentSec   = 5;
  /// 里程快照格数（每秒一格，环形覆盖）。取最老那格与当下作差即近期均速。
  /// 刻意不留逐帧历史 —— `sum_` 从不回收，每 track 多一份长数组就是长跑时的常驻
  /// 内存，而「谁最近游得快」并不需要那个精度。
  static constexpr int kRecentSlots = kRecentSec + 1;

  /// 单个 track 的累计量。全部是「只增」的派生量，故 O(1) 更新、无需留历史。
  struct Track {
    int     strokes = 0;
    float   speed   = 0.f;      ///< 末次瞬时速度（可能 NaN）
    float   vmax    = 0.f;
    double  dist_m  = 0.0;      ///< 采样后的累计里程
    int64_t first = -1, last = -1;
    float   sx = 0.f, sy = 0.f; ///< 上一个里程采样点（画布像素）
    int64_t sf = -1;            ///< 该采样点的帧号
    int     lane = 0;           ///< 带迟滞的泳道（0 = 池岸），见 lane_at()
    int     slot = 0;           ///< 道内占位号（1 起，0 = 还没分配），见 take_slot()
    int64_t rf[kRecentSlots] = {};  ///< 里程快照的帧号（环，按 rn 推进）
    float   rd[kRecentSlots] = {};  ///< 快照当时的累计里程
    int     rn = 0;                 ///< 已推入的快照数（只增，不回绕）
  };

  int    lane_of(float cy) const;
  /// 带迟滞的泳道判定：从 prev 换出去要越过分道绳再深入 kLaneHystM 米。
  int    lane_at(float cy, int prev) const;
  /// 取 lane 道内 id 的占位号（已占则返回原号，否则补最小空号，满了才加新号）。
  int    take_slot(int lane, int id);
  /// 归还占位号（换道时调用；彻底离场的人由 take_slot 按帧号回收）。
  void   free_slot(int lane, int id);
  double secs(const Track& t) const;
  /// 近 kRecentSec 秒的均速（里程差 ÷ 时长差，与 vavg 同口径）；样本不足返回 NaN。
  double recent_v(const Track& t) const;
  /// 推进跟随镜头的平滑状态（每帧一次，见 update 的注释）。
  void   track_camera(int sel);
  /// 追加一个 track 的个人指标字段（不含外层花括号）。
  void   append_person_stats(std::string& s, int id, const Track& t) const;

  int    cw_, ch_;
  float  ppm_, kpt_thr_;
  double fps_;

  std::map<int, Track> sum_;           ///< track -> 累计（有序，收尾输出稳定）
  /// 每条道的占位表（下标 0 = 池岸）：`occ_[lane][slot-1] = track id`，-1 = 空号。
  /// 「补最小空号」就是找第一个 -1，所以下标本身即号码，不需要另立号池。
  /// 只服务显示，任何统计都不看它 —— 见头注释「两套编号」。
  std::vector<int>     occ_[kNumLanes + 1];
  std::vector<Person>  cur_;           ///< 本帧人物（含 ghost，供前端画面一致）
  int64_t index_   = -1;
  int64_t persons_ = 0, ghosts_ = 0;
  int     now_real_ = 0;               ///< 本帧真检出人数（不含 ghost 占位）
  double  elapsed_ = 0.0, cur_fps_ = 0.0;
  double  fps_t0_  = 0.0;              ///< 吞吐采样窗起点
  int64_t fps_i0_  = 0;
  int     fsel_ = -1;                  ///< 跟随状态属于哪个 track（-1 = 无）
  float   fx_ = 0.f, fy_ = 0.f;        ///< 平滑后的镜头中心（画布像素）
};

}  // namespace swim
