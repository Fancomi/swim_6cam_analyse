// 跟踪与运动指标：IoU 贪心跟踪、划水计数、瞬时速度。
//
// 判定准则与 Python 版（src/swim_analyse/tracking.py、metrics.py）逐项对齐：
//   跟踪   包含率去重已在 detect 端完成（yolo26 end2end），此处只做逐帧 IoU 贪心匹配
//   划水   原始信号 -> 缺帧线性插值 -> 中值(5) + Savitzky-Golay(11,3) 平滑 ->
//          按信号派发（同 metrics.py 的 SIGNALS 字典）：
//            elbow_angle   肩-肘-腕夹角，局部均值 + 拓扑 prominence 的波谷计数
//            wrist_x_head  手腕相对鼻子的 x 位移，滞回过零计数
//          自由泳/仰泳左右分开计数取 min，其余泳姿左右取均值后统一计数
//   速度   框中心位移 / ppm，每 0.2 s 一个采样点、窗口 2 s，采样点之间前向填充
//
// 刻意保留的设计差异（在线约束：live 场景不能等全序列结束）：
//   * Python 用 scipy.find_peaks 扫全序列，冲突的 distance 按峰高优先取舍；此处
//     在滑窗内逐位因果判定，最小间隔以"上一次已计数的事件"为基准，后来的更高峰
//     不会挤掉先前已计入的峰。
//   * 局部均值与 prominence 的搜索范围都限制在滑窗内（窗内平滑样本数
//     = cap_ - 2*kLag，比 Python 以候选点为中心的 ADAPTIVE_WINDOW 窄 2*kLag，
//     且位置偏向过去而非居中）。过零信号的滞回门限不受此限：它用累计全程幅度，
//     与 Python 的全序列 np.ptp 同义。
//   * 判定点滞后最新样本 7 位（中值 2 + SG 5 的右邻域），确保其平滑值不含边界
//     偏差；波谷判定再多滞后一个 MIN_STROKE_INTERVAL 供 prominence 向右回升。
//     因此 strokes() 的读数比画面晚约 0.1 s（+ 波谷再 0.67 s）。
#pragma once

#include <cmath>
#include <deque>
#include <limits>
#include <string>
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

/// 划水信号，与 metrics.py 的 SIGNALS 一一对应。
enum class StrokeSignal {
  ElbowAngle,    ///< 肩-肘-腕夹角的波谷（默认）
  WristXHead,    ///< 手腕沿前进方向相对鼻子的位移，由负转正过零
};

/// 单个 track 的运动指标（滑动窗口，适合 live）。
class MetricsTracker {
 public:
  /// fps 用于帧数与秒的换算；ppm 为画布每米像素数；kpt_thr 为关键点置信度门限
  /// （对应 CLI --kpt-thr）；split 为真时左右分开计数取 min（自由泳/仰泳），
  /// 否则左右取均值后统一计数；signal 选择划水信号。
  MetricsTracker(double fps, float ppm, float kpt_thr = 0.4f, bool split = true,
                 StrokeSignal signal = StrokeSignal::ElbowAngle);

  /// 泳姿 -> 是否左右分开计数（对齐 metrics.py 的 split_sides）。
  static bool split_sides(const std::string& stroke_type) {
    return stroke_type == "freestyle" || stroke_type == "backstroke";
  }

  /// 喂入一帧某个 track 的观测。
  void update(int track_id, int64_t frame, const Person& p);

  int   strokes(int track_id) const;
  /// 最近一次速度（m/s）；无数据返回 NaN。
  float speed(int track_id) const;
  /// 清理长时间未更新的 track，避免 live 场景内存增长。
  void prune(int64_t current_frame, int64_t max_idle);

 private:
  /// "从未发生"哨兵：远小于任何位置序号，且与之相减不会溢出。
  static constexpr int64_t kNever = std::numeric_limits<int64_t>::min() / 4;

  /// 一路信号的滑窗。位置 = 该 track 的第几次观测（与 Python 一致：信号数组按
  /// 观测序排列，未检出的帧不占位置），末尾 NaN 空洞在下一个有效样本到达时插值。
  struct Chan {
    std::deque<float> v;          ///< 已补齐的样本，末位对应位置 pos
    int64_t pos       = -1;       ///< v.back() 的位置序号
    int64_t hole      = 0;        ///< 末尾待插值补齐的 NaN 个数
    int64_t last_ev   = kNever;   ///< 上次计入的划水事件位置
    int64_t last_eval = kNever;   ///< 已判定过的最后位置
    int     count     = 0;
    bool    below = false, below_init = false;   ///< 过零检测的滞回状态
    /// 迄今所有平滑值的极值，滞回门限用它（对齐 Python 的全序列 np.ptp）。
    /// 只增不减，故是因果的、O(1) 的，不必留全序列。
    float   gmin =  std::numeric_limits<float>::max();
    float   gmax = -std::numeric_limits<float>::max();
    int64_t begin() const { return pos - int64_t(v.size()) + 1; }
  };
  struct Pt { int64_t f; float x, y; };

  struct State {
    Chan           ch[2];             ///< split 时为左/右，否则只用 ch[0]（左右均值）
    std::deque<Pt> center;            ///< 框中心轨迹，用于速度
    float   nose_x0 = std::nanf("");  ///< 首个可信的鼻子 x，用于定前进方向
    float   dir = 1.f;                ///< 前进方向符号（仅 wrist_x_head 会变）
    int64_t first_frame = -1;         ///< 速度采样网格的原点
    int64_t speed_slot  = -1;         ///< 已取值的采样格序号（格内前向填充）
    int64_t last_frame  = -1;
    float   speed = std::nanf("");
  };

  /// 本帧左右两路原始信号（NaN 表示该帧不可信）。
  void raw_signals(const Person& p, State& s, float& l, float& r) const;
  /// 压入一帧样本（NaN 只记账，等下一个有效值来插值），并把窗口裁到 cap_。
  void push(Chan& c, float x) const;
  /// 窗口内平滑，并按信号类型对新暴露的位置逐个判定（每个位置只判一次）。
  /// dir 为信号方向符号，在此处施加（中值与 SG 都与取负可交换）。
  void detect(Chan& c, float dir);
  /// 波谷计数：对齐 find_peaks(-x, distance, height=局部均值, prominence=局部幅度*0.2)。
  /// p 为位置序号，t 为其在 sm_ 中的下标。
  void count_valley(Chan& c, int64_t p, int t, float mean, float span) const;
  /// 滞回过零计数：先跌破 -delta 再升破 +delta 记一次。
  void count_cross(Chan& c, int64_t p, int t, float delta) const;
  void update_speed(State& s, int64_t frame) const;

  std::unordered_map<int, State> st_;
  std::vector<float> med_, sm_;      ///< 平滑用的复用缓冲（热路径不做堆分配）
  double  fps_;
  float   ppm_, kpt_thr_;
  bool    split_;
  StrokeSignal signal_;
  int64_t gap_stroke_, gap_zc_;      ///< 两次事件的最小间隔（样本数）
  int64_t look_;                     ///< 判定点右侧前视（样本数），供拓扑 prominence
  int64_t cap_;                      ///< 信号滑窗样本数，同时是局部自适应统计窗宽
  double  keep_speed_;               ///< 速度窗跨度（帧）
  int64_t speed_step_;               ///< 速度采样格步长（帧）
};

}  // namespace swim
