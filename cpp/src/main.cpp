// CLI：离线视频 / live 流的实时游泳分析。
//
//   swim_analyse --input data/xxx.mp4 --models cpp/models [--out out.mp4] [--json r.json]
//   swim_analyse --input rtsp://...   --models cpp/models --preview --show-fps
//
// 可视化用 OpenCV 在 CPU 侧画（只在需要落盘/预览时才 D2H 整帧；纯分析模式
// 不拷图像，全链路只回读关键点）。--out 落盘、--preview 开窗，两者可独立或同用。
#include <opencv2/opencv.hpp>

#ifdef _WIN32
// windows.h 默认定义 min/max 宏，会打断 std::max / cv:: 里的模板调用
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "swim/frame_source.h"
#include "swim/metrics.h"
#include "swim/pipeline.h"
#include "swim/proc.h"
#ifdef SWIM_HAS_STITCH
#include "swim/stitch.h"
#endif

using namespace swim;

namespace {

// COCO17 骨架：边表与 Python 版 pose.py 的 SKELETON 逐项一致；
// 那边按左右肢分色，这里统一绿色（C++ 侧只用于目视核对，不追求配色一致）
constexpr int kSkel[][2] = {{15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},
                            {5,6},{5,7},{7,9},{6,8},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4}};

cv::Scalar id_color(int id) {           // 与 Python 版同思路：按 id 稳定散列
  const uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
  return cv::Scalar((h >> 16) & 255, (h >> 8) & 255, h & 255);
}

/// 叠加层的三个开关。预览窗口里按 1/2/3 实时切换，`--out` 落盘用当前状态
/// （落盘没有键盘，所以静态开关 --no-kpts / --grid 也要保留）。
/// 一个结构体而不是三个 bool 参数：draw()、Preview::show() 与 main 的 sink
/// 共用同一份状态，加第四个开关时不必再改一遍签名。
struct Overlay {
  bool kpts  = true;                    // 1: 人体关键点骨架
  bool boxes = true;                    // 2: 检测框 + ID/划水/速度 标签
  bool grid  = false;                   // 3: 米制标尺网格（按 --ppm）
};

struct Args {
  PipelineOptions opt;                  // 参数默认值只在 pipeline.h 定义一份
                                        // （--help 也从这里现取，见 help_rows）
  std::string input, out, json, dump;
  std::string cam_dir, lut;             // 六路拼接输入：片段目录 + 查找表
  std::string dump_canvas;              // 首帧画布原样落 PNG（拼接对照用）
  Overlay     ov;                       // 叠加层初始状态
  bool        show_fps = false, preview = false;
  float       preview_scale = 0.f;      // 0 = 自适应到 1600x900 以内
  float       fps = 0.f;                // 0 = 用源自报的帧率
  DecoderPref decoder   = DecoderPref::Auto;
};

/// --queue-depth 的上限，只在 CLI 这一层拦：队列每格在渲染模式下对应一整帧
/// 锁页内存（5002x2102 约 31.5 MB），16 格已经是 500 MB 量级，再大只会白占内存。
/// 不放进 validate() 是因为库内部（非 CLI 调用方）可以自行决定更大的深度。
constexpr int kMaxQueueDepth = 16;

/// 带区间校验的数值解析：std::stoi/stof 对 "abc" 抛异常、对 "12abc" 静默截断，
/// 两种都要在这里拦住，否则 --ppm 0 之类会一路走到"速度 = inf"。
double parse_num(const char* s, const std::string& key, double lo, double hi,
                 bool integer) {
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  SWIM_CHECK(end && end != s && *end == '\0', key + " 需要数值，收到 " + s);
  SWIM_CHECK(v >= lo && v <= hi,
             key + " 超出范围 [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
  SWIM_CHECK(!integer || v == std::floor(v), key + " 需要整数");
  return v;
}

// ── --help 表 ─────────────────────────────────────────────────────────────
// 帮助文本里的默认值全部从 PipelineOptions{} / Args{} 的字段现取现格式化，
// 于是「校验、行为、帮助」三处共用同一份默认值，改 pipeline.h 一处即三处同步
// （早先 help 把 0.25/0.4/100 等硬编码了一遍，改默认值必漏其中之一）。

/// printf 风格拼 std::string。帮助文本要就地插默认值，用一次性缓冲比拼 ostream 短。
std::string sfmt(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

/// 枚举默认值也只写一份：--signal / --decoder 的默认取自字段，名字在这里翻译。
const char* signal_name(StrokeSignal s) {
  return s == StrokeSignal::WristXHead ? "wrist_x_head" : "elbow_angle";
}
const char* decoder_name(DecoderPref d) {
  return d == DecoderPref::Cpu ? "cpu" : "auto";
}

/// 一行帮助。arg 是参数占位，**保持 ASCII**：列宽按 strlen 算字节数，
/// 占位里放中文会让这一列错位。desc 里的 '\n' 表示续行，由 print_help 缩进对齐。
struct HelpRow {
  const char* opt;
  const char* arg;
  std::string desc;
};

std::vector<HelpRow> help_rows() {
  const PipelineOptions d{};        // 流水线参数默认值的唯一来源
  const Args a{};                   // 非流水线参数（预览、绘制开关）的默认值
  return {
    {"--input", "FILE|URL", "已拼好的画布视频或流地址（rtsp:// 等）"},
    {"--cam-dir", "DIR|LIST", "六路输入，GPU 上实时拼接后直接分析（与 --input 二选一）\n"
                              "目录 = 离线片段（名字以 _<相机>.mp4 结尾）\n"
                              "文件 = 相机清单，每行 <相机>=<地址>，地址可为 rtsp://"},
    {"--stitch-lut", "FILE", sfmt("拼接查找表 (默认 <models>/stitch.lut，\n"
                                  "由 cpp/tools/build_stitch_lut.py 生成)")},
    {"--models", "DIR", sfmt("detect.onnx/pose.onnx 所在目录 (默认 %s)",
                             d.models_dir.c_str())},
    {"--out", "FILE", "写出标注视频 (H.264)"},
    {"--json", "FILE", "写出每个 track 的划水次数与速度"},
    {"--dump", "FILE", "逐帧写出每个框(帧号,id,xyxy,conf)+17 个关键点\n"
                       "(kx,ky,ks)，用于与 Python 对照"},
    {"--dump-canvas", "FILE", "把首帧画布原样存成 PNG（未画标注），\n"
                              "用于与离线拼接逐像素对照"},
    {"--max-frames", "N", sfmt("只处理前 N 帧 (默认 %lld = 不限)",
                               static_cast<long long>(d.max_frames))},
    {"--detect-size", "N", sfmt("detect 输入方形边长, 须与 onnx 一致 (默认 %d)",
                                d.detect_size)},
    {"--max-persons", "N", sfmt("pose 最大人数，显存按此预留 (默认 %d, 上限 %d)",
                                d.max_persons, kMaxPersons)},
    {"--conf", "F", sfmt("检测置信度阈值 [0,1) (默认 %g)", d.conf_thr)},
    {"--kpt-thr", "F", sfmt("关键点置信度阈值 [0,1) (默认 %g)", d.kpt_thr)},
    {"--ppm", "F", sfmt("画布每米像素数 (默认 %g)", d.ppm)},
    {"--containment", "F", sfmt("包含率去重阈值(交集/自身面积) (0,1] (默认 %g)",
                                d.containment)},
    {"--track-iou", "F", sfmt("跟踪匹配的 IoU 阈值 (0,1] (默认 %g)", d.track_iou)},
    {"--track-max-lost", "N", sfmt("track 连续丢失多少帧后销毁 (默认 %d)",
                                   d.track_max_lost)},
    {"--queue-depth", "N", sfmt("推理->后处理队列深度 1..%d (默认 %d)。渲染时每格\n"
                                "多占一整帧锁页内存，调大只在后处理抖动时有用",
                                kMaxQueueDepth, d.queue_depth)},
    {"--stroke-type", "S", sfmt("泳姿，决定左右是否分开计数 (默认 %s)\n可选: %s",
                                d.stroke_type.c_str(), kStrokeTypes)},
    {"--signal", "S", sfmt("划水信号 elbow_angle|wrist_x_head (默认 %s)",
                           signal_name(d.signal))},
    {"--decoder", "MODE", sfmt("auto|cpu (默认 %s=ffmpeg 管道，探测失败回退 OpenCV;\n"
                               "cpu=强制 OpenCV; nvdec 已废弃，等价 auto)",
                               decoder_name(a.decoder))},
    {"--no-kpts", "", sfmt("启动时不画骨架 (默认%s画，预览中按 1 切换)",
                           a.ov.kpts ? "" : "不")},
    {"--no-boxes", "", sfmt("启动时不画框与标签 (默认%s画，预览中按 2 切换)",
                            a.ov.boxes ? "" : "不")},
    {"--grid", "", sfmt("启动时画米制标尺网格，每 1 m 一条 (默认%s画，\n"
                        "预览中按 3 切换；间距由 --ppm 决定)",
                        a.ov.grid ? "" : "不")},
    {"--fps", "F", "覆盖时间轴与限速用的帧率 (默认取源自报的值)。\n"
                   "相机改成 30fps 而流里报错时用它对齐划水/速度"},
    {"--preview", "", "开实时窗口（画布会缩放后显示，按 q/ESC 退出，\n"
                      "1/2/3 切关键点/分析/网格）"},
    {"--preview-scale", "F", "预览缩放比 0.05~1（默认自适应 1600x900 以内），\n"
                             "给了它就等于同时开 --preview"},
    {"--show-fps", "", "实时打印吞吐"},
    {"--fp32", "", "engine 算子用 fp32 (默认 fp16)。注意 ONNX 若是\n"
                   "fp16 权重导出的，这只放宽算子精度，不等于真 fp32"},
    {"-h, --help", "", "打印本帮助并退出"},
  };
}

/// 打印 --help。列宽取表内最长的「选项 + 占位」，说明的续行自动缩进到同一列，
/// 于是加参数时不必手数空格。
void print_help() {
  const auto rows = help_rows();
  size_t w = 0;
  for (const auto& r : rows)
    w = std::max(w, strlen(r.opt) + (*r.arg ? strlen(r.arg) + 1 : 0));

  printf("用法: swim_analyse (--input <画布视频|流> | --cam-dir <六路片段目录>) [选项]\n");
  for (const auto& r : rows) {
    std::string head = r.opt;
    if (*r.arg) head += " " + std::string(r.arg);
    bool first = true;
    for (size_t p = 0; p <= r.desc.size();) {
      const size_t nl = r.desc.find('\n', p);
      const std::string seg = r.desc.substr(p, nl - p);   // npos 时取到末尾
      if (first) printf("  %-*s  %s\n", int(w), head.c_str(), seg.c_str());
      else       printf("  %-*s  %s\n", int(w), "", seg.c_str());
      first = false;
      if (nl == std::string::npos) break;
      p = nl + 1;
    }
  }
}

bool parse(int argc, char** argv, Args& a) {
  auto need = [&](int& i) { SWIM_CHECK(i + 1 < argc, "参数缺少值"); return argv[++i]; };
  auto num  = [&](int& i, const std::string& k, double lo, double hi) {
    return static_cast<float>(parse_num(need(i), k, lo, hi, false));
  };
  auto integer = [&](int& i, const std::string& k, long lo, long hi) {
    return static_cast<long>(parse_num(need(i), k, double(lo), double(hi), true));
  };
  // 区间 (0,1]：parse_num 的边界是闭的，0 要单独挡 —— 包含率/IoU 取 0 会让
  // 去重与跟踪匹配退化成「任意两框都算同一目标」，validate() 也会拦，
  // 但在这里报错才带得上参数名。
  auto frac = [&](int& i, const std::string& k) {
    const float v = num(i, k, 0.0, 1.0);
    SWIM_CHECK(v > 0.f, k + " 须在 (0,1]");
    return v;
  };
  auto& o = a.opt;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if      (k == "--input")       a.input = need(i);
    else if (k == "--cam-dir")     a.cam_dir = need(i);
    else if (k == "--stitch-lut")  a.lut = need(i);
    else if (k == "--models")      o.models_dir = o.engine_dir = need(i);
    else if (k == "--out")         a.out = need(i);
    else if (k == "--json")        a.json = need(i);
    else if (k == "--dump")        a.dump = need(i);
    else if (k == "--dump-canvas") a.dump_canvas = need(i);
    else if (k == "--max-frames")  o.max_frames  = integer(i, k, 0, 1 << 30);
    else if (k == "--detect-size") o.detect_size = int(integer(i, k, 64, 4096));
    else if (k == "--max-persons") o.max_persons = int(integer(i, k, 1, kMaxPersons));
    else if (k == "--conf")        o.conf_thr = num(i, k, 0.0, 0.999);
    else if (k == "--kpt-thr")     o.kpt_thr  = num(i, k, 0.0, 0.999);
    else if (k == "--ppm")         o.ppm      = num(i, k, 1e-3, 1e5);
    else if (k == "--containment") o.containment = frac(i, k);
    else if (k == "--track-iou")   o.track_iou   = frac(i, k);
    else if (k == "--track-max-lost")
      o.track_max_lost = int(integer(i, k, 1, 1 << 20));
    else if (k == "--queue-depth")
      o.queue_depth = int(integer(i, k, 1, kMaxQueueDepth));
    else if (k == "--stroke-type") o.stroke_type = need(i);
    else if (k == "--signal") {
      const std::string v = need(i);
      SWIM_CHECK(v == "elbow_angle" || v == "wrist_x_head",
                 "--signal 只能是 elbow_angle|wrist_x_head");
      o.signal = v == "wrist_x_head" ? StrokeSignal::WristXHead
                                     : StrokeSignal::ElbowAngle;
    }
    else if (k == "--no-kpts")     a.ov.kpts  = false;
    else if (k == "--no-boxes")    a.ov.boxes = false;
    else if (k == "--grid")        a.ov.grid  = true;
    else if (k == "--fps")         a.fps = num(i, k, 1e-3, 1000.0);
    else if (k == "--show-fps")    a.show_fps = true;
    else if (k == "--preview")     a.preview = true;
    else if (k == "--preview-scale") {
      a.preview = true;                 // 给了比例显然是要预览，免得漏加 --preview
      a.preview_scale = num(i, k, 0.05, 1.0);
    }
    else if (k == "--fp32")        o.fp16 = false;
    else if (k == "--decoder") {
      // 严格校验：拼错的值（如 cpuu）必须报错。早先的三元链把一切非
      // nvdec/cpu 的输入都当 auto，于是 --decoder cpuu 会静默走 ffmpeg 路径，
      // 用它做「强制 OpenCV」的像素对照会得出完全错误的结论。
      const std::string v = need(i);
      if (v == "nvdec") {
        printf("[Source] --decoder nvdec 已废弃：CUVID H.264 上限 4096 "
               "装不下 5002 宽画布，按 auto 处理\n");
        a.decoder = DecoderPref::Auto;
      } else {
        SWIM_CHECK(v == "auto" || v == "cpu",
                   "--decoder 只能是 auto|cpu，收到 " + v);
        a.decoder = v == "cpu" ? DecoderPref::Cpu : DecoderPref::Auto;
      }
    } else if (k == "-h" || k == "--help") {
      print_help();
      return false;
    } else {
      throw std::runtime_error("未知参数 " + k);
    }
  }
  SWIM_CHECK(a.input.empty() != a.cam_dir.empty(),
             "--input（已拼画布）与 --cam-dir（六路实时拼接）必须且只能给一个");
  if (a.lut.empty()) a.lut = o.models_dir + "/stitch.lut";
  // 只有要图像（落盘、预览或导出画布）才付整帧 D2H；纯分析模式只回读关键点
  o.need_image = !a.out.empty() || a.preview || !a.dump_canvas.empty();
  o.validate();
  return true;
}

/// 按参数选帧源：给了 --cam-dir 走六路 NVDEC 拼接，否则读已拼好的画布。
/// 两条路都产出同一个 GpuFrame（BGR uint8 显存），下游完全不感知差异。
std::unique_ptr<FrameSource> make_source(const Args& a) {
  if (a.cam_dir.empty()) return FrameSource::open(a.input, a.decoder, a.fps);
#ifdef SWIM_HAS_STITCH
  return StitchSource::open(a.cam_dir, a.lut, a.fps);
#else
  throw std::runtime_error(
      "本二进制未编入拼接源（configure 时没找到 FFmpeg libav*），"
      "--cam-dir 不可用；请装 ffmpeg 开发包后重新构建，或改用 --input");
#endif
}

/// H.264 写出器：先试 OpenCV(avc1→mp4v)，都开不了则退到外部 ffmpeg 管道。
///
/// 回退存在的原因：Windows 上 vcpkg 的 opencv4 未启用 ffmpeg 特性，videoio 只剩
/// MSMF。MSMF 的 H.264 编码器在 1080p/4K 能开，但画布分辨率 5002x2102 直接
/// isOpened()==false。同机 NVENC 也用不了（驱动 571.96 只提供 NVENC API 13.0，
/// ffmpeg 8.1 要求 13.1），所以编码器定为 libx264。
/// 走独立进程还有个附带好处：编码与本进程的 GPU 推理天然并行。
class Writer {
 public:
  Writer(const std::string& path, int w, int h, double fps) {
    for (const char* cc : {"avc1", "mp4v"}) {
      w_.open(path, cv::VideoWriter::fourcc(cc[0], cc[1], cc[2], cc[3]), fps,
              {w, h});
      if (w_.isOpened()) {
        printf("[Writer] %s codec=%s (opencv)\n", path.c_str(), cc);
        return;
      }
    }
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt bgr24 "
             "-s %dx%d -r %.4f -i - -c:v libx264 -preset veryfast -crf 23 "
             "-pix_fmt yuv420p \"%s\"", w, h, fps, path.c_str());
    // 写侧不需要大缓冲：编码比我们写得快，攒下的字节数就是 ffmpeg 的一次读间隔
    pipe_ = std::make_unique<Proc>(cmd, Proc::Mode::Write);
    SWIM_CHECK(bool(*pipe_), "无法打开输出 " + path + "（OpenCV 与 ffmpeg 管道均失败）");
    bytes_ = size_t(w) * h * 3;
    printf("[Writer] %s codec=libx264 (ffmpeg 管道)\n", path.c_str());
  }

  /// 析构只收尸不报错：正常路径由 main 显式调 close() 检查退出码，
  /// 异常路径已经在抛别的东西了，析构里再抛会直接 terminate。
  /// 收尸本身由 ~Proc 完成（幂等）。
  ~Writer() = default;

  void write(const cv::Mat& m) {
    if (!pipe_) { w_.write(m); return; }
    SWIM_CHECK(m.isContinuous(), "帧数据非连续，无法直接写管道");
    ++frames_;
    if (fwrite(m.data, 1, bytes_, pipe_->file()) == bytes_) return;
    // 短写只说明管道断了，真正的原因在 ffmpeg 的退出码里 —— 先收尸拿到它，
    // 否则只剩"写失败"这种没有指向性的报错，还会漏掉断在第几帧
    throw std::runtime_error("写 ffmpeg 管道失败：第 " + std::to_string(frames_) +
                             " 帧短写，ffmpeg 退出码 " +
                             std::to_string(pipe_->close()) +
                             "（其 stderr 见上方；PATH 里有 ffmpeg 吗？）");
  }

  /// 显式关闭：管道要等 ffmpeg 落完 moov 才算写完，别拖到 main 结束后。
  /// 退出码非 0 必须报出来，否则视频缺 moov 而程序声称成功。
  void close() {
    const int code = pipe_ ? pipe_->close() : 0;   // OpenCV 模式无子进程
    SWIM_CHECK(code == 0, "ffmpeg 编码进程退出码 " + std::to_string(code) +
                              "，视频可能不完整（已写 " +
                              std::to_string(frames_) + " 帧）");
  }

 private:
  cv::VideoWriter w_;
  std::unique_ptr<Proc> pipe_;         // 空 = 走 OpenCV VideoWriter
  size_t  bytes_ = 0;
  int64_t frames_ = 0;
};

/// 画布纵向两端各有一条池岸不属于水面，网格的 0 m 线要落在池边而不是画布边缘。
/// 取值来自标定本身：`configs/pool_mesh.json` 的 y 顶点是
/// 4.2358 / 4.7358 / 7.2358 … 22.2358 / 24.7358 / 25.2358 —— 中间 9 行按 2.5 m
/// 等距（8 条泳道，正好是转出的画布里那 9 排红色分道绳），首尾各多出 0.5 m。
/// 所以画布 21 m = 0.5 + 20 + 0.5，横向 50 m 是完整池长、不需要内缩。
/// 改了标定就核这几个顶点；写死一个常数是刻意的 —— 网格只是目视标尺，
/// 不值得为它把 LUT 的几何再读一遍进来。
constexpr double kGridMarginM = 0.5;

/// 米制标尺网格：按 --ppm 每 1 m 一条线，每 5 m 加粗并标米数，左下角写出泳池的
/// 米制尺寸。用途是目视量距离、核对速度口径（速度 = 框中心位移 / ppm），
/// 与标定的几何无关，所以只要 ppm 和画面尺寸，画布与六路现拼两条路完全一样。
/// 文案一律英文（图内文字规范），且画在缩放后的画面上，故坐标按 s 折算。
void draw_grid(cv::Mat& img, float ppm, float s) {
  const double step = double(ppm) * double(s);        // 1 m 在当前画面上的像素数
  if (step < 4.0) return;                             // 太密画满屏噪声，直接不画
  const cv::Scalar thin{90, 90, 90}, bold{0, 210, 210};
  const double fs = std::max(0.35, 0.5 * double(s));
  const double off = kGridMarginM * step;             // 纵向 0 m 线的内缩量
  // 竖线沿 x 量池长（从画布左边缘起，跨度即 50 m 池长）；横线沿 y 量池宽
  // （从池边起，两端各扣掉一条池岸）。两者只差起点与跨度，故共用一个 lambda。
  const double span[2] = {double(img.cols),
                          std::max(0.0, double(img.rows) - 2.0 * off)};
  char buf[32];
  auto axis = [&](bool vertical) {
    const double o = vertical ? 0.0 : off, len = span[vertical ? 0 : 1];
    for (int m = 0; double(m) * step <= len; ++m) {
      const int p = int(o + double(m) * step);
      const bool major = m % 5 == 0;
      const cv::Point a = vertical ? cv::Point{p, 0} : cv::Point{0, p};
      const cv::Point b = vertical ? cv::Point{p, img.rows} : cv::Point{img.cols, p};
      cv::line(img, a, b, major ? bold : thin, 1, cv::LINE_AA);
      if (!major || m == 0) continue;
      snprintf(buf, sizeof buf, "%d", m);
      const cv::Point at = vertical ? cv::Point{p + 3, int(off) + 14}
                                    : cv::Point{3, p - 3};
      cv::putText(img, buf, at, cv::FONT_HERSHEY_SIMPLEX, fs, bold, 1, cv::LINE_AA);
    }
  };
  axis(true);
  axis(false);
  snprintf(buf, sizeof buf, "%.1f x %.1f m", span[0] / step, span[1] / step);
  cv::putText(img, buf, {3, img.rows - 5}, cv::FONT_HERSHEY_SIMPLEX, fs, bold, 1,
              cv::LINE_AA);
}

/// 画标注。`s` 是画面缩放比（预览窗口 <1，落盘视频为 1）：坐标乘 s，字号与
/// 线宽跟着降但留下限，否则缩到 1/3 后文字糊成一团。
/// ov 是三个叠加层开关（预览里按 1/2/3 实时切换），关掉的层连计算都跳过。
void draw(cv::Mat& img, const FrameResult& fr, float kpt_thr, const Overlay& ov,
          float ppm, float s = 1.f) {
  if (ov.grid) draw_grid(img, ppm, s);
  const double fs = std::max(0.4, double(s));   // 字号下限，保证小窗仍可读
  const int    th = s < 0.6f ? 1 : 2;           // 线宽/字宽
  const int    r  = s < 0.6f ? 2 : 3;           // 关键点半径
  auto pt = [s](float x, float y) {
    return cv::Point(int(x * s), int(y * s));
  };

  for (const auto& p : fr.persons) {
    const auto col = id_color(p.track_id);
    if (ov.boxes) {
      cv::rectangle(img, pt(p.x1, p.y1), pt(p.x2, p.y2), col, th);

      char buf[96];
      if (std::isnan(p.speed))
        snprintf(buf, sizeof buf, "ID:%d S:%d", p.track_id, p.strokes);
      else
        snprintf(buf, sizeof buf, "ID:%d S:%d %.2fm/s", p.track_id, p.strokes,
                 p.speed);
      int base = 0;
      const auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, fs, th, &base);
      const int tx = int(p.x1 * s);
      const int ty = std::max(int(p.y1 * s) - 6, sz.height + 4);
      cv::rectangle(img, {tx, ty - sz.height - 4}, {tx + sz.width + 4, ty + 2},
                    col, cv::FILLED);
      cv::putText(img, buf, {tx + 2, ty}, cv::FONT_HERSHEY_SIMPLEX, fs,
                  {255, 255, 255}, th, cv::LINE_AA);
    }

    if (!ov.kpts) continue;
    for (const auto& e : kSkel) {
      if (p.scores[e[0]] < kpt_thr || p.scores[e[1]] < kpt_thr) continue;
      cv::line(img, pt(p.kpts[e[0] * 2], p.kpts[e[0] * 2 + 1]),
               pt(p.kpts[e[1] * 2], p.kpts[e[1] * 2 + 1]),
               {0, 255, 0}, th, cv::LINE_AA);
    }
    for (int k = 0; k < kNumKpts; ++k)
      if (p.scores[k] >= kpt_thr)
        cv::circle(img, pt(p.kpts[k * 2], p.kpts[k * 2 + 1]), r,
                   {0, 0, 255}, cv::FILLED, cv::LINE_AA);
  }
}

/// 实时预览窗口。画布 5002x2102 装不进任何屏幕，先缩小再画标注，所以窗口里的
/// 字号/线宽是原生清晰度而不是被一起缩糊。**窗口可自由缩放/最大化**：画面按当前
/// 客户区等比放到最大并居中（letterbox），不会只占左上一角。
///
/// 必须在**后处理线程**里构造与使用：Win32 的消息队列按线程分，窗口只能由
/// 创建它的线程 pump（waitKey 做的就是 pump），跨线程会不刷新甚至卡住。
/// 也因此不调 destroyWindow —— 该线程退出时 OS 自动销毁其窗口，而此时若从
/// main 线程 SendMessage(WM_CLOSE) 反而会等一个已死的线程。
class Preview {
 public:
  /// scale 只决定**初始**窗口大小（0 = 自适应到 1600x900 以内）；之后随窗口变。
  Preview(int w, int h, float scale)
      : s0_(scale > 0 ? scale : std::min(1.f, std::min(1600.f / float(w),
                                                      900.f / float(h)))) {
    // 必须 WINDOW_NORMAL：AUTOSIZE 把窗口钉在图像尺寸上，最大化后画面仍按原
    // 尺寸缩在左上角、四周留灰，且 resizeWindow 对它无效。
    cv::namedWindow(kWin, cv::WINDOW_NORMAL);
    cv::resizeWindow(kWin, int(float(w) * s0_), int(float(h) * s0_));
    printf("[Preview] 窗口 %dx%d (x%.3f)，可缩放/最大化（等比居中）；"
           "焦点在窗口上：q/ESC 退出，1 关键点  2 分析框  3 米制网格\n",
           int(float(w) * s0_), int(float(h) * s0_), s0_);
    fflush(stdout);
  }

  /// 返回 false 表示用户要求退出。raw 必须是**未画过**的原始帧。
  /// ov 按引用传：1/2/3 就地翻转，落盘那条路随即用同一份状态，
  /// 于是「窗口里看到的」与「写进 mp4 的」永远一致，无需第二份开关。
  bool show(const cv::Mat& raw, const FrameResult& fr, float kpt_thr,
            Overlay& ov, float ppm) {
    fit(raw.cols, raw.rows);
    // 直接缩到画板的画面区里：尺寸与类型都和 roi_ 一致，resize 不会重新分配，
    // 于是每帧只有一次缩放、零额外拷贝（黑边是建 pad_ 时就写好的）。
    cv::Mat view = pad_(roi_);
    cv::resize(raw, view, roi_.size(), 0, 0, cv::INTER_AREA);
    draw(view, fr, kpt_thr, ov, ppm, s_);   // 标注画在画面区内，溢不到黑边上
    cv::imshow(kWin, pad_);
    // waitKey 只在本线程（创建窗口的那个）有效，也是唯一能读到按键的地方。
    // 高位是修饰键与平台位，取低 8 位才能与字符比。
    const int k = cv::waitKey(1) & 0xff;
    switch (k) {
      case '1': toggle(ov.kpts,  "关键点"); break;
      case '2': toggle(ov.boxes, "分析框"); break;
      case '3': toggle(ov.grid,  "米制网格"); break;
      case 27: case 'q': case 'Q': return false;
      default: break;
    }
    return true;
  }

 private:
  /// 按当前客户区算「等比放到最大 + 居中」的画板与画面矩形；尺寸没变则沿用，
  /// 所以稳态下只多一次 getWindowImageRect。自己做 letterbox 而不用
  /// WINDOW_KEEPRATIO：后者在 Win32 后端与 Qt 后端行为不一致，自己算则处处相同。
  void fit(int w, int h) {
    const cv::Rect r = cv::getWindowImageRect(kWin);
    const int cw = r.width  > 0 ? r.width  : int(float(w) * s0_);
    const int ch = r.height > 0 ? r.height : int(float(h) * s0_);
    if (cw == pad_.cols && ch == pad_.rows) return;
    s_ = std::min(float(cw) / float(w), float(ch) / float(h));
    const int vw = std::max(1, int(float(w) * s_));
    const int vh = std::max(1, int(float(h) * s_));
    pad_ = cv::Mat::zeros(ch, cw, CV_8UC3);   // 黑边只在这里建一次，之后恒黑
    roi_ = cv::Rect((cw - vw) / 2, (ch - vh) / 2, vw, vh);
  }

  static void toggle(bool& flag, const char* name) {
    flag = !flag;
    printf("\n[Preview] %s %s\n", name, flag ? "开" : "关");
    fflush(stdout);
  }

  static constexpr const char* kWin = "swim_analyse";
  float    s0_;                         // 初始窗口缩放（--preview-scale）
  float    s_ = 1.f;                    // 当前画面缩放，随窗口变
  cv::Mat  pad_;                        // 整块客户区（画面 + 黑边）
  cv::Rect roi_;                        // 画面在 pad_ 里的位置（居中）
};

}  // namespace

int main(int argc, char** argv) try {
#ifdef _WIN32
  // 源码是 UTF-8（/utf-8 编译），控制台默认 936 会把中文输出成乱码
  SetConsoleOutputCP(CP_UTF8);
#endif
  Args a;
  if (!parse(argc, argv, a)) return 0;

  auto src = make_source(a);
  printf("[Input] %dx%d %.2f fps 总帧 %lld 后端 %s\n", src->width(), src->height(),
         src->fps(), static_cast<long long>(src->total()), src->backend());

  Pipeline pipe(a.opt, src->fps());

  std::unique_ptr<Writer> writer;
  if (!a.out.empty())
    writer = std::make_unique<Writer>(a.out, src->width(), src->height(), src->fps());

  // 逐帧 dump：定位框消失发生在哪一层，并把关键点一并导出供 Python 侧复算
  // （CSV: frame,id,x1,y1,x2,y2,conf, 之后 17 组 kN_x,kN_y,kN_s）
  std::ofstream dump;
  if (!a.dump.empty()) {
    dump.open(a.dump);
    // 默认 6 位有效数字在 5002 宽画布上只剩 0.01 px，够不上逐点复算：短 track 的
    // 位移本就只有几 px，速度对照会被舍入噪声抬到 1% 量级
    dump.precision(9);
    dump << "frame,id,x1,y1,x2,y2,conf";
    for (int k = 0; k < kNumKpts; ++k)
      dump << ",k" << k << "_x,k" << k << "_y,k" << k << "_s";
    dump << '\n';
  }

  std::map<int, std::pair<int, float>> summary;   // track -> (划水, 末速)
  int64_t n_person = 0;
  const double t0 = now_ms();
  double last_log = t0;
  // 在 Sink 里首帧惰性构造：窗口必须由 pump 它的线程（后处理线程）创建
  std::unique_ptr<Preview> preview;
  bool quit = false;

  pipe.run(*src, [&](const FrameResult& fr) {
    n_person += static_cast<int64_t>(fr.persons.size());
    for (const auto& p : fr.persons) summary[p.track_id] = {p.strokes, p.speed};
    if (dump.is_open())
      for (const auto& p : fr.persons) {
        dump << fr.index << ',' << p.track_id << ',' << p.x1 << ',' << p.y1
             << ',' << p.x2 << ',' << p.y2 << ',' << p.conf;
        for (int k = 0; k < kNumKpts; ++k)
          dump << ',' << p.kpts[k * 2] << ',' << p.kpts[k * 2 + 1] << ','
               << p.scores[k];
        dump << '\n';
      }

    if (fr.bgr) {
      cv::Mat img(fr.h, fr.w, CV_8UC3, const_cast<uint8_t*>(fr.bgr));
      // 首帧原样落盘必须最先做：下面的 draw 是在整帧上原地画的
      if (!a.dump_canvas.empty() && fr.index == 0) {
        SWIM_CHECK(cv::imwrite(a.dump_canvas, img), "无法写入 " + a.dump_canvas);
        printf("[Output] 首帧画布 %s (%dx%d)\n", a.dump_canvas.c_str(), fr.w, fr.h);
      }
      // 预览次之：它要的是未画过的原图（缩小后再画，字才不糊），
      // 而 writer 的 draw 是在整帧上原地画的。
      if (a.preview && !quit) {
        if (!preview) preview = std::make_unique<Preview>(fr.w, fr.h, a.preview_scale);
        if (!preview->show(img, fr, a.opt.kpt_thr, a.ov, a.opt.ppm)) {
          quit = true;
          printf("\n[Preview] 收到退出键，停止取帧（已入队的帧仍会处理完）\n");
          pipe.request_stop();
        }
      }
      if (writer) {
        draw(img, fr, a.opt.kpt_thr, a.ov, a.opt.ppm);
        writer->write(img);
      }
    }
    if (a.show_fps && now_ms() - last_log > 1000) {
      const double el = (now_ms() - t0) / 1000.0;
      // 源侧摘要（六路现拼才有：每路到帧率 + 丢/顶/重连）跟在后面。掉帧时能当场
      // 分清是「某一路网络」还是「下游算不过来」，见 FrameSource::status()。
      printf("\r[%.0fs] %lld 帧 %.1f fps  当前 %zu 人 %s  ", el,
             static_cast<long long>(fr.index + 1), (fr.index + 1) / el,
             fr.persons.size(), src->status().c_str());
      fflush(stdout);
      last_log = now_ms();
    }
  });

  const double total = now_ms() - t0;
  printf("\n");
  if (writer) writer->close();          // 等 ffmpeg 收尾，否则 mp4 缺 moov
  pipe.timers().report("GPU 流水线", total, pipe.frames_done());
  printf("[Summary] %lld 帧, 累计 %lld 人次, %zu 个 track\n",
         static_cast<long long>(pipe.frames_done()),
         static_cast<long long>(n_person), summary.size());
  int total_strokes = 0;
  for (const auto& [tid, v] : summary) total_strokes += v.first;
  printf("[Summary] 划水合计 %d 次\n", total_strokes);

  if (!a.json.empty()) {
    std::ofstream f(a.json);
    SWIM_CHECK(f.good(), "无法写入 " + a.json);
    f << "{\n";
    bool first = true;
    for (const auto& [tid, v] : summary) {
      if (!first) f << ",\n";
      first = false;
      f << "  \"" << tid << "\": {\"strokes\": " << v.first << ", \"speed\": ";
      // JSON 没有 NaN/Inf 字面量，非有限值统一写 null（否则下游解析直接失败）
      if (std::isfinite(v.second)) f << v.second; else f << "null";
      f << "}";
    }
    f << "\n}\n";
    SWIM_CHECK(f.good(), "写入未完成 " + a.json);
    printf("[Output] %s\n", a.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "[错误] %s\n", e.what());
  return 1;
}
