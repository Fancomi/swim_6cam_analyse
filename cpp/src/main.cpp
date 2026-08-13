// CLI：离线视频 / live 流的实时游泳分析。
//
//   swim_analyse --input data/xxx.mp4 --models cpp/models [--out out.mp4] [--json r.json]
//   swim_analyse --input rtsp://...   --models cpp/models --show-fps
//
// 可视化用 OpenCV 在 CPU 侧画（只在需要落盘/预览时才 D2H 整帧；纯分析模式
// 不拷图像，全链路只回读关键点）。
#include <opencv2/opencv.hpp>

#ifdef _WIN32
// windows.h 默认定义 min/max 宏，会打断 std::max / cv:: 里的模板调用
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

#include "swim/frame_source.h"
#include "swim/metrics.h"
#include "swim/pipeline.h"

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

struct Args {
  PipelineOptions opt;                  // 参数默认值只在 pipeline.h 定义一份
  std::string input, out, json, dump;
  bool        draw_kpts = true, show_fps = false;
  DecoderPref decoder   = DecoderPref::Auto;
};

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

bool parse(int argc, char** argv, Args& a) {
  auto need = [&](int& i) { SWIM_CHECK(i + 1 < argc, "参数缺少值"); return argv[++i]; };
  auto num  = [&](int& i, const std::string& k, double lo, double hi) {
    return static_cast<float>(parse_num(need(i), k, lo, hi, false));
  };
  auto integer = [&](int& i, const std::string& k, long lo, long hi) {
    return static_cast<long>(parse_num(need(i), k, double(lo), double(hi), true));
  };
  auto& o = a.opt;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if      (k == "--input")       a.input = need(i);
    else if (k == "--models")      o.models_dir = o.engine_dir = need(i);
    else if (k == "--out")         a.out = need(i);
    else if (k == "--json")        a.json = need(i);
    else if (k == "--dump")        a.dump = need(i);
    else if (k == "--max-frames")  o.max_frames  = integer(i, k, 0, 1 << 30);
    else if (k == "--detect-size") o.detect_size = int(integer(i, k, 64, 4096));
    else if (k == "--max-persons") o.max_persons = int(integer(i, k, 1, kMaxPersons));
    else if (k == "--conf")        o.conf_thr = num(i, k, 0.0, 0.999);
    else if (k == "--kpt-thr")     o.kpt_thr  = num(i, k, 0.0, 0.999);
    else if (k == "--ppm")         o.ppm      = num(i, k, 1e-3, 1e5);
    else if (k == "--stroke-type") o.stroke_type = need(i);
    else if (k == "--signal") {
      const std::string v = need(i);
      SWIM_CHECK(v == "elbow_angle" || v == "wrist_x_head",
                 "--signal 只能是 elbow_angle|wrist_x_head");
      o.signal = v == "wrist_x_head" ? StrokeSignal::WristXHead
                                     : StrokeSignal::ElbowAngle;
    }
    else if (k == "--no-kpts")     a.draw_kpts = false;
    else if (k == "--show-fps")    a.show_fps = true;
    else if (k == "--fp32")        o.fp16 = false;
    else if (k == "--decoder") {
      const std::string v = need(i);
      a.decoder = v == "nvdec" ? DecoderPref::Nvdec
                : v == "cpu"   ? DecoderPref::Cpu : DecoderPref::Auto;
    } else if (k == "-h" || k == "--help") {
      printf("用法: swim_analyse --input <视频|流> [选项]\n"
             "  --models DIR       detect.onnx/pose.onnx 所在目录 (默认 cpp/models)\n"
             "  --out FILE         写出标注视频 (H.264)\n"
             "  --json FILE        写出每个 track 的划水次数与速度\n"
             "  --dump FILE        逐帧写出每个框(帧号,id,xyxy,conf)+17 个关键点\n"
             "                     (kx,ky,ks)，用于与 Python 对照\n"
             "  --max-frames N     只处理前 N 帧\n"
             "  --detect-size N    detect 输入方形边长, 须与 onnx 一致 (默认 640)\n"
             "  --max-persons N    pose 最大人数，显存按此预留 (默认 %d, 上限 %d)\n"
             "  --conf F           检测置信度阈值 (默认 0.25)\n"
             "  --kpt-thr F        关键点置信度阈值 (默认 0.4)\n"
             "  --ppm F            画布每米像素数 (默认 100)\n"
             "  --stroke-type S    泳姿，决定左右是否分开计数 (默认 freestyle)\n"
             "                     可选: %s\n"
             "  --signal S         划水信号 elbow_angle|wrist_x_head (默认 elbow_angle)\n"
             "  --decoder MODE     auto|cpu (默认 auto=ffmpeg 管道，探测失败回退 OpenCV;\n"
             "                     cpu=强制 OpenCV; nvdec 已废弃，等价 auto)\n"
             "  --no-kpts          不画骨架\n"
             "  --show-fps         实时打印吞吐\n"
             "  --fp32             engine 算子用 fp32 (默认 fp16)。注意 ONNX 若是\n"
             "                     fp16 权重导出的，这只放宽算子精度，不等于真 fp32\n",
             kMaxPersons, kMaxPersons, kStrokeTypes);
      return false;
    } else {
      throw std::runtime_error("未知参数 " + k);
    }
  }
  SWIM_CHECK(!a.input.empty(), "必须指定 --input");
  o.need_image = !a.out.empty();        // 只有要写视频才付整帧 D2H
  o.validate();
  return true;
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
#ifdef _WIN32
    pipe_ = _popen(cmd, "wb");            // 必须 "wb"：文本模式会把 0x0A 换成 CRLF
#else
    pipe_ = popen(cmd, "w");
#endif
    SWIM_CHECK(pipe_, "无法打开输出 " + path + "（OpenCV 与 ffmpeg 管道均失败）");
    bytes_ = size_t(w) * h * 3;
    printf("[Writer] %s codec=libx264 (ffmpeg 管道)\n", path.c_str());
  }

  /// 析构只收尸不报错：正常路径由 main 显式调 close() 检查退出码，
  /// 异常路径已经在抛别的东西了，析构里再抛会直接 terminate。
  ~Writer() { reap(); }

  void write(const cv::Mat& m) {
    if (!pipe_) { w_.write(m); return; }
    SWIM_CHECK(m.isContinuous(), "帧数据非连续，无法直接写管道");
    ++frames_;
    if (fwrite(m.data, 1, bytes_, pipe_) == bytes_) return;
    // 短写只说明管道断了，真正的原因在 ffmpeg 的退出码里 —— 先收尸拿到它，
    // 否则只剩"写失败"这种没有指向性的报错，还会漏掉断在第几帧
    throw std::runtime_error("写 ffmpeg 管道失败：第 " + std::to_string(frames_) +
                             " 帧短写，ffmpeg 退出码 " + std::to_string(reap()) +
                             "（其 stderr 见上方；PATH 里有 ffmpeg 吗？）");
  }

  /// 显式关闭：管道要等 ffmpeg 落完 moov 才算写完，别拖到 main 结束后。
  /// 退出码非 0 必须报出来，否则视频缺 moov 而程序声称成功。
  void close() {
    const int code = reap();
    SWIM_CHECK(code == 0, "ffmpeg 编码进程退出码 " + std::to_string(code) +
                              "，视频可能不完整（已写 " +
                              std::to_string(frames_) + " 帧）");
  }

 private:
  /// 关管道并等 ffmpeg 结束，返回其退出码；幂等，非管道模式返回 0。
  int reap() {
    if (!pipe_) return 0;
    FILE* p = pipe_;
    pipe_ = nullptr;
#ifdef _WIN32
    return _pclose(p);
#else
    const int st = pclose(p);
    return WIFEXITED(st) ? WEXITSTATUS(st) : st;
#endif
  }

  cv::VideoWriter w_;
  FILE*   pipe_  = nullptr;
  size_t  bytes_ = 0;
  int64_t frames_ = 0;
};

void draw(cv::Mat& img, const FrameResult& fr, float kpt_thr, bool draw_kpts) {
  for (const auto& p : fr.persons) {
    const auto col = id_color(p.track_id);
    cv::rectangle(img, {int(p.x1), int(p.y1)}, {int(p.x2), int(p.y2)}, col, 2);

    char buf[96];
    if (std::isnan(p.speed))
      snprintf(buf, sizeof buf, "ID:%d S:%d", p.track_id, p.strokes);
    else
      snprintf(buf, sizeof buf, "ID:%d S:%d %.2fm/s", p.track_id, p.strokes,
               p.speed);
    int base = 0;
    const auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &base);
    const int ty = std::max(int(p.y1) - 6, sz.height + 4);
    cv::rectangle(img, {int(p.x1), ty - sz.height - 4},
                  {int(p.x1) + sz.width + 4, ty + 2}, col, cv::FILLED);
    cv::putText(img, buf, {int(p.x1) + 2, ty}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                {255, 255, 255}, 2, cv::LINE_AA);

    if (!draw_kpts) continue;
    for (const auto& e : kSkel) {
      if (p.scores[e[0]] < kpt_thr || p.scores[e[1]] < kpt_thr) continue;
      cv::line(img, {int(p.kpts[e[0] * 2]), int(p.kpts[e[0] * 2 + 1])},
               {int(p.kpts[e[1] * 2]), int(p.kpts[e[1] * 2 + 1])},
               {0, 255, 0}, 2, cv::LINE_AA);
    }
    for (int k = 0; k < kNumKpts; ++k)
      if (p.scores[k] >= kpt_thr)
        cv::circle(img, {int(p.kpts[k * 2]), int(p.kpts[k * 2 + 1])}, 3,
                   {0, 0, 255}, cv::FILLED, cv::LINE_AA);
  }
}

}  // namespace

int main(int argc, char** argv) try {
#ifdef _WIN32
  // 源码是 UTF-8（/utf-8 编译），控制台默认 936 会把中文输出成乱码
  SetConsoleOutputCP(CP_UTF8);
#endif
  Args a;
  if (!parse(argc, argv, a)) return 0;

  auto src = FrameSource::open(a.input, a.decoder);
  printf("[Input] %dx%d %.2f fps 总帧 %lld 后端 %s\n", src->width(), src->height(),
         src->fps(), static_cast<long long>(src->total()), src->backend());

  Pipeline pipe(a.opt, src->fps());

  std::unique_ptr<Writer> writer;
  if (a.opt.need_image)
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

    if (writer && fr.bgr) {
      cv::Mat img(fr.h, fr.w, CV_8UC3, const_cast<uint8_t*>(fr.bgr));
      draw(img, fr, a.opt.kpt_thr, a.draw_kpts);
      writer->write(img);
    }
    if (a.show_fps && now_ms() - last_log > 1000) {
      const double el = (now_ms() - t0) / 1000.0;
      printf("\r[%.0fs] %lld 帧 %.1f fps  当前 %zu 人   ", el,
             static_cast<long long>(fr.index + 1), (fr.index + 1) / el,
             fr.persons.size());
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
