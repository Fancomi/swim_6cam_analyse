// CLI：离线视频 / live 流的实时游泳分析。
//
//   swim_analyse --input data/xxx.mp4 --models cpp/models [--out out.mp4] [--json r.json]
//   swim_analyse --input rtsp://...   --models cpp/models --show-fps
//
// 可视化用 OpenCV 在 CPU 侧画（只在需要落盘/预览时才 D2H 整帧；纯分析模式
// 不拷图像，全链路只回读关键点）。
#include <opencv2/opencv.hpp>

#include <cstring>
#include <fstream>
#include <map>
#include <string>

#include "swim/frame_source.h"
#include "swim/metrics.h"
#include "swim/pipeline.h"

using namespace swim;

namespace {

// COCO17 骨架（与 Python 版 draw.py 一致）
constexpr int kSkel[][2] = {{15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},
                            {5,6},{5,7},{7,9},{6,8},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4}};

cv::Scalar id_color(int id) {           // 与 Python 版同思路：按 id 稳定散列
  const uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
  return cv::Scalar((h >> 16) & 255, (h >> 8) & 255, h & 255);
}

struct Args {
  std::string input, models = "cpp/models", out, json, dump;
  int    max_frames = 0, detect_size = 640, max_persons = kMaxPersons;
  float  conf = 0.25f, kpt_thr = 0.4f, ppm = 100.f;
  bool   draw_kpts = true, show_fps = false, fp32 = false;
  DecoderPref decoder = DecoderPref::Auto;
};

bool parse(int argc, char** argv, Args& a) {
  auto need = [&](int& i) { SWIM_CHECK(i + 1 < argc, "参数缺少值"); return argv[++i]; };
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if      (k == "--input")       a.input = need(i);
    else if (k == "--models")      a.models = need(i);
    else if (k == "--out")         a.out = need(i);
    else if (k == "--json")        a.json = need(i);
    else if (k == "--dump")        a.dump = need(i);
    else if (k == "--max-frames")  a.max_frames = std::stoi(need(i));
    else if (k == "--detect-size") a.detect_size = std::stoi(need(i));
    else if (k == "--max-persons") a.max_persons = std::stoi(need(i));
    else if (k == "--conf")        a.conf = std::stof(need(i));
    else if (k == "--kpt-thr")     a.kpt_thr = std::stof(need(i));
    else if (k == "--ppm")         a.ppm = std::stof(need(i));
    else if (k == "--no-kpts")     a.draw_kpts = false;
    else if (k == "--show-fps")    a.show_fps = true;
    else if (k == "--fp32")        a.fp32 = true;
    else if (k == "--decoder") {
      const std::string v = need(i);
      a.decoder = v == "nvdec" ? DecoderPref::Nvdec
                : v == "cpu"   ? DecoderPref::Cpu : DecoderPref::Auto;
    } else if (k == "-h" || k == "--help") {
      printf("用法: swim_analyse --input <视频|流> [选项]\n"
             "  --models DIR       detect.onnx/pose.onnx 所在目录 (默认 cpp/models)\n"
             "  --out FILE         写出标注视频 (H.264)\n"
             "  --json FILE        写出每个 track 的划水次数与速度\n"
             "  --dump FILE        逐帧写出每个框(帧号,id,xyxy,conf)，用于与 Python 对照\n"
             "  --max-frames N     只处理前 N 帧\n"
             "  --detect-size N    detect 输入方形边长, 须与 onnx 一致 (默认 640)\n"
             "  --max-persons N    pose 最大人数，显存按此预留 (默认 40)\n"
             "  --conf F           检测置信度阈值 (默认 0.25)\n"
             "  --kpt-thr F        关键点置信度阈值 (默认 0.4)\n"
             "  --ppm F            画布每米像素数 (默认 100)\n"
             "  --decoder MODE     auto|nvdec|cpu (默认 auto)\n"
             "  --no-kpts          不画骨架\n"
             "  --show-fps         实时打印吞吐\n"
             "  --fp32             engine 用 fp32 (默认 fp16)\n");
      return false;
    } else {
      throw std::runtime_error("未知参数 " + k);
    }
  }
  SWIM_CHECK(!a.input.empty(), "必须指定 --input");
  return true;
}

/// H.264 写出器：优先 avc1，失败回退 mp4v。
class Writer {
 public:
  Writer(const std::string& path, int w, int h, double fps) {
    for (const char* cc : {"avc1", "mp4v"}) {
      w_.open(path, cv::VideoWriter::fourcc(cc[0], cc[1], cc[2], cc[3]), fps,
              {w, h});
      if (w_.isOpened()) { printf("[Writer] %s codec=%s\n", path.c_str(), cc); return; }
    }
    throw std::runtime_error("无法打开输出 " + path);
  }
  void write(const cv::Mat& m) { w_.write(m); }

 private:
  cv::VideoWriter w_;
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
  Args a;
  if (!parse(argc, argv, a)) return 0;

  auto src = FrameSource::open(a.input, a.decoder);
  printf("[Input] %dx%d %.2f fps 总帧 %lld 后端 %s\n", src->width(), src->height(),
         src->fps(), static_cast<long long>(src->total()), src->backend());

  PipelineOptions opt;
  opt.models_dir = opt.engine_dir = a.models;
  opt.detect_size = a.detect_size;
  opt.conf_thr = a.conf;
  opt.kpt_thr = a.kpt_thr;
  opt.max_persons = a.max_persons;
  opt.ppm = a.ppm;
  opt.fp16 = !a.fp32;
  opt.max_frames = a.max_frames;
  opt.need_image = !a.out.empty();      // 只有要写视频才付整帧 D2H
  Pipeline pipe(opt, src->fps());

  std::unique_ptr<Writer> writer;
  if (opt.need_image)
    writer = std::make_unique<Writer>(a.out, src->width(), src->height(), src->fps());

  // 逐帧 dump：定位框消失发生在哪一层（CSV: frame,id,x1,y1,x2,y2,conf）
  std::ofstream dump;
  if (!a.dump.empty()) {
    dump.open(a.dump);
    dump << "frame,id,x1,y1,x2,y2,conf\n";
  }

  std::map<int, std::pair<int, float>> summary;   // track -> (划水, 末速)
  int64_t n_person = 0;
  const double t0 = now_ms();
  double last_log = t0;

  pipe.run(*src, [&](const FrameResult& fr) {
    n_person += static_cast<int64_t>(fr.persons.size());
    for (const auto& p : fr.persons) summary[p.track_id] = {p.strokes, p.speed};
    if (dump.is_open())
      for (const auto& p : fr.persons)
        dump << fr.index << ',' << p.track_id << ',' << p.x1 << ',' << p.y1
             << ',' << p.x2 << ',' << p.y2 << ',' << p.conf << '\n';

    if (writer && fr.bgr) {
      cv::Mat img(fr.h, fr.w, CV_8UC3, const_cast<uint8_t*>(fr.bgr));
      draw(img, fr, a.kpt_thr, a.draw_kpts);
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
  pipe.timers().report("GPU 流水线", total, pipe.frames_done());
  printf("[Summary] %lld 帧, 累计 %lld 人次, %zu 个 track\n",
         static_cast<long long>(pipe.frames_done()),
         static_cast<long long>(n_person), summary.size());
  int total_strokes = 0;
  for (const auto& [tid, v] : summary) total_strokes += v.first;
  printf("[Summary] 划水合计 %d 次\n", total_strokes);

  if (!a.json.empty()) {
    std::ofstream f(a.json);
    f << "{\n";
    bool first = true;
    for (const auto& [tid, v] : summary) {
      if (!first) f << ",\n";
      first = false;
      f << "  \"" << tid << "\": {\"strokes\": " << v.first << ", \"speed\": ";
      if (std::isnan(v.second)) f << "null"; else f << v.second;
      f << "}";
    }
    f << "\n}\n";
    printf("[Output] %s\n", a.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  fprintf(stderr, "[错误] %s\n", e.what());
  return 1;
}
