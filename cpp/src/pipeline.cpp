#include "swim/pipeline.h"

#include <algorithm>
#include <cstring>

namespace swim {
namespace {

/// 分配 device / 锁页 host 缓冲的小助手，避免重复的 cast + 错误检查
template <typename T>
T* dev_alloc(size_t n) {
  void* p = nullptr;
  SWIM_CUDA(cudaMalloc(&p, n * sizeof(T)));
  return static_cast<T*>(p);
}
template <typename T>
T* host_alloc(size_t n) {
  void* p = nullptr;
  SWIM_CUDA(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));
  return static_cast<T*>(p);
}

/// 计时用不上，禁掉可省一次 GPU 时间戳写入
cudaEvent_t make_event() {
  cudaEvent_t e = nullptr;
  SWIM_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}

}  // namespace

void PipelineOptions::validate() const {
  SWIM_CHECK(max_persons >= 1 && max_persons <= kMaxPersons,
             "--max-persons 须在 1.." + std::to_string(kMaxPersons));
  SWIM_CHECK(detect_size >= 64 && detect_size % 32 == 0,
             "--detect-size 须为 >=64 且 32 的倍数（且与 detect.onnx 一致）");
  SWIM_CHECK(conf_thr >= 0.f && conf_thr < 1.f, "--conf 须在 [0,1)");
  SWIM_CHECK(kpt_thr >= 0.f && kpt_thr < 1.f, "--kpt-thr 须在 [0,1)");
  SWIM_CHECK(containment > 0.f && containment <= 1.f, "包含率阈值须在 (0,1]");
  SWIM_CHECK(track_iou > 0.f && track_iou <= 1.f, "跟踪 IoU 阈值须在 (0,1]");
  SWIM_CHECK(track_max_lost >= 1, "track_max_lost 须 >=1");
  SWIM_CHECK(ppm > 0.f, "--ppm 须为正（否则速度为 inf）");
  SWIM_CHECK(valid_stroke(stroke_type),
             "--stroke-type 只能是: " + std::string(kStrokeTypes));
  SWIM_CHECK(queue_depth >= 1, "queue_depth 须 >=1");
  SWIM_CHECK(max_frames >= 0, "--max-frames 不能为负");
}

Pipeline::Pipeline(const PipelineOptions& opt, double fps)
    : opt_(opt), fps_(fps), chan_(opt.queue_depth),
      tracker_(opt.track_iou, opt.track_max_lost),
      metrics_(fps, opt.ppm, opt.kpt_thr,
               MetricsTracker::split_sides(opt.stroke_type), opt.signal) {
  opt_.validate();
  SWIM_CHECK(fps_ > 0.0, "输入帧率非法");
  SWIM_CUDA(cudaStreamCreate(&stream_));
  ev_count_ = make_event();

  const int P = opt_.max_persons;
  det_ = TrtEngine::load(opt_.models_dir + "/detect.onnx",
                         opt_.engine_dir + "/detect.engine", "", 1, 1, 1, opt_.fp16);
  // opt batch 取常见人数 8，但必须夹在 [1, P] 内：opt > max 会让
  // OptimizationProfile 非法，buildSerializedNetwork 直接返回空
  // （--max-persons 5 时曾表现为"engine 构建失败"）
  pose_ = TrtEngine::load(opt_.models_dir + "/pose.onnx",
                          opt_.engine_dir + "/pose.engine", "input",
                          1, std::min(8, P), P, opt_.fp16);

  // 形状契约：kernel 按 CLI/常量写，engine 按 ONNX 分配，两者不校验就会越界。
  // detect 输入是 ONNX 里写死的静态形状，engine 缓存的身份戳只跟 ONNX 绑定，
  // 不含 --detect-size —— 所以这里必须当场比对（--detect-size 1280 曾意味着
  // 往 640² 的缓冲写 1280² 的数据）。
  const auto& din = det_->input().dims;
  SWIM_CHECK(din.nbDims == 4 && int(din.d[2]) == opt_.detect_size &&
                 int(din.d[3]) == opt_.detect_size,
             "--detect-size " + std::to_string(opt_.detect_size) +
                 " 与 detect.onnx 的输入 " + std::to_string(din.d[3]) + "x" +
                 std::to_string(din.d[2]) + " 不一致（改这个参数不会重建 engine，"
                 "请改传正确的值或重新导出 ONNX）");
  const auto& dout = det_->output().dims;
  SWIM_CHECK(dout.nbDims == 3 && int(dout.d[1]) == kMaxDet && int(dout.d[2]) == 6,
             "detect.onnx 输出应为 [1," + std::to_string(kMaxDet) +
                 ",6]（max_det 变了就要同步 kMaxDet），实际第 1/2 维为 " +
                 std::to_string(dout.d[1]) + "/" + std::to_string(dout.d[2]));
  // simcc_x/simcc_y 靠 IO 索引区分，TRT 未承诺该顺序等于 ONNX 输出顺序；
  // 两条轴的长度不同（384 vs 512），据此就能确认没被互换
  SWIM_CHECK(pose_->num_outputs() == 2 &&
                 int(pose_->output(0).dims.d[2]) == int(kPoseW * kSimccRatio) &&
                 int(pose_->output(1).dims.d[2]) == int(kPoseH * kSimccRatio),
             "pose.onnx 的两个输出不是 simcc_x[.,384] + simcc_y[.,512]，"
             "顺序可能被 TRT 调换，会导致 x/y 轴互换");

  // boxes/conf/kpts/scores 连成一块：D2H 从 4 次合成 1 次（约 9 KB，一次就够）
  blob_n_ = size_t(P) * (5 + 3 * kNumKpts);
  d_blob_ = dev_alloc<float>(blob_n_);
  d_boxes_  = d_blob_;
  d_conf_   = d_boxes_ + size_t(P) * 4;
  d_kpts_   = d_conf_  + size_t(P);
  d_scores_ = d_kpts_  + size_t(P) * kNumKpts * 2;
  d_count_   = dev_alloc<int>(1);
  d_centers_ = dev_alloc<float>(size_t(P) * 2);
  d_scales_  = dev_alloc<float>(size_t(P) * 2);
  h_count_ = host_alloc<int>(1);

  // host 侧回读环：容量 = 队列 + 消费者手上 1 + 生产者正在写 1。
  // 有了环，推理线程排完 D2H 就能去做下一帧，不必等拷贝完成（跨帧重叠）。
  blob_ring_.resize(size_t(opt_.queue_depth) + 2);
  for (auto& s : blob_ring_) {
    s.host  = host_alloc<float>(blob_n_);
    s.ready = make_event();
  }

  printf("[Pipeline] detect 工作区 %.1f MB, pose 工作区 %.1f MB (batch<=%d), "
         "回读 %.1f KB/帧 x%zu\n",
         det_->workspace_bytes() / 1e6, pose_->workspace_bytes() / 1e6, P,
         blob_n_ * sizeof(float) / 1e3, blob_ring_.size());
}

Pipeline::~Pipeline() {
  for (void* p : {(void*)d_blob_, (void*)d_count_, (void*)d_centers_,
                  (void*)d_scales_})
    if (p) cudaFree(p);
  if (h_count_) cudaFreeHost(h_count_);
  for (auto& s : blob_ring_) {
    if (s.host) cudaFreeHost(s.host);
    if (s.ready) cudaEventDestroy(s.ready);
  }
  for (auto& s : frame_ring_) {
    if (s.host) cudaFreeHost(s.host);
    if (s.ready) cudaEventDestroy(s.ready);
  }
  if (ev_count_) cudaEventDestroy(ev_count_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Pipeline::alloc_frame_ring(int w, int h) {
  SWIM_CHECK(w > 0 && h > 0, "整帧回读需要有效的画面尺寸");
  frame_bytes_ = size_t(w) * h * 3;
  frame_ring_.resize(size_t(opt_.queue_depth) + 2);
  for (auto& s : frame_ring_) {
    s.host  = host_alloc<uint8_t>(frame_bytes_);
    s.ready = make_event();
  }
  printf("[Pipeline] 整帧回读环 %zu x %.1f MB\n", frame_ring_.size(),
         frame_bytes_ / 1e6);
}

int Pipeline::infer_frame(const GpuFrame& f, Slot<float>& blob) {
  const auto lb = Letterbox::make(f.w, f.h, opt_.detect_size);

  // 1~4 阶段之间没有 CPU 依赖，全部排进同一 stream 不做中间同步；
  // 只在需要读回框数时等一次（pose 的 batch 取决于它，无法避免）。
  {
    SWIM_TIME(timers_, "1_pre+detect");
    launch_preprocess(f.data, f.w, f.h,
                      static_cast<__half*>(det_->input().ptr),
                      opt_.detect_size, lb, stream_);
    det_->enqueue(stream_);
    // yolo26 是 end2end（NMS 已内置），这里只做阈值筛选与坐标还原
    launch_filter_boxes(static_cast<const float*>(det_->output().ptr), kMaxDet,
                        opt_.conf_thr, lb, f.w, f.h, opt_.max_persons,
                        d_boxes_, d_conf_, d_count_, stream_);
    // 再去掉被大框包住的重复框：IoU-NMS 抓不到"一大一小套同一人"，
    // 残留会让跟踪频繁新建 ID（实测 track 数 36 -> 163）
    launch_dedup_boxes(d_boxes_, d_conf_, d_count_, opt_.max_persons,
                       opt_.containment, stream_);
    SWIM_CUDA(cudaMemcpyAsync(h_count_, d_count_, sizeof(int),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaEventRecord(ev_count_, stream_));
    SWIM_CUDA(cudaEventSynchronize(ev_count_));
  }
  const int n = *h_count_;                 // kernel 内已按 max_persons 夹紧
  if (n <= 0) return 0;

  {
    SWIM_TIME(timers_, "2_crop+pose+decode");
    launch_crop_affine(f.data, f.w, f.h, d_boxes_, n,
                       static_cast<__half*>(pose_->input().ptr),
                       d_centers_, d_scales_, stream_);
    pose_->set_batch(n);
    pose_->enqueue(stream_);
    const auto& sx = pose_->output(0);
    const auto& sy = pose_->output(1);
    launch_simcc_decode(static_cast<const float*>(sx.ptr),
                        static_cast<const float*>(sy.ptr), n, kNumKpts,
                        int(sx.dims.d[sx.dims.nbDims - 1]),
                        int(sy.dims.d[sy.dims.nbDims - 1]),
                        d_centers_, d_scales_, d_kpts_, d_scores_, stream_);
    // 全链路唯一的实质 D2H：框 + 关键点一并取回（约 9 KB）。
    // 不在这里等：记事件交给后处理线程，推理线程直接排下一帧（跨帧重叠）。
    SWIM_CUDA(cudaMemcpyAsync(blob.host, d_blob_, blob_n_ * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaEventRecord(blob.ready, stream_));
  }
  return n;
}

void Pipeline::infer_loop(FrameSource& src) {
  GpuFrame f;
  for (;;) {
    {
      // 单独计时：解码线程供不上帧时，这一项就是真实瓶颈。没有它只能靠
      // 「端到端减各 GPU 段」倒推，而两者的差额还混着队列背压，读数会骗人。
      SWIM_TIME(timers_, "0_src_wait");
      if (stop_ || !src.next(f)) break;
    }
    if (opt_.max_frames && f.index >= opt_.max_frames) break;
    auto& blob = blob_ring_[cursor_ % blob_ring_.size()];
    const int n = infer_frame(f, blob);

    Raw raw;
    raw.index = f.index;
    raw.n     = n;
    raw.blob  = n > 0 ? &blob : nullptr;
    if (opt_.need_image) {
      // 渲染路径才付这次整帧 D2H；同样只记事件，由消费者等
      SWIM_TIME(timers_, "8_frame_d2h");
      auto& fr = frame_ring_[cursor_ % frame_ring_.size()];
      SWIM_CUDA(cudaMemcpyAsync(fr.host, f.data, frame_bytes_,
                                cudaMemcpyDeviceToHost, stream_));
      SWIM_CUDA(cudaEventRecord(fr.ready, stream_));
      raw.frame = &fr;
      raw.w = f.w; raw.h = f.h;
    }
    ++cursor_;
    if (!chan_.push(std::move(raw))) break;
  }
  chan_.close();
}

void Pipeline::post_loop(const Sink& sink) {
  Raw raw;
  while (chan_.pop(raw)) {
    FrameResult fr;
    fr.index = raw.index;
    {
      SWIM_TIME(timers_, "7_track_metrics");
      if (raw.blob) {
        // 等这一格的 D2H 落地（GPU 早已在跑后面的帧）
        SWIM_CUDA(cudaEventSynchronize(raw.blob->ready));
        const float* b  = raw.blob->host;
        const int    P  = opt_.max_persons;
        const float* cf = b + size_t(P) * 4;
        const float* kp = cf + size_t(P);
        const float* sc = kp + size_t(P) * kNumKpts * 2;
        fr.persons.resize(raw.n);
        for (int i = 0; i < raw.n; ++i) {
          auto& p = fr.persons[i];
          p.x1 = b[i * 4];     p.y1 = b[i * 4 + 1];
          p.x2 = b[i * 4 + 2]; p.y2 = b[i * 4 + 3];
          p.conf = cf[i];
          std::memcpy(p.kpts.data(), kp + size_t(i) * kNumKpts * 2,
                      kNumKpts * 2 * sizeof(float));
          std::memcpy(p.scores.data(), sc + size_t(i) * kNumKpts,
                      kNumKpts * sizeof(float));
        }
      }
      tracker_.update(fr.persons);
      for (auto& p : fr.persons) {
        metrics_.update(p.track_id, raw.index, p);
        p.strokes = metrics_.strokes(p.track_id);
        p.speed   = metrics_.speed(p.track_id);
      }
      // live 场景定期清理离场 track，避免内存单调增长
      if (raw.index % 300 == 0) metrics_.prune(raw.index, int64_t(fps_ * 10));
    }
    if (raw.frame) {
      SWIM_CUDA(cudaEventSynchronize(raw.frame->ready));
      fr.bgr = raw.frame->host;
      fr.w = raw.w; fr.h = raw.h;
    }
    if (sink) sink(fr);
    frames_done_ = raw.index + 1;
  }
}

void Pipeline::run(FrameSource& src, const Sink& sink) {
  if (opt_.need_image && frame_ring_.empty())
    alloc_frame_ring(src.width(), src.height());   // 提前分配，热路径无判空

  // 解码在 FrameSource 内部（含 H2D），与推理天然分离；
  // 这里只需两个线程：推理 + 后处理。任一侧抛异常都要让对侧退出，
  // 否则 close 不到、join 会永久阻塞。
  std::thread post([&] {
    try {
      post_loop(sink);
    } catch (...) {
      err_ = std::current_exception();
      stop_ = true;
      chan_.close();          // 唤醒可能正阻塞在 push 的推理线程
    }
  });
  std::exception_ptr infer_err;
  try {
    infer_loop(src);
  } catch (...) {
    infer_err = std::current_exception();
    chan_.close();
  }
  post.join();
  if (infer_err) std::rethrow_exception(infer_err);
  if (err_)      std::rethrow_exception(err_);
}

}  // namespace swim
