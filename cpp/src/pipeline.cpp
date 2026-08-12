#include "swim/pipeline.h"

#include <cstring>

namespace swim {
namespace {

/// 分配 device 缓冲的小助手，避免重复的 cast + 错误检查
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

}  // namespace

Pipeline::Pipeline(const PipelineOptions& opt, double fps)
    : opt_(opt), fps_(fps), chan_(opt.queue_depth),
      tracker_(opt.track_iou, opt.track_max_lost),
      metrics_(fps, opt.ppm) {
  SWIM_CUDA(cudaStreamCreate(&stream_));

  const int P = opt_.max_persons;
  det_ = TrtEngine::load(opt_.models_dir + "/detect.onnx",
                         opt_.engine_dir + "/detect.engine", "", 1, 1, 1, opt_.fp16);
  pose_ = TrtEngine::load(opt_.models_dir + "/pose.onnx",
                          opt_.engine_dir + "/pose.engine", "input",
                          1, 8, P, opt_.fp16);

  d_boxes_   = dev_alloc<float>(P * 4);
  d_conf_    = dev_alloc<float>(P);
  d_count_   = dev_alloc<int>(1);
  d_centers_ = dev_alloc<float>(P * 2);
  d_scales_  = dev_alloc<float>(P * 2);
  d_kpts_    = dev_alloc<float>(size_t(P) * kNumKpts * 2);
  d_scores_  = dev_alloc<float>(size_t(P) * kNumKpts);
  h_count_   = host_alloc<int>(1);
  h_boxes_   = host_alloc<float>(P * 4);
  h_conf_    = host_alloc<float>(P);
  h_kpts_    = host_alloc<float>(size_t(P) * kNumKpts * 2);
  h_scores_  = host_alloc<float>(size_t(P) * kNumKpts);

  printf("[Pipeline] detect engine 工作区 %.1f MB, pose engine 工作区 %.1f MB "
         "(batch<=%d)\n",
         det_->workspace_bytes() / 1e6, pose_->workspace_bytes() / 1e6, P);
}

Pipeline::~Pipeline() {
  for (void* p : {(void*)d_boxes_, (void*)d_conf_, (void*)d_count_,
                  (void*)d_centers_, (void*)d_scales_, (void*)d_kpts_,
                  (void*)d_scores_})
    if (p) cudaFree(p);
  for (void* p : {(void*)h_count_, (void*)h_boxes_, (void*)h_conf_,
                  (void*)h_kpts_, (void*)h_scores_})
    if (p) cudaFreeHost(p);
  for (auto* p : h_frames_) cudaFreeHost(p);
  if (stream_) cudaStreamDestroy(stream_);
}

int Pipeline::infer_frame(const GpuFrame& f) {
  const auto lb = Letterbox::make(f.w, f.h, opt_.detect_size);

  // 1~4 阶段之间没有 CPU 依赖，全部排进同一 stream 不做中间同步；
  // 只在需要读回框数时同步一次（pose 的 batch 取决于它）。
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
    launch_dedup_boxes(d_boxes_, d_conf_, d_count_, opt_.containment, stream_);
    SWIM_CUDA(cudaMemcpyAsync(h_count_, d_count_, sizeof(int),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaStreamSynchronize(stream_));
  }
  const int n = std::min(*h_count_, opt_.max_persons);
  if (n == 0) return 0;

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
    // 全链路唯一的实质 D2H：框 + 关键点，约 5 KB
    SWIM_CUDA(cudaMemcpyAsync(h_boxes_, d_boxes_, size_t(n) * 4 * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaMemcpyAsync(h_conf_, d_conf_, size_t(n) * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaMemcpyAsync(h_kpts_, d_kpts_,
                              size_t(n) * kNumKpts * 2 * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaMemcpyAsync(h_scores_, d_scores_,
                              size_t(n) * kNumKpts * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    SWIM_CUDA(cudaStreamSynchronize(stream_));
  }
  return n;
}

void Pipeline::infer_loop(FrameSource& src) {
  GpuFrame f;
  while (!stop_ && src.next(f)) {
    if (opt_.max_frames && f.index >= opt_.max_frames) break;
    const int n = infer_frame(f);

    Raw raw;
    raw.index = f.index;
    raw.persons.resize(n);
    for (int i = 0; i < n; ++i) {
      auto& p = raw.persons[i];
      p.x1 = h_boxes_[i * 4]; p.y1 = h_boxes_[i * 4 + 1];
      p.x2 = h_boxes_[i * 4 + 2]; p.y2 = h_boxes_[i * 4 + 3];
      p.conf = h_conf_[i];
      std::memcpy(p.kpts.data(), h_kpts_ + size_t(i) * kNumKpts * 2,
                  kNumKpts * 2 * sizeof(float));
      std::memcpy(p.scores.data(), h_scores_ + size_t(i) * kNumKpts,
                  kNumKpts * sizeof(float));
    }
    if (opt_.need_image) {
      // 渲染路径才付这次整帧 D2H；用环形缓冲避免覆写消费者仍在用的帧
      SWIM_TIME(timers_, "8_frame_d2h");
      if (h_frames_.empty()) {
        frame_bytes_ = size_t(f.w) * f.h * 3;
        h_frames_.resize(opt_.queue_depth + 1);
        for (auto& p : h_frames_) p = host_alloc<uint8_t>(frame_bytes_);
      }
      raw.bgr = h_frames_[frame_cursor_];
      frame_cursor_ = (frame_cursor_ + 1) % h_frames_.size();
      raw.w = f.w; raw.h = f.h;
      SWIM_CUDA(cudaMemcpyAsync(raw.bgr, f.data, frame_bytes_,
                                cudaMemcpyDeviceToHost, stream_));
      SWIM_CUDA(cudaStreamSynchronize(stream_));
    }
    if (!chan_.push(std::move(raw))) break;
  }
  chan_.close();
}

void Pipeline::post_loop(const Sink& sink) {
  Raw raw;
  while (chan_.pop(raw)) {
    {
      SWIM_TIME(timers_, "7_track_metrics");
      tracker_.update(raw.persons);
      for (auto& p : raw.persons) {
        metrics_.update(p.track_id, raw.index, p);
        p.strokes = metrics_.strokes(p.track_id);
        p.speed   = metrics_.speed(p.track_id);
      }
      // live 场景定期清理离场 track，避免内存单调增长
      if (raw.index % 300 == 0) metrics_.prune(raw.index, int64_t(fps_ * 10));
    }
    FrameResult fr;
    fr.index   = raw.index;
    fr.persons = std::move(raw.persons);
    fr.bgr     = raw.bgr;
    fr.w = raw.w; fr.h = raw.h;
    if (sink) sink(fr);
    frames_done_ = raw.index + 1;
  }
}

void Pipeline::run(FrameSource& src, const Sink& sink) {
  // 解码在 FrameSource 内部（含 H2D），与推理天然分离；
  // 这里只需两个线程：推理 + 后处理。
  std::thread post([&] { post_loop(sink); });
  infer_loop(src);
  post.join();
}

}  // namespace swim
